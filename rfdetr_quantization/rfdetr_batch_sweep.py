#!/usr/bin/env python3
"""Batch-size sweep for RF-DETR INT8 vs FP16 TensorRT engines.

Experiment #1 from the "no INT8 speedup" investigation: the broad INT8 engine
is *slower* than FP16 at batch 1 (0.85x) because rfdetr-nano's transformer
GEMMs are memory/launch-bound at batch 1 — the Q/DQ reformat tax beats the
INT8 GEMM saving. INT8 tensor cores only win once the GEMMs are compute-bound,
which happens as the batch grows.

This script re-exports both ONNX graphs with a **dynamic batch axis**, builds
dynamic-shape TensorRT engines (one INT8, one FP16), and benchmarks each at a
sweep of batch sizes. If INT8 overtakes FP16 at batch >= 8, the quantization
pipeline is proven correct and the batch-1 regression is purely an
arithmetic-intensity artifact.

Usage::

    python rfdetr_quantization/rfdetr_batch_sweep.py \\
        --qat-ckpt runs/modelopt_qat/rfdetr_nano/rfdetr_nano_qat.pth \\
        --batches 1,4,8,16,32
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rfdetr_quantization.nvidia_modelopt_rfdetr_qat import (  # noqa: E402
    _RFDETRTracingWrapper,
    _patch_interpolate_no_antialias,
    load_rfdetr,
)
from rfdetr_quantization.rfdetr_trt import TRTRunner, load_engine  # noqa: E402


# ---------------------------------------------------------------------------
# Dynamic-batch ONNX re-export
# ---------------------------------------------------------------------------

def export_dynamic_onnx(
    model_name: str,
    dst: Path,
    *,
    imgsz: int,
    device: torch.device,
    opset: int,
    qat_ckpt: Path | None,
) -> Path:
    """Re-export RF-DETR to ONNX with a dynamic batch axis.

    ``qat_ckpt`` set -> restore the ModelOpt-quantized state and emit a Q/DQ
    ONNX (INT8). ``qat_ckpt`` None -> plain FP32 ONNX (the FP16 baseline).
    """
    import modelopt.torch.opt as mto
    from modelopt.torch._deploy.utils.torch_onnx import (
        OnnxBytes,
        get_onnx_bytes_and_metadata,
    )

    model, _, _ = load_rfdetr(model_name, device)
    if qat_ckpt is not None:
        print(f"  [export] restoring ModelOpt state from {qat_ckpt}")
        mto.restore(model, str(qat_ckpt))
    model.eval()

    wrapper = _RFDETRTracingWrapper(model).to(device).eval()
    dummy = torch.randn(2, 3, imgsz, imgsz, device=device)  # batch 2 -> traces batch as dynamic
    dynamic_axes = {"images": {0: "batch"}}

    print(f"  [export] exporting {model_name} (dynamic batch) -> {dst.name}")
    with _patch_interpolate_no_antialias():
        onnx_bytes, _ = get_onnx_bytes_and_metadata(
            wrapper, dummy, dynamic_axes=dynamic_axes, onnx_opset=opset
        )
    onnx_model = OnnxBytes.from_bytes(onnx_bytes).get_onnx_model_file_bytes()
    dst.write_bytes(onnx_model)

    import onnx

    g = onnx.load(str(dst)).graph
    n_q = sum(n.op_type == "QuantizeLinear" for n in g.node)
    in0 = g.input[0]
    dims = [d.dim_param or d.dim_value for d in in0.type.tensor_type.shape.dim]
    print(f"  [export] wrote {dst} input={in0.name}{dims} Q-nodes={n_q}")
    return dst


# ---------------------------------------------------------------------------
# Dynamic-shape engine build
# ---------------------------------------------------------------------------

def build_dynamic_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    int8: bool,
    imgsz: int,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    workspace_gb: int = 4,
) -> Path:
    """Build a TensorRT engine with a dynamic-batch optimization profile."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n{errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)

    input_name = network.get_input(0).name
    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        min=(min_batch, 3, imgsz, imgsz),
        opt=(opt_batch, 3, imgsz, imgsz),
        max=(max_batch, 3, imgsz, imgsz),
    )
    config.add_optimization_profile(profile)

    precision = "INT8" if int8 else "FP16"
    print(f"  [trt] building dynamic {precision} engine from {onnx_path.name} "
          f"(batch {min_batch}-{max_batch}, {network.num_layers} layers) ...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")
    engine_path.write_bytes(serialized)
    print(f"  [trt] wrote {engine_path} ({engine_path.stat().st_size/1e6:.1f} MB) "
          f"in {time.time()-t0:.1f}s")
    return engine_path


# ---------------------------------------------------------------------------
# Benchmark at a given batch size
# ---------------------------------------------------------------------------

def benchmark_at_batch(
    runner: TRTRunner,
    batch: int,
    imgsz: int,
    device: torch.device,
    *,
    warmup: int = 30,
    iters: int = 200,
) -> dict:
    """GPU-compute latency for one fixed batch size (CUDA events)."""
    dummy = torch.randn(batch, 3, imgsz, imgsz, device=device)
    for _ in range(warmup):
        runner.infer(dummy)
    torch.cuda.synchronize(device)

    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        runner.infer(dummy)
        end.record()
        torch.cuda.synchronize(device)
        times_ms.append(start.elapsed_time(end))

    times_ms.sort()
    mean = statistics.fmean(times_ms)
    return {
        "batch": batch,
        "mean_ms": mean,
        "median_ms": times_ms[len(times_ms) // 2],
        "p99_ms": times_ms[min(len(times_ms) - 1, int(0.99 * len(times_ms)))],
        "per_image_ms": mean / batch,
        "throughput_fps": 1000.0 * batch / mean,
        "iters": iters,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="rfdetr-nano")
    p.add_argument("--qat-ckpt", type=Path, required=True,
                   help="ModelOpt-quantized *_qat.pth (INT8 engine source)")
    p.add_argument("--imgsz", type=int, default=384)
    p.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batches", default="1,4,8,16,32", help="Comma-separated batch sizes")
    p.add_argument("--onnx-opset", type=int, default=17)
    p.add_argument("--bench-iters", type=int, default=200)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--reexport", action="store_true",
                   help="Force ONNX re-export even if dynamic ONNX already exists")
    args = p.parse_args(argv)

    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    batches = sorted({int(b) for b in args.batches.split(",") if b.strip()})
    max_batch = max(batches)
    out_dir = args.out_dir or args.qat_ckpt.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    int8_onnx = out_dir / f"{args.model.replace('-', '_')}_qat_dyn.onnx"
    fp32_onnx = out_dir / f"{args.model.replace('-', '_')}_fp32_dyn.onnx"

    print(f"\n=== {args.model} :: INT8 vs FP16 batch sweep {batches} ===")

    if args.reexport or not int8_onnx.exists():
        export_dynamic_onnx(args.model, int8_onnx, imgsz=args.imgsz, device=device,
                            opset=args.onnx_opset, qat_ckpt=args.qat_ckpt)
    else:
        print(f"  [export] reusing {int8_onnx}")
    if args.reexport or not fp32_onnx.exists():
        export_dynamic_onnx(args.model, fp32_onnx, imgsz=args.imgsz, device=device,
                            opset=args.onnx_opset, qat_ckpt=None)
    else:
        print(f"  [export] reusing {fp32_onnx}")

    int8_engine = out_dir / f"{args.model.replace('-', '_')}_int8_dyn.engine"
    fp16_engine = out_dir / f"{args.model.replace('-', '_')}_fp16_dyn.engine"
    opt_batch = batches[len(batches) // 2]
    build_dynamic_engine(int8_onnx, int8_engine, int8=True, imgsz=args.imgsz,
                         min_batch=1, opt_batch=opt_batch, max_batch=max_batch)
    build_dynamic_engine(fp32_onnx, fp16_engine, int8=False, imgsz=args.imgsz,
                         min_batch=1, opt_batch=opt_batch, max_batch=max_batch)

    int8_runner = TRTRunner(load_engine(int8_engine), device)
    fp16_runner = TRTRunner(load_engine(fp16_engine), device)

    rows = []
    for b in batches:
        i8 = benchmark_at_batch(int8_runner, b, args.imgsz, device, iters=args.bench_iters)
        f16 = benchmark_at_batch(fp16_runner, b, args.imgsz, device, iters=args.bench_iters)
        speedup = f16["mean_ms"] / i8["mean_ms"]
        rows.append({"batch": b, "int8": i8, "fp16": f16, "int8_speedup_vs_fp16": speedup})
        print(f"  batch {b:3d} | INT8 {i8['mean_ms']:7.3f} ms "
              f"({i8['per_image_ms']:.3f} ms/img, {i8['throughput_fps']:6.1f} fps) | "
              f"FP16 {f16['mean_ms']:7.3f} ms "
              f"({f16['per_image_ms']:.3f} ms/img, {f16['throughput_fps']:6.1f} fps) | "
              f"speedup {speedup:.3f}x")

    summary = {
        "model": args.model,
        "imgsz": args.imgsz,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "int8_engine": str(int8_engine),
        "fp16_engine": str(fp16_engine),
        "sweep": rows,
    }
    dst = out_dir / "batch_sweep_summary.json"
    dst.write_text(json.dumps(summary, indent=2))
    print(f"  [trt] wrote {dst}")

    best = max(rows, key=lambda r: r["int8_speedup_vs_fp16"])
    print(f"\n  best INT8 speedup: {best['int8_speedup_vs_fp16']:.3f}x at batch {best['batch']}")
    if best["int8_speedup_vs_fp16"] > 1.0:
        print("  -> INT8 overtakes FP16 in the compute-bound regime: pipeline is correct.")
    else:
        print("  -> INT8 never overtakes FP16: bottleneck is structural (reformat tax / Myelin).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
