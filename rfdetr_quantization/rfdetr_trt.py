#!/usr/bin/env python3
"""
TensorRT build + latency benchmark + COCO mAP for RF-DETR quantized ONNX.

Consumes the ONNX produced by ``nvidia_modelopt_rfdetr_qat.py``:

* the QAT model exports a **Q/DQ ONNX** — TensorRT reads the embedded scales
  and builds an explicit-quantization **INT8** engine;
* the FP32 model exports a plain ONNX — built as an **FP16** baseline so the
  INT8 speedup is measured against a realistic deployment baseline.

Engines are built, benchmarked (GPU-compute latency via CUDA events), and
optionally evaluated for COCO ``val2017`` mAP by running the *engine itself*
through RF-DETR's ``PostProcess`` + ``CocoEvaluator`` — i.e. the accuracy of
the actually-deployed TensorRT graph, not the torch model.

I/O buffers are plain CUDA ``torch`` tensors addressed by ``data_ptr()``, so no
``pycuda`` dependency is needed.

Usage::

    python rfdetr_quantization/rfdetr_trt.py \\
        --qat-onnx runs/modelopt_qat/rfdetr_nano/rfdetr_nano_qat.onnx \\
        --fp32-onnx runs/modelopt_qat/rfdetr_nano/rfdetr_nano_fp32.onnx \\
        --eval-limit 1500
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rfdetr_quantization.nvidia_modelopt_rfdetr_qat import (  # noqa: E402
    ANN_FILE,
    VAL_DIR,
    _imagenet_normalize,
    _preprocess_image,
    load_rfdetr,
)


# ---------------------------------------------------------------------------
# Engine build
# ---------------------------------------------------------------------------

def build_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    int8: bool,
    fp16: bool = True,
    workspace_gb: int = 4,
) -> Path:
    """Parse ``onnx_path`` and serialize a TensorRT engine to ``engine_path``.

    ``int8=True`` enables explicit-quantization INT8 (the ONNX must carry Q/DQ
    nodes); ``fp16`` lets non-quantized layers run in half precision.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # TRT 10: strongly-typed/explicit batch
    parser = trt.OnnxParser(network, logger)
    onnx_bytes = onnx_path.read_bytes()
    if not parser.parse(onnx_bytes):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n{errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)

    precision = "INT8" if int8 else ("FP16" if fp16 else "FP32")
    print(f"  [trt] building {precision} engine from {onnx_path.name} "
          f"({network.num_layers} layers) ...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")
    engine_path.write_bytes(serialized)
    print(f"  [trt] wrote {engine_path} ({engine_path.stat().st_size/1e6:.1f} MB) "
          f"in {time.time()-t0:.1f}s")
    return engine_path


def load_engine(engine_path: Path):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to deserialize {engine_path}")
    return engine


class TRTRunner:
    """Minimal TensorRT executor using CUDA ``torch`` tensors as I/O buffers."""

    def __init__(self, engine, device: torch.device):
        import tensorrt as trt

        self.engine = engine
        self.context = engine.create_execution_context()
        self.device = device
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.input_name = next(
            n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        )
        # Deterministic output order: rfdetr export names them out.0, out.1, ...
        self.output_names = sorted(
            n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
        )

    @torch.no_grad()
    def infer(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Run the engine on a contiguous CUDA float32 ``[B,3,H,W]`` tensor."""
        images = images.to(self.device, dtype=torch.float32).contiguous()
        self.context.set_input_shape(self.input_name, tuple(images.shape))
        self.context.set_tensor_address(self.input_name, images.data_ptr())
        outputs = []
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            buf = torch.empty(shape, dtype=torch.float32, device=self.device)
            outputs.append(buf)
            self.context.set_tensor_address(name, buf.data_ptr())
        stream = torch.cuda.current_stream(self.device)
        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return outputs


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def benchmark_engine(
    runner: TRTRunner,
    imgsz: int,
    device: torch.device,
    *,
    warmup: int = 50,
    iters: int = 300,
) -> dict:
    """Measure single-image GPU-compute latency with CUDA events."""
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
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
    median = times_ms[len(times_ms) // 2]
    p99 = times_ms[min(len(times_ms) - 1, int(0.99 * len(times_ms)))]
    return {
        "mean_ms": mean,
        "median_ms": median,
        "p99_ms": p99,
        "throughput_fps": 1000.0 / mean,
        "iters": iters,
    }


# ---------------------------------------------------------------------------
# COCO mAP of the TensorRT engine
# ---------------------------------------------------------------------------

def evaluate_engine_map(
    runner: TRTRunner,
    postprocess: torch.nn.Module,
    image_dir: Path,
    ann_file: Path,
    device: torch.device,
    imgsz: int,
    *,
    limit: int = 0,
) -> dict:
    """Run COCO ``val2017`` through the TensorRT engine and return ``{map, map50}``."""
    from pycocotools.coco import COCO
    from rfdetr.evaluation.coco_eval import CocoEvaluator

    coco_gt = COCO(str(ann_file))
    evaluator = CocoEvaluator(coco_gt, ["bbox"])
    img_ids = sorted(coco_gt.getImgIds())
    if limit:
        img_ids = img_ids[:limit]

    t0 = time.time()
    for iid in img_ids:
        info = coco_gt.loadImgs(iid)[0]
        tensor = _preprocess_image(image_dir / info["file_name"], imgsz)
        batch = _imagenet_normalize(tensor.unsqueeze(0)).to(device)
        logits, boxes = runner.infer(batch)
        out = {"pred_logits": logits, "pred_boxes": boxes}
        target_sizes = torch.tensor(
            [[info["height"], info["width"]]], device=device, dtype=torch.float32
        )
        results = postprocess(out, target_sizes)
        evaluator.update({iid: results[0]})

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.coco_eval["bbox"].stats
    print(f"  [trt-eval] {len(img_ids)} imgs in {time.time()-t0:.1f}s "
          f"-> mAP50-95={float(stats[0]):.4f} mAP50={float(stats[1]):.4f}")
    return {"map": float(stats[0]), "map50": float(stats[1]), "num_images": len(img_ids)}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_trt_stage(
    qat_onnx: Path,
    fp32_onnx: Path | None,
    *,
    model: str,
    imgsz: int,
    device: torch.device,
    eval_limit: int,
    bench_iters: int,
    out_dir: Path,
) -> dict:
    """Build / benchmark / evaluate the INT8 engine (and FP16 baseline if given)."""
    postprocess = None
    if eval_limit >= 0 and ANN_FILE.exists():
        # PostProcess is a tiny stateless module; grab it from a fresh load.
        _, _, postprocess = load_rfdetr(model, torch.device("cpu"))

    summary: dict = {}

    int8_engine = out_dir / f"{qat_onnx.stem}_int8.engine"
    build_engine(qat_onnx, int8_engine, int8=True, fp16=True)
    int8_runner = TRTRunner(load_engine(int8_engine), device)
    summary["int8"] = {"engine": str(int8_engine)}
    summary["int8"]["latency"] = benchmark_engine(int8_runner, imgsz, device, iters=bench_iters)
    print(f"  [trt] INT8  latency: mean={summary['int8']['latency']['mean_ms']:.3f}ms "
          f"median={summary['int8']['latency']['median_ms']:.3f}ms "
          f"throughput={summary['int8']['latency']['throughput_fps']:.1f} fps")
    if postprocess is not None:
        summary["int8"]["accuracy"] = evaluate_engine_map(
            int8_runner, postprocess, VAL_DIR, ANN_FILE, device, imgsz, limit=eval_limit
        )

    if fp32_onnx is not None and fp32_onnx.exists():
        fp16_engine = out_dir / f"{fp32_onnx.stem}_fp16.engine"
        build_engine(fp32_onnx, fp16_engine, int8=False, fp16=True)
        fp16_runner = TRTRunner(load_engine(fp16_engine), device)
        summary["fp16"] = {"engine": str(fp16_engine)}
        summary["fp16"]["latency"] = benchmark_engine(fp16_runner, imgsz, device, iters=bench_iters)
        print(f"  [trt] FP16  latency: mean={summary['fp16']['latency']['mean_ms']:.3f}ms "
              f"median={summary['fp16']['latency']['median_ms']:.3f}ms "
              f"throughput={summary['fp16']['latency']['throughput_fps']:.1f} fps")
        if postprocess is not None:
            summary["fp16"]["accuracy"] = evaluate_engine_map(
                fp16_runner, postprocess, VAL_DIR, ANN_FILE, device, imgsz, limit=eval_limit
            )
        speedup = (summary["fp16"]["latency"]["mean_ms"]
                   / summary["int8"]["latency"]["mean_ms"])
        summary["int8_speedup_vs_fp16"] = speedup
        print(f"  [trt] INT8 speedup vs FP16: {speedup:.2f}x")

    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qat-onnx", type=Path, required=True, help="Q/DQ ONNX from the QAT model")
    p.add_argument("--fp32-onnx", type=Path, default=None, help="FP32 ONNX baseline (built as FP16)")
    p.add_argument("--model", default="rfdetr-nano", help="RF-DETR variant (for PostProcess)")
    p.add_argument("--imgsz", type=int, default=384)
    p.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval-limit", type=int, default=1500,
                   help="Cap mAP eval images (0 = full val2017, -1 = skip accuracy)")
    p.add_argument("--bench-iters", type=int, default=300)
    p.add_argument("--out-dir", type=Path, default=None, help="Engine output dir (default: alongside ONNX)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    if not args.qat_onnx.exists():
        print(f"ERROR: {args.qat_onnx} not found", file=sys.stderr)
        return 2
    out_dir = args.out_dir or args.qat_onnx.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {args.model} :: TensorRT build / benchmark / accuracy ===")
    summary = run_trt_stage(
        args.qat_onnx, args.fp32_onnx,
        model=args.model, imgsz=args.imgsz, device=device,
        eval_limit=args.eval_limit, bench_iters=args.bench_iters, out_dir=out_dir,
    )

    dst = out_dir / "trt_summary.json"
    dst.write_text(json.dumps(summary, indent=2))
    print(f"  [trt] wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
