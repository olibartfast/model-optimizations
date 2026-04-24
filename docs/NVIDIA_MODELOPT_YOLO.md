# NVIDIA ModelOpt on Ultralytics YOLO

Apply [nvidia-modelopt](https://pypi.org/project/nvidia-modelopt/) to
Ultralytics YOLO checkpoints (`yolo11x`, `yolo26x`, or any other
Ultralytics stem) on the COCO 2017 dataset.

Two pipelines are provided:

1. **QAT** (primary) — `examples/nvidia_modelopt_yolo_qat.py`.
   Mirrors [`NVIDIA/Model-Optimizer/examples/cnn_qat`](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/cnn_qat):
   PTQ calibration → `mtq.quantize` → short QAT fine-tune → `mto.save`
   → ONNX export. Recovers accuracy lost to INT8 quantization.
2. **PTQ (ONNX)** (baseline) — `examples/nvidia_modelopt_yolo.py`.
   `modelopt.onnx.quantization.quantize()` over an FP32 ONNX export
   (INT8 / FP8 / INT4). Fast, training-free, deployable to TensorRT.

## QAT pipeline (recommended)

Mirrors the three-stage flow from NVIDIA's `cnn_qat/torchvision_qat.py`:

```text
load FP32 YOLO  →  calibrate closure  →  mtq.quantize(model, INT8_DEFAULT_CFG, calibrate)
             │                                                         │
             │                                                   mto.save (PTQ ckpt)
             │                                                         │
             │                                     yolo.train(data=coco.yaml, epochs=N, lr0=1e-4)
             │                                                         │
             │                                                   mto.save (QAT ckpt)
             ▼                                                         │
   val FP32 mAP                                                        ▼
                                                           val PTQ mAP  /  val QAT mAP
                                                                        │
                                                                  ONNX export
```

Key modelopt APIs used (same as the cnn_qat reference):

```python
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

def calibrate(m):                             # forward-loop closure
    m.eval()
    with torch.no_grad():
        for b in calib_batches:
            m(b.cuda())

mtq.quantize(yolo.model, mtq.INT8_DEFAULT_CFG, calibrate)   # PTQ
mto.save(yolo.model, "yolo11x_ptq.pth")
yolo.train(data="coco.yaml", epochs=N, lr0=1e-4)            # QAT fine-tune
mto.save(yolo.model, "yolo11x_qat.pth")
```

QAT training itself is delegated to `YOLO.train()` because YOLO has its
own detection loss, mosaic/mixup augmentation and COCO dataloader. The
quantizer layers injected by `mtq.quantize` stay in place because the
trainer mutates `yolo.model` in place.

### Run

```bash
./scripts/run_venv.sh && source venv/bin/activate
pip install -r configs/requirements.txt
./scripts/download_coco_dataset.sh

# Default: QAT for yolo11x + yolo26x, 2 epochs, 512 calibration images
./scripts/run_modelopt_yolo.sh

# Or call directly with overrides:
python examples/nvidia_modelopt_yolo_qat.py \
    --models yolo11x yolo26x \
    --qat-epochs 3 \
    --calib-size 512 \
    --batch 16 \
    --imgsz 640
```

### QAT CLI

| flag              | default             | purpose                                     |
| ----------------- | ------------------- | ------------------------------------------- |
| `--models`        | `yolo11x yolo26x`   | Ultralytics weights stems                   |
| `--qat-epochs`    | `2`                 | fine-tune epochs after PTQ calibration      |
| `--qat-lr`        | `1e-4`              | constant LR for the QAT fine-tune           |
| `--calib-size`    | `512`               | val2017 images for PTQ calibration          |
| `--batch`         | `16`                | train + calibration batch size              |
| `--val-batch`     | `8`                 | batch size for `model.val()`                |
| `--imgsz`         | `640`               | input resolution                            |
| `--device`        | `0` if CUDA else `cpu` | Ultralytics device string                |
| `--skip-fp32-eval`| off                 | skip the FP32 baseline mAP                  |
| `--no-export`     | off                 | skip the final ONNX export                  |

### Example report

```
model      stage   mAP50      mAP    secs  notes
-------------------------------------------------
yolo11x    fp32   0.6820   0.5340   180.2
yolo11x    ptq    0.6612   0.5121   179.4  ptq.pth  (-2.1 mAP50 vs FP32)
yolo11x    qat    0.6781   0.5298   178.9  qat.pth  (recovered)
yolo26x    fp32   0.7015   0.5522   184.6
yolo26x    ptq    0.6790   0.5310   184.1
yolo26x    qat    0.6972   0.5485   184.0
```

## PTQ pipeline (ONNX-only baseline)

Training-free INT8/FP8/INT4 quantization via
`modelopt.onnx.quantization.quantize()`. Useful as a quick baseline to
compare against QAT recovery, and produces ONNX files directly
compatible with TensorRT.

```bash
./scripts/run_modelopt_yolo.sh ptq --quant-modes int8 fp8 --calib-size 512
# or:
python examples/nvidia_modelopt_yolo.py --models yolo11x yolo26x --quant-modes int8 fp8
```

See the docstring of `examples/nvidia_modelopt_yolo.py` for the full
PTQ flag reference.

## Output layout

```
runs/
├── modelopt_qat/                       # QAT pipeline
│   ├── report.json
│   ├── yolo11x/
│   │   ├── yolo11x_ptq.pth            # mto.save after PTQ calibration
│   │   ├── yolo11x_qat.pth            # mto.save after QAT fine-tune
│   │   ├── yolo11x_qat.onnx           # deployable export
│   │   ├── qat_train/                 # Ultralytics trainer run dir
│   │   └── summary.json
│   └── yolo26x/...
└── modelopt/                           # PTQ-only ONNX pipeline
    ├── report.json
    ├── calib_val2017_n256_imgsz640.npy
    └── yolo11x/
        ├── yolo11x.onnx               # FP32 export
        ├── yolo11x_int8.onnx          # modelopt PTQ
        └── summary.json
```

## Troubleshooting

- **`Unknown model yolo26x`** — pass a local weight path (e.g.
  `--models /path/to/yolo26x.pt`) or upgrade `ultralytics`.
- **`CUDA out of memory` during QAT** — lower `--batch` (try `8` or `4`)
  and/or `--val-batch`. The PTQ calibration phase also uses `--batch`.
- **QAT mAP regresses below PTQ** — reduce `--qat-lr` (try `5e-5`)
  and/or increase `--calib-size`. The default constant-LR schedule is a
  starting point; for long fine-tunes you may prefer to enable warmup
  and cosine decay by modifying `yolo.train()` kwargs in
  `run_qat_for_model`.
- **`modelopt.torch.quantization` import fails** — `nvidia-modelopt`
  wasn't installed. See `docs/INSTALLATION_FIX.md`.
- **`ONNX export failed` after QAT** — the quantizer modules produced by
  `mtq.quantize` are exportable but some Ultralytics head ops may need
  explicit treatment; the QAT `.pth` checkpoint from `mto.save` is still
  usable for further work. Use `--no-export` to skip.

## References

- [NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer)
- [cnn_qat example](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/cnn_qat) — the reference this pipeline follows
- [ModelOpt docs](https://docs.nvidia.com/deeplearning/modelopt/)
- [Ultralytics training](https://docs.ultralytics.com/modes/train/)
- `docs/COCO_DATASET.md` — COCO download / layout
- `docs/LINEAR_QUANTIZATION_THEORY.md` — quantization background
