# NVIDIA ModelOpt on Ultralytics YOLO

End-to-end post-training quantization (PTQ) of Ultralytics YOLO checkpoints
(`yolo11x`, `yolo26x`, or any other Ultralytics stem) using
[nvidia-modelopt](https://pypi.org/project/nvidia-modelopt/), with calibration
and evaluation on the COCO 2017 dataset.

## Pipeline

1. **Export** – load the Ultralytics weights and export FP32 ONNX
   (`opset=17`, `simplify=True`, static shapes).
2. **Calibrate** – sample N images from `datasets/coco/val2017`, letterbox to
   `imgsz`, normalize to `[0,1]` NCHW float32 and cache as `.npy`.
3. **Quantize** – call `modelopt.onnx.quantization.quantize()` with the
   calibration tensor to produce an `INT8`/`FP8`/`INT4` ONNX file.
4. **Validate** – run `YOLO(onnx).val(data=configs/coco.yaml)` to get
   mAP50 / mAP50-95 on the 5,000 image COCO val set.
5. **Test** – optionally run inference on a random subset of COCO
   `test2017` images and save annotated outputs (no mAP — test2017 has no
   public labels).

All artifacts are cached under `runs/modelopt/<model>/` so steps can be
re-run in isolation. Calibration tensors are keyed by
`(calib-size, imgsz)` and shared across models.

## Quickstart

```bash
# 1. One-time setup
./scripts/run_venv.sh
source venv/bin/activate
pip install -r configs/requirements.txt

# 2. Download COCO val + test + annotations (~25GB)
./scripts/download_coco_dataset.sh

# 3. Run PTQ on yolo11x and yolo26x (INT8, default 256 calib images)
./scripts/run_modelopt_yolo.sh

# Or call the Python script directly with explicit options:
python examples/nvidia_modelopt_yolo.py \
    --models yolo11x yolo26x \
    --quant-modes int8 fp8 \
    --calib-size 512 \
    --imgsz 640 \
    --val-batch 8 \
    --test-samples 10
```

## CLI options

| flag               | default                | purpose                                                         |
| ------------------ | ---------------------- | --------------------------------------------------------------- |
| `--models`         | `yolo11x yolo26x`      | Ultralytics model stems (auto-downloaded if available)          |
| `--quant-modes`    | `int8`                 | any of `int8`, `fp8`, `int4`                                    |
| `--imgsz`          | `640`                  | inference / calibration image size                              |
| `--calib-size`     | `256`                  | number of val2017 images for calibration                        |
| `--val-batch`      | `8`                    | batch size for `model.val()`                                    |
| `--test-samples`   | `0`                    | random test2017 images to run inference on and save annotated   |
| `--device`         | `0` if CUDA else `cpu` | Ultralytics device string                                       |
| `--skip-val`       | off                    | quantize only; skip mAP eval (smoke test)                       |

## Output layout

```
runs/modelopt/
├── calib_val2017_n256_imgsz640.npy     # shared calibration tensor
├── report.json                          # combined per-model results
├── yolo11x/
│   ├── yolo11x.onnx                    # FP32 export
│   ├── yolo11x_int8.onnx               # modelopt PTQ output
│   ├── summary.json                    # {map50, map, size_mb, ...}
│   ├── test_fp32/                      # annotated test2017 samples
│   └── test_int8/
└── yolo26x/
    └── ...
```

## Example report

```
model      variant  size MB   mAP50     mAP  notes
----------------------------------------------------
yolo11x    fp32       254.3  0.6820  0.5340
yolo11x    int8        67.1  0.6755  0.5280
yolo26x    fp32       268.8  0.7015  0.5522
yolo26x    int8        70.9  0.6948  0.5460
```

## Troubleshooting

- **`Unknown model yolo26x`** – Ultralytics can't resolve the weights name.
  Either download the `.pt` manually and pass `--models /path/to/yolo26x.pt`
  (the script strips extensions when naming outputs), or upgrade
  `ultralytics` to a version that ships YOLO26.
- **`No module named 'cv2'`** – install `opencv-python`. It's listed in
  `configs/requirements.txt`.
- **`CUDA out of memory` during val** – lower `--val-batch`. INT8 ONNX via
  Ultralytics runs on `onnxruntime`; if no GPU EP is available it silently
  falls back to CPU and mAP eval will be slow.
- **`modelopt.onnx.quantization` import fails** – `nvidia-modelopt` wasn't
  installed. See `docs/INSTALLATION_FIX.md`.
- **Poor INT8 mAP** – increase `--calib-size` (512 or 1024 images is typical
  for detection) and re-run. The calibration cache is keyed by size so it
  will be regenerated.

## References

- [NVIDIA ModelOpt docs](https://docs.nvidia.com/deeplearning/modelopt/)
- [Ultralytics export](https://docs.ultralytics.com/modes/export/)
- `docs/COCO_DATASET.md` – COCO download / layout details
- `docs/LINEAR_QUANTIZATION_THEORY.md` – background on linear quantization
