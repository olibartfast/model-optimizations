# Quantization on a Custom Dataset

This guide shows how to run the canonical YOLO quantization pipelines on a
non-COCO Ultralytics detection dataset.

Run commands from the repository root after creating the repo-local Python
environment.

## Python Environment

`<venv-dir>` is a placeholder for your chosen Python 3.12 virtual environment
directory. The environment contains PyTorch, Ultralytics, NVIDIA ModelOpt,
ONNX export dependencies, and the extra packages needed by
`modelopt.torch.export`.

Create it before running quantization:

```bash
python3.12 -m venv <venv-dir>
source <venv-dir>/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install --no-build-isolation \
  --extra-index-url https://pypi.ngc.nvidia.com \
  -r configs/requirements.txt
```

The important details are `--no-build-isolation` and the NVIDIA NGC package
index required for `nvidia-modelopt[torch]`.

Equivalent convenience wrapper:

```bash
export QUANTIZATION_VENV=<venv-dir>
./scripts/run_venv.sh
source <venv-dir>/bin/activate
```

After activation, use `python` for repo commands.

Verify the environment before running a long quantization job:

```bash
python -c 'import torch, ultralytics, modelopt.torch.quantization; print("ok")'
```

## Dataset YAML

Create an Ultralytics detection YAML with `path`, `train`, `val`, and `names`:

```yaml
path: /absolute/path/to/my_dataset
train: images/train
val: images/val
names:
  0: person
  1: vehicle
```

Relative `train` and `val` paths are resolved under `path`. The scripts pass
this YAML directly to Ultralytics for validation and QAT training.

Expected image/label layout is the normal Ultralytics YOLO layout:

```text
my_dataset/
  images/
    train/*.jpg
    val/*.jpg
  labels/
    train/*.txt
    val/*.txt
```

## Calibration Source

Use `--calib-source` to choose the images used for PTQ calibration. It accepts:

- `train` — resolves to the YAML `train` split.
- `val` — resolves to the YAML `val` split.
- `train2017` / `val2017` — legacy COCO aliases; with a custom YAML these map
  to `train` / `val`.
- A direct path to an image directory.
- A direct path to a newline-delimited image list.

For final metrics, prefer `--calib-source train` so calibration does not reuse
the validation split used for mAP.

## QAT Pipeline

This runs FP32 validation, PTQ calibration, QAT distillation, final validation,
and ONNX export:

```bash
python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
  --models yolo26s \
  --data configs/my_dataset.yaml \
  --calib-source train \
  --calib-size 260 \
  --qat-recipe auto \
  --qat-log-every 20 \
  --qat-eval-every 5 \
  --seed 0
```

For a shorter smoke test:

```bash
python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
  --models yolo26s \
  --data configs/my_dataset.yaml \
  --calib-source train \
  --calib-size 32 \
  --qat-epochs 1 \
  --qat-batches-per-epoch 10 \
  --batch 4 \
  --no-export
```

Resume QAT from a saved PTQ checkpoint:

```bash
python yolo_quantization/qat/nvidia_modelopt_yolo_qat.py \
  --models yolo26s \
  --data configs/my_dataset.yaml \
  --from-ptq runs/modelopt_qat/yolo26s/yolo26s_ptq.pth \
  --skip-fp32-eval \
  --qat-mode distill
```

QAT outputs land under:

```text
runs/modelopt_qat/<model>/
  <model>_ptq.pth
  <model>_qat.pth
  <model>_qat.onnx
  qat_train.csv
  summary.json
```

## PTQ-Only ONNX Pipeline

Use this path when you only need ONNX PTQ export:

```bash
python yolo_quantization/ptq/nvidia_modelopt_yolo.py \
  --models yolo11x \
  --data configs/my_dataset.yaml \
  --calib-source train \
  --quant-modes int8 fp8 \
  --calib-size 256
```

PTQ outputs land under:

```text
runs/modelopt/<model>/
  <model>.onnx
  <model>_int8.onnx
  <model>_fp8.onnx
  summary.json
```

## Wrapper Script

The top-level wrapper forwards all extra flags to the canonical scripts and
skips COCO auto-download when `--data` is present:

```bash
./scripts/run_modelopt_yolo.sh \
  --models yolo26s \
  --data configs/my_dataset.yaml \
  --calib-source train

./scripts/run_modelopt_yolo.sh ptq \
  --models yolo11x \
  --data configs/my_dataset.yaml \
  --calib-source train \
  --quant-modes int8
```

## Notes

- Validation uses the YAML `val` split with `conf=0.001` and `iou=0.6`.
- `--qat-recipe auto` still selects by model family: YOLO26 uses
  `yolo26-distill`; single-head models use `yolo11-distill`.
- If your dataset is much smaller than COCO, reduce `--calib-size`,
  `--qat-batches-per-epoch`, and `--batch` for the first smoke test.
- Direct image-list calibration files should contain one image path per line.
