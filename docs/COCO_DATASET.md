# COCO Dataset Setup for Quantization Playground

This directory contains scripts and configuration for downloading and using the COCO dataset for model validation and quantization benchmarking.

## 📥 Quick Start

### 1. Download COCO Dataset
```bash
export WORKSPACE="yourworkspacename"

# Make sure you're in the model optimizations directory
cd $HOME/$WORKSPACE/model-optimizations

# Run the download script
./download_coco_dataset.sh
```

**Note**: The download requires approximately 25GB of disk space and may take 30-60 minutes depending on your internet connection.

### 2. Verify Installation
```bash
# Check if dataset is properly downloaded
python coco_examples.py
# Choose option 1: "Check COCO dataset structure"
```

### 3. Validate Models
```bash
# Activate environment
source venv/bin/activate

# Validate a YOLO model on COCO
yolo val model=yolov8n.pt data=coco.yaml
```

## 📂 What Gets Downloaded

- **Validation Set**: 5,000 images from COCO 2017 validation set
- **Test Set**: 40,670 images from COCO 2017 test set  
- **Annotations**: Instance segmentation, captions, and keypoint annotations
- **Total Size**: ~25GB (can be reduced to ~15GB after cleanup)

## 🎯 Use Cases

### Model Validation
```bash
# Compare different model formats
yolo val model=yolov8n.pt data=coco.yaml      # PyTorch
yolo val model=yolov8n.onnx data=coco.yaml    # ONNX
yolo val model=yolov8n.engine data=coco.yaml  # TensorRT
```

### Quantization Benchmarking
```python
# Use the provided examples
python coco_examples.py
# Choose option 4: "Benchmark quantized models"
```

### Custom Evaluation
```python
from ultralytics import YOLO

# Load your model
model = YOLO('your_model.pt')

# Validate on COCO
results = model.val(data='coco.yaml')
print(f"mAP50: {results.box.map50}")
print(f"mAP50-95: {results.box.map}")
```

## 🗂️ Dataset Structure

After download, your dataset will be organized as:

```
datasets/coco/
├── val2017/                    # Validation images
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   └── ... (5,000 total)
├── test2017/                   # Test images  
│   ├── 000000000001.jpg
│   ├── 000000000016.jpg
│   └── ... (40,670 total)
├── annotations/                # COCO annotations
│   ├── instances_val2017.json  # Object detection annotations
│   ├── captions_val2017.json   # Image captions
│   └── person_keypoints_val2017.json  # Keypoint annotations
├── val2017.zip                 # Original zip (can delete)
├── test2017.zip                # Original zip (can delete)
└── annotations_trainval2017.zip # Original zip (can delete)
```

## 🔧 Configuration

The `coco.yaml` file contains the dataset configuration for YOLO:

```yaml
# Dataset paths
path: ./datasets/coco
val: val2017
test: test2017

# 80 COCO classes
nc: 80
names: [person, bicycle, car, ...]
```

## 🧹 Cleanup

To save disk space after download:
```bash
# Remove original zip files (saves ~10GB)
cd datasets/coco
rm *.zip

# Or use the interactive cleanup in the download script
./download_coco_dataset.sh
# Choose 'y' when prompted to delete zip files
```

## 🚨 Troubleshooting

### Download Issues
- **Slow download**: The dataset is large, be patient
- **Interrupted download**: Re-run the script, it will skip existing files
- **Disk space**: Ensure you have at least 25GB free space

### Validation Issues
- **Can't find dataset**: Check that `datasets/coco/` exists
- **CUDA out of memory**: Reduce batch size in validation command
- **No annotations**: Ensure `annotations/` directory was extracted

### Common Errors
```bash
# If you get "command not found" for yolo
source venv/bin/activate

# If dataset path is wrong
# Edit coco.yaml and update the 'path' field

# If validation is slow
# Use smaller batch size: yolo val model=yolov8n.pt data=coco.yaml batch=8
```


## 🔗 Resources

- [COCO Dataset](https://cocodataset.org/)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [PyTorch Quantization](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
