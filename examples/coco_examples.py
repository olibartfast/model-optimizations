#!/usr/bin/env python3
"""
COCO Dataset Usage Examples
Demonstrates how to use the downloaded COCO dataset for model validation and testing.
"""

import os
import sys
import torch
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_coco_dataset():
    """Check if COCO dataset is properly downloaded and structured"""
    print("🔍 Checking COCO dataset structure...")
    
    dataset_path = Path("../datasets/coco") if not Path("datasets/coco").exists() else Path("datasets/coco")
    
    if not dataset_path.exists():
        print("❌ COCO dataset not found!")
        print("Run: ./download_coco_dataset.sh to download the dataset")
        return False
    
    # Check validation images
    val_path = dataset_path / "val2017"
    if val_path.exists():
        val_count = len(list(val_path.glob("*.jpg")))
        print(f"✓ Validation images: {val_count:,}")
    else:
        print("❌ Validation images not found")
        return False
    
    # Check test images
    test_path = dataset_path / "test2017"
    if test_path.exists():
        test_count = len(list(test_path.glob("*.jpg")))
        print(f"✓ Test images: {test_count:,}")
    else:
        print("❌ Test images not found")
        return False
    
    # Check annotations
    ann_path = dataset_path / "annotations"
    if ann_path.exists():
        annotations = list(ann_path.glob("*.json"))
        print(f"✓ Annotation files: {len(annotations)}")
        for ann in annotations:
            print(f"  - {ann.name}")
    else:
        print("❌ Annotations not found")
        return False
    
    return True

def validate_yolo_model():
    """Example: Validate a YOLO model on COCO dataset"""
    print("\n🔬 YOLO Model Validation Example")
    print("=" * 50)
    
    if not check_coco_dataset():
        return
    
    try:
        from ultralytics import YOLO
        
        # Load a YOLO model
        model_path = "../models/yolov8n.pt" if not os.path.exists("yolov8n.pt") else "yolov8n.pt"
        if not os.path.exists(model_path) and not os.path.exists("yolov8n.pt"):
            print(f"📥 Downloading yolov8n.pt...")
            model = YOLO('yolov8n.pt')  # This will download if not present
        else:
            model = YOLO(model_path if os.path.exists(model_path) else 'yolov8n.pt')
        
        print(f"✓ Loaded model: {model_path}")
        
        # Validate on COCO dataset
        print("🚀 Running validation on COCO dataset...")
        print("Note: This may take several minutes...")
        
        # Run validation
        coco_config = "../configs/coco.yaml"
        if not os.path.exists(coco_config):
            coco_config = "coco.yaml"  # Fallback for direct execution
        results = model.val(
            data=coco_config,  # Use our COCO configuration
            imgsz=640,         # Image size
            batch=16,          # Batch size (adjust based on GPU memory)
            conf=0.001,        # Confidence threshold
            iou=0.6,           # IoU threshold for NMS
            device='0' if torch.cuda.is_available() else 'cpu',
            verbose=True
        )
        
        print("✓ Validation completed!")
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
        
    except ImportError:
        print("❌ Ultralytics not available. Make sure to activate the virtual environment:")
        print("source quantization_venv/bin/activate")
    except Exception as e:
        print(f"❌ Validation failed: {e}")

def test_model_inference():
    """Example: Run inference on a sample of test images"""
    print("\n🎯 Model Inference Example")
    print("=" * 50)
    
    if not check_coco_dataset():
        return
    
    try:
        from ultralytics import YOLO
        import random
        
        # Load model
        model = YOLO('yolov8n.pt')
        
        # Get some random test images
        test_path = dataset_path / "test2017"
        test_images = list(test_path.glob("*.jpg"))
        
        if not test_images:
            print("❌ No test images found")
            return
        
        # Select 5 random images
        sample_images = random.sample(test_images, min(5, len(test_images)))
        
        print(f"🔍 Running inference on {len(sample_images)} sample images...")
        
        for img_path in sample_images:
            print(f"Processing: {img_path.name}")
            
            # Run inference
            results = model(str(img_path), verbose=False)
            
            # Print detection results
            for r in results:
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    print(f"  Detected {len(boxes)} objects:")
                    for i, box in enumerate(boxes):
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls_id]
                        print(f"    {i+1}. {class_name}: {conf:.3f}")
                else:
                    print("  No objects detected")
            
            # Optionally save results with bounding boxes
            save_path = f"results_{img_path.stem}.jpg"
            r.save(filename=save_path)
            print(f"  Results saved to: {save_path}")
        
        print("✓ Inference completed!")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")

def benchmark_quantized_models():
    """Example: Compare performance of FP32 vs quantized models"""
    print("\n⚡ Quantized Model Benchmark")
    print("=" * 50)
    
    if not check_coco_dataset():
        return
    
    try:
        from ultralytics import YOLO
        import time
        
        # Original model
        print("📥 Loading original FP32 model...")
        model_fp32 = YOLO('yolov8n.pt')
        
        # Export quantized versions
        print("🔧 Creating quantized models...")
        
        # ONNX FP16
        onnx_path = model_fp32.export(format='onnx', half=True, verbose=False)
        print(f"✓ Exported ONNX FP16: {onnx_path}")
        
        # Get a sample image for benchmarking
        test_path = dataset_path / "val2017"
        test_images = list(test_path.glob("*.jpg"))[:10]  # Use first 10 images
        
        if not test_images:
            print("❌ No validation images found for benchmarking")
            return
        
        # Benchmark FP32 model
        print("🏃 Benchmarking FP32 model...")
        start_time = time.time()
        for img in test_images:
            _ = model_fp32(str(img), verbose=False)
        fp32_time = time.time() - start_time
        
        # Benchmark ONNX model
        print("🏃 Benchmarking ONNX FP16 model...")
        model_onnx = YOLO(onnx_path)
        start_time = time.time()
        for img in test_images:
            _ = model_onnx(str(img), verbose=False)
        onnx_time = time.time() - start_time
        
        # Results
        print("\n📊 Benchmark Results:")
        print(f"FP32 model time: {fp32_time:.2f}s ({fp32_time/len(test_images)*1000:.1f}ms per image)")
        print(f"ONNX FP16 time: {onnx_time:.2f}s ({onnx_time/len(test_images)*1000:.1f}ms per image)")
        print(f"Speedup: {fp32_time/onnx_time:.2f}x")
        
        # Model sizes
        fp32_size = os.path.getsize('yolov8n.pt') / (1024*1024)
        onnx_size = os.path.getsize(onnx_path) / (1024*1024)
        print(f"FP32 model size: {fp32_size:.1f} MB")
        print(f"ONNX FP16 size: {onnx_size:.1f} MB")
        print(f"Size reduction: {(1-onnx_size/fp32_size)*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

def main():
    """Main function to run all examples"""
    print("🚀 COCO Dataset Usage Examples")
    print("=" * 50)
    
    # Check if virtual environment is activated
    if 'quantization_venv' not in sys.prefix:
        print("⚠️  Warning: Virtual environment not detected.")
        print("Activate with: source quantization_venv/bin/activate")
        print()
    
    print("Choose an example to run:")
    print("1. Check COCO dataset structure")
    print("2. Validate YOLO model on COCO")
    print("3. Run inference on sample images")
    print("4. Benchmark quantized models")
    print("5. Run all examples")
    print("q. Quit")
    
    choice = input("\nEnter your choice (1-5, q): ").strip().lower()
    
    if choice == '1':
        check_coco_dataset()
    elif choice == '2':
        validate_yolo_model()
    elif choice == '3':
        test_model_inference()
    elif choice == '4':
        benchmark_quantized_models()
    elif choice == '5':
        check_coco_dataset()
        validate_yolo_model()
        test_model_inference()
        benchmark_quantized_models()
    elif choice == 'q':
        print("Goodbye! 👋")
        return
    else:
        print("Invalid choice. Please try again.")
        main()

if __name__ == "__main__":
    main()
