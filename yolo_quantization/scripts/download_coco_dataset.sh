#!/bin/bash
# COCO Dataset Download Script
# Downloads COCO validation and test datasets for model evaluation

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATASET_DIR="$PROJECT_ROOT/datasets/coco"
VAL_IMAGES_URL="http://images.cocodataset.org/zips/val2017.zip"
TEST_IMAGES_URL="http://images.cocodataset.org/zips/test2017.zip"
ANNOTATIONS_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔄 COCO Dataset Download Script${NC}"
echo "This script will download COCO validation and test datasets"
echo "Dataset will be saved to: ${DATASET_DIR}"
echo ""

# Create dataset directory
echo -e "${YELLOW}📁 Creating dataset directory...${NC}"
mkdir -p "${DATASET_DIR}"
cd "${DATASET_DIR}"

# Function to download with progress
download_file() {
    local url=$1
    local filename=$(basename "$url")
    
    if [[ -f "$filename" ]]; then
        echo -e "${GREEN}✓ $filename already exists, skipping download${NC}"
        return 0
    fi
    
    echo -e "${BLUE}⬇️  Downloading $filename...${NC}"
    if command -v wget &> /dev/null; then
        wget --progress=bar:force:noscroll "$url" -O "$filename"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar "$url" -o "$filename"
    else
        echo -e "${RED}❌ Error: Neither wget nor curl is available${NC}"
        exit 1
    fi
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ Successfully downloaded $filename${NC}"
    else
        echo -e "${RED}❌ Failed to download $filename${NC}"
        exit 1
    fi
}

# Function to extract zip files
extract_file() {
    local filename=$1
    
    if [[ ! -f "$filename" ]]; then
        echo -e "${RED}❌ Error: $filename not found${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}📦 Extracting $filename...${NC}"
    if command -v unzip &> /dev/null; then
        unzip -q "$filename"
        if [[ $? -eq 0 ]]; then
            echo -e "${GREEN}✓ Successfully extracted $filename${NC}"
        else
            echo -e "${RED}❌ Failed to extract $filename${NC}"
            exit 1
        fi
    else
        echo -e "${RED}❌ Error: unzip is not available${NC}"
        exit 1
    fi
}

# Check available disk space
check_disk_space() {
    local required_gb=25  # Approximate space needed in GB
    local available_gb=$(df . | awk 'NR==2 {print int($4/1024/1024)}')
    
    echo -e "${BLUE}💾 Checking disk space...${NC}"
    echo "Available space: ${available_gb}GB"
    echo "Required space: ~${required_gb}GB"
    
    if [[ $available_gb -lt $required_gb ]]; then
        echo -e "${RED}❌ Warning: Low disk space! You may need at least ${required_gb}GB${NC}"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}✓ Sufficient disk space available${NC}"
    fi
}

# Main download process
main() {
    echo -e "${BLUE}🚀 Starting COCO dataset download...${NC}"
    echo ""
    
    # Check prerequisites
    check_disk_space
    
    # Download files
    echo -e "${YELLOW}📥 Downloading dataset files...${NC}"
    download_file "$VAL_IMAGES_URL"
    download_file "$TEST_IMAGES_URL"
    download_file "$ANNOTATIONS_URL"
    echo ""
    
    # Extract files
    echo -e "${YELLOW}📦 Extracting dataset files...${NC}"
    extract_file "val2017.zip"
    extract_file "test2017.zip"
    extract_file "annotations_trainval2017.zip"
    echo ""
    
    # Create directory structure info
    echo -e "${BLUE}📁 Dataset structure:${NC}"
    echo "datasets/coco/"
    echo "├── val2017/          # Validation images (5,000 images)"
    echo "├── test2017/         # Test images (40,670 images)"
    echo "├── annotations/      # COCO annotations"
    echo "│   ├── instances_val2017.json"
    echo "│   ├── captions_val2017.json"
    echo "│   └── person_keypoints_val2017.json"
    echo "└── *.zip             # Original zip files (can be deleted)"
    echo ""
    
    # Show dataset statistics
    echo -e "${GREEN}📊 Dataset statistics:${NC}"
    if [[ -d "val2017" ]]; then
        val_count=$(find val2017 -name "*.jpg" | wc -l)
        echo "Validation images: ${val_count}"
    fi
    if [[ -d "test2017" ]]; then
        test_count=$(find test2017 -name "*.jpg" | wc -l)
        echo "Test images: ${test_count}"
    fi
    
    # Calculate total size
    total_size=$(du -sh . | cut -f1)
    echo "Total dataset size: ${total_size}"
    echo ""
    
    # Cleanup option
    echo -e "${YELLOW}🧹 Cleanup options:${NC}"
    read -p "Delete zip files to save space? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f *.zip
        echo -e "${GREEN}✓ Zip files deleted${NC}"
        new_size=$(du -sh . | cut -f1)
        echo "New dataset size: ${new_size}"
    fi
    
    echo ""
    echo -e "${GREEN}🎉 COCO dataset download complete!${NC}"
    echo -e "${BLUE}💡 Usage tips:${NC}"
    echo "• Use 'yolo val model=yolov8n.pt data=coco.yaml' to validate models"
    echo "• Validation images are in: $(pwd)/val2017/"
    echo "• Test images are in: $(pwd)/test2017/"
    echo "• Annotations are in: $(pwd)/annotations/"
}

# Handle interruption
trap 'echo -e "\n${RED}❌ Download interrupted${NC}"; exit 1' INT

# Run main function
main

# Return to original directory
cd - > /dev/null
