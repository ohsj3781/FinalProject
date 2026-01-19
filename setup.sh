# setup.sh
#!/bin/bash

echo "Setting up training environment for quantized ResNet on COCO dataset"

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data
mkdir -p checkpoints
mkdir -p logs
mkdir -p results

# Download COCO 2017 dataset
echo "Downloading COCO 2017 dataset..."
cd data
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

echo "Extracting COCO dataset..."
unzip -q train2017.zip
unzip -q val2017.zip
unzip -q annotations_trainval2017.zip

rm train2017.zip val2017.zip annotations_trainval2017.zip

cd ..

echo "Setup completed successfully!"