#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Installing Python Dependencies..."
# Install CPU-only PyTorch to save space and time (Critical for Render Free Tier)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

echo "Building Frontend..."
cd frontend
npm install
npm run build
cd ..

echo "Moving Frontend Build..."
mkdir -p backend/static
cp -r frontend/dist/* backend/static/

echo "Build Complete!"
