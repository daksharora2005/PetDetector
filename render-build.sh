#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Installing Python Dependencies..."
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
