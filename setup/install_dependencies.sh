#!/bin/bash
# AMD Hackathon - MI300X Environment Setup Script
# Run this script on the MI300X instance after SSH login

set -e  # Exit on error

echo "=========================================="
echo "AMD Hackathon - MI300X Setup"
echo "=========================================="
echo ""

# Check if running on AMD GPU
echo "Step 1: Verifying AMD GPU availability..."
if command -v rocm-smi &> /dev/null; then
    echo "✓ ROCm detected"
    rocm-smi --showproductname
else
    echo "⚠ Warning: rocm-smi not found. Are you on an MI300X instance?"
fi
echo ""

# Check Python version
echo "Step 2: Checking Python version..."
python_version=$(python3 --version)
echo "✓ $python_version"
echo ""

# Upgrade pip
echo "Step 3: Upgrading pip..."
python3 -m pip install --upgrade pip
echo "✓ pip upgraded"
echo ""

# Install Unsloth for ROCm
echo "Step 4: Installing Unsloth with ROCm support..."
echo "This may take 5-10 minutes..."
pip install "unsloth[rocm] @ git+https://github.com/unslothai/unsloth.git"
echo "✓ Unsloth installed"
echo ""

# Install core dependencies
echo "Step 5: Installing training dependencies..."
pip install transformers datasets trl accelerate peft bitsandbytes
echo "✓ Training libraries installed"
echo ""

# Install additional utilities
echo "Step 6: Installing additional utilities..."
pip install pandas numpy scikit-learn tqdm jupyter ipywidgets
echo "✓ Utilities installed"
echo ""

# Verify PyTorch and GPU
echo "Step 7: Verifying PyTorch GPU support..."
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠ Warning: CUDA not available!")
EOF
echo ""

# Verify Unsloth
echo "Step 8: Verifying Unsloth installation..."
python3 << EOF
try:
    from unsloth import FastLanguageModel
    print("✓ Unsloth imported successfully")
except ImportError as e:
    print(f"✗ Unsloth import failed: {e}")
EOF
echo ""

# Create directories
echo "Step 9: Creating project directories..."
mkdir -p ~/AMD_Hackathon/{data,models,outputs,logs}
echo "✓ Directories created"
echo ""

# Download example dataset (optional)
echo "Step 10: Setup complete!"
echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Clone your GitHub repo:"
echo "   git clone https://github.com/bledden/AMD_Hackathon.git"
echo ""
echo "2. Test GPU with:"
echo "   rocm-smi"
echo ""
echo "3. Start training with scripts in training/scripts/"
echo ""
echo "4. Monitor GPU usage:"
echo "   watch -n 1 rocm-smi"
echo ""
echo "Environment ready! Time to fine-tune! 🚀"
echo "=========================================="
