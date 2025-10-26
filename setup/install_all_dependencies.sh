#!/bin/bash
# AMD Hackathon - Complete Dependency Installation Script
# For use inside the ROCm Docker container

set -e  # Exit on error

echo "================================"
echo "AMD Hackathon Dependency Setup"
echo "================================"
echo ""

# Check if running inside Docker container
if [ ! -f /.dockerenv ]; then
    echo "Warning: This script should be run inside the ROCm Docker container"
fi

echo "Step 1/5: Installing PyTorch with ROCm support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

echo ""
echo "Step 2/5: Installing Transformers and related libraries..."
pip install transformers accelerate datasets

echo ""
echo "Step 3/5: Installing training libraries (TRL, PEFT, BitsAndBytes)..."
pip install trl peft bitsandbytes

echo ""
echo "Step 4/5: Installing Unsloth with ROCm support..."
pip install "unsloth[rocm] @ git+https://github.com/unslothai/unsloth.git"

echo ""
echo "Step 5/5: Installing additional utilities..."
pip install sentencepiece protobuf scipy ninja einops

echo ""
echo "================================"
echo "Verifying installations..."
echo "================================"

python3 << 'EOF'
import sys

print("Python version:", sys.version)

try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    print(f"   ROCm available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU detected: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ PyTorch error: {e}")

try:
    import transformers
    print(f"✅ Transformers {transformers.__version__}")
except Exception as e:
    print(f"❌ Transformers error: {e}")

try:
    import unsloth
    print(f"✅ Unsloth installed")
except Exception as e:
    print(f"❌ Unsloth error: {e}")

try:
    import peft
    print(f"✅ PEFT {peft.__version__}")
except Exception as e:
    print(f"❌ PEFT error: {e}")

try:
    import trl
    print(f"✅ TRL {trl.__version__}")
except Exception as e:
    print(f"❌ TRL error: {e}")

EOF

echo ""
echo "================================"
echo "Installation Complete!"
echo "================================"
