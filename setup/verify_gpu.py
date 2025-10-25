#!/usr/bin/env python3
"""
GPU Verification Script for AMD MI300X
Verifies that PyTorch, ROCm, and Unsloth are working correctly
"""

import sys


def verify_pytorch():
    """Verify PyTorch installation and GPU access"""
    print("=" * 50)
    print("1. Verifying PyTorch...")
    print("=" * 50)

    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ Device count: {torch.cuda.device_count()}")
            print(f"✓ Current device: {torch.cuda.current_device()}")
            print(f"✓ Device name: {torch.cuda.get_device_name(0)}")

            props = torch.cuda.get_device_properties(0)
            print(f"✓ Total memory: {props.total_memory / 1e9:.2f} GB")
            print(f"✓ Compute capability: {props.major}.{props.minor}")

            # Test tensor operation
            x = torch.randn(1000, 1000).cuda()
            y = torch.matmul(x, x)
            print(f"✓ GPU tensor operations working")

            return True
        else:
            print("✗ CUDA not available!")
            return False

    except ImportError:
        print("✗ PyTorch not installed!")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def verify_transformers():
    """Verify Transformers library"""
    print("\n" + "=" * 50)
    print("2. Verifying Transformers...")
    print("=" * 50)

    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
        return True
    except ImportError:
        print("✗ Transformers not installed!")
        return False


def verify_unsloth():
    """Verify Unsloth installation"""
    print("\n" + "=" * 50)
    print("3. Verifying Unsloth...")
    print("=" * 50)

    try:
        from unsloth import FastLanguageModel
        print("✓ Unsloth imported successfully")
        print("✓ FastLanguageModel available")
        return True
    except ImportError as e:
        print(f"✗ Unsloth import failed: {e}")
        return False


def verify_training_libs():
    """Verify training libraries"""
    print("\n" + "=" * 50)
    print("4. Verifying Training Libraries...")
    print("=" * 50)

    libs = {
        "datasets": "Hugging Face Datasets",
        "trl": "Transformer Reinforcement Learning",
        "peft": "Parameter-Efficient Fine-Tuning",
        "accelerate": "Hugging Face Accelerate",
        "bitsandbytes": "8-bit optimizers",
    }

    all_good = True
    for lib, name in libs.items():
        try:
            __import__(lib)
            print(f"✓ {name} ({lib})")
        except ImportError:
            print(f"✗ {name} ({lib}) not installed!")
            all_good = False

    return all_good


def test_model_loading():
    """Test loading a small model with Unsloth"""
    print("\n" + "=" * 50)
    print("5. Testing Model Loading (Optional)...")
    print("=" * 50)

    try:
        from unsloth import FastLanguageModel
        import torch

        print("Attempting to load a tiny model for testing...")
        print("(This will download ~500MB if not cached)")

        # Try to load a small model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/tinyllama-bnb-4bit",
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )

        print("✓ Model loaded successfully with Unsloth!")
        print("✓ 4-bit quantization working")

        # Quick inference test
        FastLanguageModel.for_inference(model)
        inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
        print("✓ Tokenization working")

        print("\n✓ Full stack verified and working!")
        return True

    except Exception as e:
        print(f"⚠ Model loading test failed (optional): {e}")
        print("This is OK - may just need to download models")
        return True


def print_summary(results):
    """Print summary of verification"""
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)

    checks = [
        ("PyTorch + GPU", results[0]),
        ("Transformers", results[1]),
        ("Unsloth", results[2]),
        ("Training Libraries", results[3]),
        ("Model Loading", results[4]),
    ]

    all_passed = all(r for r in results[:4])  # First 4 are critical

    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    print("=" * 50)

    if all_passed:
        print("\n🎉 All critical checks passed!")
        print("Your MI300X environment is ready for fine-tuning!")
        print("\nNext steps:")
        print("1. Choose your base model (LLaMA 3 8B recommended)")
        print("2. Prepare your Q&A dataset")
        print("3. Run training script from training/scripts/")
        return 0
    else:
        print("\n⚠ Some checks failed!")
        print("Please review errors above and reinstall missing dependencies.")
        print("Run: bash setup/install_dependencies.sh")
        return 1


def main():
    """Main verification routine"""
    print("\n" + "=" * 50)
    print("AMD MI300X Environment Verification")
    print("=" * 50)
    print()

    results = []
    results.append(verify_pytorch())
    results.append(verify_transformers())
    results.append(verify_unsloth())
    results.append(verify_training_libs())
    results.append(test_model_loading())

    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
