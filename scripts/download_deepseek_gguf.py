#!/usr/bin/env python3
"""
Download DeepSeek-V3 in GGUF format with 1-bit quantization
Target: ~50-60GB file size for 192GB VRAM
"""

import logging
from pathlib import Path
import subprocess
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_gguf_model():
    """Download DeepSeek-V3 GGUF model from HuggingFace"""

    logging.info("=" * 80)
    logging.info("📥 DOWNLOADING DEEPSEEK-V3 GGUF MODEL")
    logging.info("=" * 80)

    # Model options (in order of preference)
    model_options = [
        {
            'repo': 'bartowski/DeepSeek-V3-GGUF',
            'file': 'DeepSeek-V3-IQ1_M.gguf',
            'size': '~50GB',
            'quant': '1-bit (IQ1_M)',
            'priority': 1
        },
        {
            'repo': 'bartowski/DeepSeek-V3-GGUF',
            'file': 'DeepSeek-V3-IQ1_S.gguf',
            'size': '~45GB',
            'quant': '1-bit (IQ1_S)',
            'priority': 2
        },
        {
            'repo': 'bartowski/DeepSeek-V3-GGUF',
            'file': 'DeepSeek-V3-Q2_K.gguf',
            'size': '~90GB',
            'quant': '2-bit',
            'priority': 3
        },
        {
            'repo': 'unsloth/DeepSeek-V3-GGUF',
            'file': 'DeepSeek-V3-Q4_K_M.gguf',
            'size': '~170GB',
            'quant': '4-bit',
            'priority': 4
        }
    ]

    # Display options
    logging.info("Available quantization options:")
    for i, opt in enumerate(model_options, 1):
        logging.info(f"  {i}. {opt['quant']} - {opt['size']} - {opt['repo']}/{opt['file']}")

    # Create models directory
    models_dir = Path('/home/rocm-user/AMD_Hackathon/models')
    models_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"\n📁 Download directory: {models_dir}")

    # Try to download in order of priority
    for opt in model_options:
        repo = opt['repo']
        filename = opt['file']
        output_path = models_dir / filename

        if output_path.exists():
            logging.info(f"✅ Model already exists: {output_path}")
            logging.info(f"   Size: {output_path.stat().st_size / 1024**3:.2f}GB")
            return str(output_path)

        logging.info(f"\n🔄 Attempting to download: {opt['quant']}")
        logging.info(f"   Repo: {repo}")
        logging.info(f"   File: {filename}")
        logging.info(f"   Expected size: {opt['size']}")

        # Use huggingface-cli to download
        cmd = [
            'huggingface-cli',
            'download',
            repo,
            filename,
            '--local-dir', str(models_dir),
            '--local-dir-use-symlinks', 'False'
        ]

        try:
            logging.info(f"   Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            if output_path.exists():
                size_gb = output_path.stat().st_size / 1024**3
                logging.info(f"✅ Download successful!")
                logging.info(f"   Path: {output_path}")
                logging.info(f"   Size: {size_gb:.2f}GB")
                return str(output_path)

        except subprocess.CalledProcessError as e:
            logging.warning(f"⚠️  Failed to download {filename}")
            logging.warning(f"   Error: {e.stderr}")
            logging.info(f"   Trying next option...")
            continue

        except FileNotFoundError:
            logging.error("❌ huggingface-cli not found!")
            logging.info("💡 Install with: pip install huggingface-hub[cli]")
            break

    # If we get here, no downloads succeeded
    logging.error("❌ Failed to download any GGUF model")
    logging.info("\n💡 Manual download options:")
    logging.info("   1. Visit: https://huggingface.co/bartowski/DeepSeek-V3-GGUF")
    logging.info("   2. Download DeepSeek-V3-IQ1_M.gguf (~50GB)")
    logging.info(f"   3. Place in: {models_dir}")

    return None


def install_llama_cpp_python():
    """Install llama-cpp-python with ROCm support"""

    logging.info("\n" + "=" * 80)
    logging.info("📦 INSTALLING LLAMA-CPP-PYTHON WITH ROCM SUPPORT")
    logging.info("=" * 80)

    # Check if already installed
    try:
        import llama_cpp
        logging.info("✅ llama-cpp-python already installed")
        return True
    except ImportError:
        pass

    logging.info("Installing llama-cpp-python with ROCm support...")

    # Set ROCm environment variables
    env = os.environ.copy()
    env['CMAKE_ARGS'] = '-DLLAMA_HIPBLAS=on -DAMDGPU_TARGETS=gfx90a,gfx942'
    env['FORCE_CMAKE'] = '1'

    cmd = [
        'pip', 'install',
        'llama-cpp-python',
        '--no-cache-dir',
        '--force-reinstall'
    ]

    try:
        logging.info(f"Running: {' '.join(cmd)}")
        logging.info("This may take 5-10 minutes to compile...")

        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=True,
            text=True
        )

        logging.info("✅ Installation successful!")
        return True

    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Installation failed: {e.stderr}")
        logging.info("\n💡 Manual installation:")
        logging.info("   CMAKE_ARGS='-DLLAMA_HIPBLAS=on' pip install llama-cpp-python --force-reinstall --no-cache-dir")
        return False


def main():
    """Main setup pipeline"""

    logging.info("🚀 SETUP: DeepSeek-V3 GGUF + llama.cpp for ROCm")
    logging.info("")

    # Step 1: Install llama-cpp-python
    logging.info("STEP 1: Install llama-cpp-python with ROCm")
    if not install_llama_cpp_python():
        logging.error("Failed to install llama-cpp-python")
        return

    # Step 2: Download model
    logging.info("\nSTEP 2: Download GGUF model")
    model_path = download_gguf_model()

    if model_path:
        logging.info("\n" + "=" * 80)
        logging.info("✅ SETUP COMPLETE!")
        logging.info("=" * 80)
        logging.info(f"Model ready at: {model_path}")
        logging.info("\nNext step:")
        logging.info("  python3 scripts/generate_cot_llama_cpp.py")
        logging.info("=" * 80)
    else:
        logging.error("\n❌ Setup incomplete - model download failed")
        logging.info("Please download manually and run CoT generation script")


if __name__ == "__main__":
    main()
