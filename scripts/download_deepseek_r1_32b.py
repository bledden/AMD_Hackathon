#!/usr/bin/env python3
"""
Download DeepSeek-R1-Distill-Qwen-32B model for distillation
"""

from huggingface_hub import snapshot_download
import os
from pathlib import Path

def main():
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    output_dir = "/home/rocm-user/AMD_Hackathon/models/deepseek_r1_qwen32b"

    print(f"Downloading {model_id}...")
    print(f"Output directory: {output_dir}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Download model
    snapshot_download(
        repo_id=model_id,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
    )

    print(f"✓ Download complete!")
    print(f"✓ Model saved to: {output_dir}")

if __name__ == "__main__":
    main()
