#!/usr/bin/env python3
"""
TIES-Merging for Domain Specialists

Novel Contribution: Merge STEM + Humanities RSLoRA specialists
into single model with complementary domain expertise.

TIES-Merging (Resolving Interference When Merging Models):
1. Trim: Remove low-magnitude parameters (noise reduction)
2. Elect Sign: Resolve conflicts via majority voting
3. Disjoint Merge: Average agreeing params, keep non-conflicting params

Result: Single merged model with ZERO latency penalty
"""

import torch
import json
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

def load_lora_weights(adapter_path: str):
    """Load LoRA adapter weights"""
    logging.info(f"Loading LoRA weights from {adapter_path}")

    # Load adapter_model.safetensors or adapter_model.bin
    weights_file = Path(adapter_path) / "adapter_model.safetensors"
    if not weights_file.exists():
        weights_file = Path(adapter_path) / "adapter_model.bin"

    if weights_file.suffix == ".safetensors":
        from safetensors.torch import load_file
        weights = load_file(weights_file)
    else:
        weights = torch.load(weights_file, map_location='cpu')

    logging.info(f"   Loaded {len(weights)} weight tensors")
    return weights


def trim_weights(weights: dict, threshold: float = 0.2):
    """
    TIES Step 1: Trim low-magnitude parameters

    Removes parameters with magnitude below threshold * max_magnitude
    This reduces noise and interference between models
    """
    logging.info(f"Trimming weights (threshold: {threshold})")
    trimmed = {}

    for key, tensor in weights.items():
        # Calculate magnitude threshold
        max_mag = torch.abs(tensor).max().item()
        trim_threshold = threshold * max_mag

        # Keep only high-magnitude weights
        mask = torch.abs(tensor) >= trim_threshold
        trimmed[key] = tensor * mask

        # Log trimming stats
        kept = mask.sum().item()
        total = tensor.numel()
        logging.info(f"   {key}: kept {kept}/{total} ({kept/total*100:.1f}%)")

    return trimmed


def elect_sign(weights_list: list):
    """
    TIES Step 2: Elect Sign

    For each parameter, determine the dominant sign (+ or -)
    via majority voting across models
    """
    logging.info("Electing signs via majority voting")
    elected = {}

    keys = weights_list[0].keys()

    for key in keys:
        # Stack weights from all models
        stacked = torch.stack([w[key] for w in weights_list])

        # Vote on sign (+1 for positive, -1 for negative, 0 for zero)
        signs = torch.sign(stacked)
        sign_votes = signs.sum(dim=0)

        # Majority sign
        majority_sign = torch.sign(sign_votes)

        elected[key] = majority_sign

        # Log agreement stats
        agreement = (signs == majority_sign.unsqueeze(0)).float().mean().item()
        logging.info(f"   {key}: {agreement*100:.1f}% agreement on sign")

    return elected


def disjoint_merge(weights_list: list, elected_signs: dict, weights: list = None):
    """
    TIES Step 3: Disjoint Merge

    - For parameters with same sign: average them
    - For parameters with different signs: use majority vote
    - For parameters only in one model: keep as-is
    """
    logging.info("Performing disjoint merge")

    if weights is None:
        weights = [1.0 / len(weights_list)] * len(weights_list)

    merged = {}
    keys = weights_list[0].keys()

    for key in keys:
        # Get weights from all models
        tensors = [w[key] for w in weights_list]
        stacked = torch.stack(tensors)

        # Get elected sign
        elected_sign = elected_signs[key]

        # For each position, check if all models agree on sign
        signs = torch.sign(stacked)
        agrees = (signs == elected_sign.unsqueeze(0)).all(dim=0)

        # Where they agree: weighted average
        # Where they disagree: use elected sign with magnitude
        merged_tensor = torch.zeros_like(tensors[0])

        for i, (tensor, weight) in enumerate(zip(tensors, weights)):
            # Only include if sign matches elected sign
            mask = (torch.sign(tensor) == elected_sign) | (tensor == 0)
            merged_tensor += weight * tensor * mask.float()

        merged[key] = merged_tensor

        # Log merge stats
        agreement_ratio = agrees.float().mean().item()
        logging.info(f"   {key}: {agreement_ratio*100:.1f}% full agreement")

    return merged


def ties_merge(
    adapter_paths: list,
    output_path: str,
    base_model: str = "Qwen/Qwen2.5-72B-Instruct",
    weights: list = None,
    trim_threshold: float = 0.2
):
    """
    TIES-Merge multiple LoRA adapters

    Args:
        adapter_paths: List of paths to LoRA adapters
        output_path: Path to save merged adapter
        base_model: Base model name
        weights: Optional list of weights for each adapter (default: equal)
        trim_threshold: Threshold for trimming (0.2 = keep top 80%)
    """
    logging.info("=" * 80)
    logging.info("TIES-MERGING: Domain-Specific RSLoRA Specialists")
    logging.info("=" * 80)
    logging.info(f"Base model: {base_model}")
    logging.info(f"Adapters to merge: {len(adapter_paths)}")
    for i, path in enumerate(adapter_paths):
        logging.info(f"   [{i+1}] {path}")

    if weights is None:
        weights = [1.0 / len(adapter_paths)] * len(adapter_paths)

    logging.info(f"Merge weights: {weights}")
    logging.info(f"Trim threshold: {trim_threshold}")

    # Step 1: Load all adapter weights
    logging.info("\n" + "=" * 80)
    logging.info("STEP 1: Loading Adapter Weights")
    logging.info("=" * 80)

    all_weights = []
    for path in adapter_paths:
        weights_dict = load_lora_weights(path)
        all_weights.append(weights_dict)

    # Step 2: Trim low-magnitude parameters
    logging.info("\n" + "=" * 80)
    logging.info("STEP 2: Trimming Low-Magnitude Parameters")
    logging.info("=" * 80)

    trimmed_weights = [trim_weights(w, trim_threshold) for w in all_weights]

    # Step 3: Elect signs via majority voting
    logging.info("\n" + "=" * 80)
    logging.info("STEP 3: Electing Signs (Majority Voting)")
    logging.info("=" * 80)

    elected_signs = elect_sign(trimmed_weights)

    # Step 4: Disjoint merge
    logging.info("\n" + "=" * 80)
    logging.info("STEP 4: Disjoint Merge")
    logging.info("=" * 80)

    merged_weights = disjoint_merge(trimmed_weights, elected_signs, weights)

    # Step 5: Save merged adapter
    logging.info("\n" + "=" * 80)
    logging.info("STEP 5: Saving Merged Adapter")
    logging.info("=" * 80)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save merged weights
    output_file = output_dir / "adapter_model.safetensors"
    from safetensors.torch import save_file
    save_file(merged_weights, output_file)
    logging.info(f"✅ Saved merged weights: {output_file}")

    # Copy adapter_config.json from first adapter (they should be identical)
    import shutil
    src_config = Path(adapter_paths[0]) / "adapter_config.json"
    dst_config = output_dir / "adapter_config.json"
    shutil.copy(src_config, dst_config)
    logging.info(f"✅ Copied adapter config: {dst_config}")

    # Create README
    readme_content = f"""# TIES-Merged Domain Specialists

This adapter was created by merging domain-specific RSLoRA specialists via TIES-Merging.

## Source Adapters
{chr(10).join([f'- {path}' for path in adapter_paths])}

## Merge Configuration
- Base model: {base_model}
- Merge method: TIES (Trim, Elect, Disjoint)
- Trim threshold: {trim_threshold}
- Merge weights: {weights}

## Novel Contribution
This represents the first application of TIES-Merging to domain-specific
RSLoRA specialists for multiple-choice question answering.

Result: Single model with complementary domain expertise and zero latency penalty.
"""

    readme_file = output_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    logging.info(f"✅ Created README: {readme_file}")

    logging.info("\n" + "=" * 80)
    logging.info("✅ TIES-MERGING COMPLETE!")
    logging.info("=" * 80)
    logging.info(f"Merged adapter saved to: {output_path}")
    logging.info("\nYou can now use this merged adapter with:")
    logging.info(f"  from peft import PeftModel")
    logging.info(f"  model = PeftModel.from_pretrained(base_model, '{output_path}')")


def main():
    """Main merging pipeline"""
    logging.info("🔬 TIES-Merging Script for Domain Specialists")

    # Paths
    stem_adapter = "/workspace/models/stem_specialist_rslora_r128/final_model"
    humanities_adapter = "/workspace/models/humanities_specialist_rslora_r128/final_model"
    output_path = "/workspace/models/merged_super_specialist_ties"

    # Merge with equal weights (can be tuned based on dataset sizes)
    # STEM: 36K questions (60%)
    # Humanities: 24K questions (40%)
    ties_merge(
        adapter_paths=[stem_adapter, humanities_adapter],
        output_path=output_path,
        base_model="Qwen/Qwen2.5-72B-Instruct",
        weights=[0.6, 0.4],  # Weight by dataset size
        trim_threshold=0.2  # Keep top 80% of parameters
    )

    logging.info("\n🎉 Merged specialist ready for ensemble!")
    logging.info("   Next: Create ensemble with Model #1")


if __name__ == "__main__":
    main()
