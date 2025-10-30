#!/usr/bin/env python3
"""
TIES-Merge 3 RSLoRA Adapters into Unified Ensemble
Combines STEM + Humanities + Math specialists using TIES algorithm
"""

import json
import logging
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_adapter_weights(adapter_path: str) -> dict:
    """Load LoRA adapter weights from safetensors"""
    logging.info(f"📂 Loading adapter: {adapter_path}")

    model_file = Path(adapter_path) / "adapter_model.safetensors"
    if not model_file.exists():
        raise FileNotFoundError(f"Adapter not found: {model_file}")

    weights = load_file(str(model_file))
    logging.info(f"   ✅ Loaded {len(weights)} weight tensors")

    return weights

def ties_merge_task_vectors(
    task_vectors: list[dict],
    density: float = 0.7,
    majority_sign_method: str = "total"
) -> dict:
    """
    TIES-Merging: Trim, Elect Sign, Disjoint Merge

    Args:
        task_vectors: List of adapter weight dicts
        density: Keep top-k% by magnitude (0.7 = keep 70%)
        majority_sign_method: "total" or "frequency"

    Returns:
        Merged weight dict
    """
    logging.info(f"🔀 TIES-Merging {len(task_vectors)} adapters...")
    logging.info(f"   Density: {density} (keep top {density*100:.0f}% weights)")

    merged_weights = {}
    all_keys = set(task_vectors[0].keys())

    for key in tqdm(all_keys, desc="Merging weights"):
        # Stack all tensors for this parameter
        tensors = [tv[key].float() for tv in task_vectors]
        stacked = torch.stack(tensors)

        # Step 1: TRIM - Keep only top-k% by magnitude
        abs_values = torch.abs(stacked)
        threshold = torch.quantile(abs_values, 1.0 - density)
        mask = abs_values >= threshold
        trimmed = torch.where(mask, stacked, torch.zeros_like(stacked))

        # Step 2: ELECT SIGN - Resolve sign conflicts
        # Count positive vs negative for each parameter
        signs = torch.sign(trimmed)
        sign_sum = signs.sum(dim=0)

        if majority_sign_method == "total":
            # Use sign with larger total magnitude
            pos_mask = signs > 0
            neg_mask = signs < 0
            pos_sum = (trimmed * pos_mask.float()).sum(dim=0)
            neg_sum = (trimmed * neg_mask.float()).abs().sum(dim=0)
            majority_sign = torch.sign(pos_sum - neg_sum)
        else:
            # Use most frequent sign
            majority_sign = torch.sign(sign_sum)

        # Keep only weights matching majority sign
        aligned = torch.where(
            signs == majority_sign.unsqueeze(0),
            trimmed,
            torch.zeros_like(trimmed)
        )

        # Step 3: DISJOINT MERGE - Average aligned weights
        count = (aligned != 0).sum(dim=0).float()
        count = torch.where(count == 0, torch.ones_like(count), count)
        merged = aligned.sum(dim=0) / count

        merged_weights[key] = merged

    logging.info("   ✅ TIES-merge complete")
    return merged_weights

def load_adapter_config(adapter_path: str) -> dict:
    """Load adapter configuration"""
    config_file = Path(adapter_path) / "adapter_config.json"
    with open(config_file) as f:
        return json.load(f)

def main():
    logging.info("=" * 80)
    logging.info("🎯 TIES-MERGING 3 RSLoRA ADAPTERS")
    logging.info("=" * 80)

    # Adapter paths
    adapters = [
        {
            "name": "STEM",
            "path": "/workspace/models/stem_specialist_rslora_r128/final_model",
            "weight": 1.0  # Equal weighting
        },
        {
            "name": "Humanities",
            "path": "/workspace/models/humanities_specialist_rslora_r128/final_model",
            "weight": 1.0
        },
        {
            "name": "Math",
            "path": "/workspace/models/math_specialist_rslora_r128/final_model",
            "weight": 1.0
        }
    ]

    output_dir = Path("/workspace/models/ties_merged_ensemble_r128")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all adapters
    logging.info("\n📦 Loading adapters...")
    task_vectors = []
    for adapter in adapters:
        logging.info(f"\n   {adapter['name']}")
        weights = load_adapter_weights(adapter['path'])

        # Apply adapter weight (for weighted merging)
        if adapter['weight'] != 1.0:
            weights = {k: v * adapter['weight'] for k, v in weights.items()}

        task_vectors.append(weights)

    # Load config from first adapter (they should all match)
    base_config = load_adapter_config(adapters[0]['path'])
    logging.info(f"\n⚙️  Adapter config:")
    logging.info(f"   r = {base_config['r']}")
    logging.info(f"   alpha = {base_config['lora_alpha']}")
    logging.info(f"   use_rslora = {base_config.get('use_rslora', False)}")

    # TIES-merge
    logging.info("\n" + "=" * 80)
    merged_weights = ties_merge_task_vectors(
        task_vectors,
        density=0.7,  # Keep 70% of weights
        majority_sign_method="total"
    )
    logging.info("=" * 80)

    # Save merged adapter
    output_file = output_dir / "adapter_model.safetensors"
    logging.info(f"\n💾 Saving merged adapter to {output_file}...")
    save_file(merged_weights, str(output_file))

    # Save config
    merged_config = base_config.copy()
    merged_config["merged_adapters"] = [a["name"] for a in adapters]
    merged_config["merge_method"] = "TIES"
    merged_config["merge_density"] = 0.7

    config_file = output_dir / "adapter_config.json"
    with open(config_file, 'w') as f:
        json.dump(merged_config, f, indent=2)

    logging.info(f"   ✅ Config saved: {config_file}")

    # Calculate merged adapter size
    adapter_size = output_file.stat().st_size / (1024**3)
    logging.info(f"\n📊 Merged adapter statistics:")
    logging.info(f"   Size: {adapter_size:.2f} GB")
    logging.info(f"   Parameters merged: {len(merged_weights)}")
    logging.info(f"   Source adapters: {len(adapters)}")
    logging.info(f"   Coverage: STEM (13K) + Humanities (24K) + Math (7.5K) = ~45K questions")

    logging.info("\n" + "=" * 80)
    logging.info("✅ TIES-MERGE COMPLETE")
    logging.info(f"   Output: {output_dir}")
    logging.info("=" * 80)

    logging.info("\n📝 Next steps:")
    logging.info("   1. Test merged adapter with base model")
    logging.info("   2. Build Q-Agent and A-Agent wrappers")
    logging.info("   3. Validate speed (<10s / <6s limits)")
    logging.info("   4. Run end-to-end tournament test")

if __name__ == "__main__":
    main()
