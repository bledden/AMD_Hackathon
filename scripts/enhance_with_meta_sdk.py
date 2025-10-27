#!/usr/bin/env python3
"""
Use Meta's Synthetic Data Kit to enhance our 50K dataset with CoT reasoning
This is our competitive advantage!
"""

import json
from pathlib import Path
from tqdm import tqdm
import os

# Meta's SDK uses vLLM for generation
# We'll use it to generate CoT reasoning for our verified questions

def setup_vllm_server():
    """Setup vLLM server for Meta's SDK"""
    print("="*70)
    print("SETTING UP META'S SYNTHETIC DATA KIT")
    print("="*70)

    print("\n📦 Installing vLLM for local inference...")

    # Meta's SDK needs vLLM running locally
    # We'll use Qwen2.5-7B for fast CoT generation

    setup_script = """
#!/bin/bash

# Start vLLM server with Qwen2.5-7B
cd /home/rocm-user/AMD_Hackathon

# Kill any existing vLLM servers
pkill -f vllm.entrypoints.openai.api_server || true

# Start vLLM with Qwen2.5-7B (fast for CoT generation)
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 2048 \
    > vllm_server.log 2>&1 &

echo "Waiting for vLLM server to start..."
sleep 30

# Test if server is up
curl -s http://localhost:8000/v1/models || echo "Server starting..."

echo "vLLM server ready!"
"""

    return setup_script

def convert_to_meta_format(questions: list) -> list:
    """Convert our questions to Meta SDK input format"""

    print("\n📝 Converting questions to Meta SDK format...")

    # Meta SDK expects specific format for Q&A generation
    formatted_qa_pairs = []

    for q in tqdm(questions, desc="Formatting"):
        # Skip if no choices
        choices = q.get('choices', q.get('options', {}))
        if not choices or not isinstance(choices, dict):
            continue

        # Format as QA pair for Meta SDK
        qa_pair = {
            'question': q['question'],
            'answer': f"{q['correct_answer']}. {choices.get(q['correct_answer'], '')}",
            'context': json.dumps(choices),
            'metadata': {
                'source': q.get('source', 'unknown'),
                'domain': q.get('domain', 'general'),
                'original_format': 'mcq'
            }
        }
        formatted_qa_pairs.append(qa_pair)

    print(f"✅ Formatted {len(formatted_qa_pairs)} Q&A pairs")
    return formatted_qa_pairs

def generate_cot_with_meta_sdk(qa_pairs: list) -> list:
    """Use Meta's SDK to generate CoT reasoning"""

    print("\n🧠 Generating Chain-of-Thought with Meta's SDK...")
    print("This will take ~2-3 hours for 50K questions")

    # Meta SDK CLI command for CoT generation
    sdk_command = """
# Use Meta's synthetic-data-kit CLI
cd /home/rocm-user/AMD_Hackathon

# Generate CoT examples
python3 -m synthetic_data_kit.cli \
    --input data/comprehensive/meta_format_qa.json \
    --output data/cot_enhanced/meta_generated_cot.json \
    --task create \
    --type cot \
    --model http://localhost:8000/v1 \
    --num-examples -1 \
    --judge-model http://localhost:8000/v1 \
    --min-quality-score 0.7

echo "CoT generation complete!"
"""

    return sdk_command

def create_meta_sdk_pipeline():
    """Create complete pipeline using Meta's SDK"""

    pipeline_script = """#!/bin/bash
set -e

echo "=============================================================="
echo "META'S SYNTHETIC DATA KIT - COT GENERATION PIPELINE"
echo "=============================================================="

cd /home/rocm-user/AMD_Hackathon

# Step 1: Start vLLM server
echo "Step 1: Starting vLLM server..."
pkill -f vllm.entrypoints.openai.api_server || true
sleep 2

python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 2048 \
    > vllm_server.log 2>&1 &

echo "Waiting for server to start..."
sleep 30

# Verify server is up
curl -s http://localhost:8000/v1/models | grep -q "Qwen" && echo "✅ vLLM server ready!" || echo "⚠️ Server may not be ready"

# Step 2: Convert our dataset to Meta format
echo ""
echo "Step 2: Converting dataset to Meta format..."
python3 scripts/convert_to_meta_format.py

# Step 3: Generate CoT with Meta SDK
echo ""
echo "Step 3: Generating CoT reasoning with Meta SDK..."
python3 -c "
from synthetic_data_kit import create_qa_dataset, DatasetConfig
import json

# Load our formatted Q&A pairs
with open('data/comprehensive/meta_format_qa.json') as f:
    qa_pairs = json.load(f)

print(f'Loaded {len(qa_pairs)} Q&A pairs')

# Configure Meta SDK for CoT generation
config = DatasetConfig(
    qa_pairs=qa_pairs,
    model_endpoint='http://localhost:8000/v1',
    output_format='cot',
    quality_threshold=0.7,
    batch_size=8
)

# Generate CoT examples
print('Generating CoT reasoning...')
cot_dataset = create_qa_dataset(config)

# Save enhanced dataset
with open('data/cot_enhanced/meta_sdk_cot.json', 'w') as f:
    json.dump(cot_dataset, f, indent=2)

print(f'✅ Generated {len(cot_dataset)} CoT examples')
"

echo ""
echo "=============================================================="
echo "✅ COT GENERATION COMPLETE!"
echo "=============================================================="
echo "Output: data/cot_enhanced/meta_sdk_cot.json"
"""

    return pipeline_script

def main():
    """Main entry point"""

    print("="*70)
    print("ENHANCING 50K DATASET WITH META'S SYNTHETIC DATA KIT")
    print("="*70)

    # Load our 50K dataset
    dataset_path = Path("data/comprehensive/full_50k_dataset.json")

    if not dataset_path.exists():
        print("❌ Dataset not found!")
        print("Make sure download_50k_comprehensive_dataset.py completed")
        return

    print("\n📂 Loading dataset...")
    with open(dataset_path) as f:
        questions = json.load(f)

    print(f"✅ Loaded {len(questions)} questions")

    # Convert to Meta SDK format
    qa_pairs = convert_to_meta_format(questions)

    # Save in Meta format
    meta_format_path = Path("data/comprehensive/meta_format_qa.json")
    meta_format_path.parent.mkdir(parents=True, exist_ok=True)

    with open(meta_format_path, 'w') as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"\n✅ Saved Meta SDK format to: {meta_format_path}")

    # Create pipeline script
    pipeline = create_meta_sdk_pipeline()

    pipeline_path = Path("scripts/run_meta_sdk_cot.sh")
    with open(pipeline_path, 'w') as f:
        f.write(pipeline)

    pipeline_path.chmod(0o755)

    print(f"\n✅ Created pipeline script: {pipeline_path}")

    print("\n" + "="*70)
    print("🚀 NEXT STEPS")
    print("="*70)
    print("\nRun the pipeline on the server:")
    print("  bash scripts/run_meta_sdk_cot.sh")
    print("\nThis will:")
    print("  1. Start vLLM server with Qwen2.5-7B")
    print("  2. Generate CoT reasoning for all 50K questions")
    print("  3. Filter by quality (>0.7 score)")
    print("  4. Output training-ready dataset")
    print("\nEstimated time: 2-3 hours")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
