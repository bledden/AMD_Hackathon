#!/usr/bin/env python3
"""
Generate Chain-of-Thought reasoning using llama.cpp with GGUF quantized models
This uses GGUF format for better ROCm support and true 1-bit quantization
Target: 50-60GB VRAM, 4-6 hours for 50K questions
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import time
import subprocess
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class LlamaCppCoTGenerator:
    """Generate CoT using llama.cpp with GGUF models"""

    def __init__(self, model_path: str, n_gpu_layers: int = -1):
        """
        Initialize llama.cpp generator

        Args:
            model_path: Path to GGUF model file
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
        """
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers

        logging.info(f"🚀 Initializing llama.cpp with GGUF model")
        logging.info(f"   Model: {model_path}")
        logging.info(f"   GPU layers: {n_gpu_layers}")

        # Try to import llama-cpp-python
        try:
            from llama_cpp import Llama
            self.Llama = Llama
            self.use_python_bindings = True
            logging.info("✅ Using llama-cpp-python bindings")
        except ImportError:
            logging.warning("⚠️  llama-cpp-python not found, will use CLI")
            self.use_python_bindings = False
            self.Llama = None

        # Initialize model if using Python bindings
        if self.use_python_bindings:
            self._init_model()

    def _init_model(self):
        """Initialize the llama.cpp model"""
        logging.info("📦 Loading model into memory...")

        self.model = self.Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=2048,  # Context window
            n_batch=512,
            verbose=False,
            use_mmap=True,
            use_mlock=False,
        )

        logging.info("✅ Model loaded successfully")

    def generate_cot(self, question: str, choices: Dict[str, str], max_tokens: int = 512) -> str:
        """
        Generate Chain-of-Thought reasoning for a single question

        Args:
            question: The question text
            choices: Dict of answer choices
            max_tokens: Max length of generated reasoning

        Returns:
            CoT reasoning string
        """
        # Format choices
        choices_text = "\n".join([f"{k}. {v}" for k, v in sorted(choices.items())])

        # Create prompt
        prompt = f"""You are a helpful assistant that provides step-by-step reasoning for multiple choice questions.

Question: {question}

Choices:
{choices_text}

Provide detailed step-by-step reasoning to solve this question. Think through:
1. What is being asked?
2. What knowledge is needed?
3. How do we eliminate wrong answers?
4. Why is one answer correct?

Reasoning:"""

        if self.use_python_bindings:
            # Use Python bindings
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                echo=False,
                stop=["Question:", "\n\n\n"]
            )
            reasoning = output['choices'][0]['text'].strip()
        else:
            # Use CLI (fallback)
            reasoning = self._generate_cli(prompt, max_tokens)

        return reasoning

    def _generate_cli(self, prompt: str, max_tokens: int) -> str:
        """Generate using llama.cpp CLI (fallback)"""
        # Save prompt to temp file
        prompt_file = "/tmp/llama_prompt.txt"
        with open(prompt_file, 'w') as f:
            f.write(prompt)

        # Run llama.cpp
        cmd = [
            "llama-cli",
            "-m", self.model_path,
            "-f", prompt_file,
            "-n", str(max_tokens),
            "--temp", "0.7",
            "--top-p", "0.9",
            "-ngl", str(self.n_gpu_layers)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            output = result.stdout.strip()
            # Extract generated text (after prompt)
            if "Reasoning:" in output:
                reasoning = output.split("Reasoning:")[-1].strip()
            else:
                reasoning = output
            return reasoning
        except Exception as e:
            logging.error(f"CLI generation failed: {e}")
            return ""

    def generate_batch(self, questions: List[Dict]) -> List[Dict]:
        """
        Generate CoT for all questions

        Args:
            questions: List of question dicts

        Returns:
            List of questions with added 'chain_of_thought' field
        """
        results = []

        logging.info(f"🔄 Generating CoT for {len(questions)} questions...")

        for i, q in enumerate(tqdm(questions, desc="CoT Generation")):
            try:
                # Get choices
                choices = q.get('choices', q.get('options', {}))
                if not choices:
                    logging.warning(f"Skipping question {i} with no choices")
                    results.append(q)
                    continue

                # Generate CoT
                cot = self.generate_cot(q['question'], choices)

                # Add to question
                enhanced_q = q.copy()
                enhanced_q['chain_of_thought'] = cot
                enhanced_q['cot_model'] = 'deepseek-v3.1-gguf'
                enhanced_q['cot_method'] = 'llama.cpp'
                results.append(enhanced_q)

                # Progress logging
                if (i + 1) % 100 == 0:
                    elapsed = (i + 1) / len(questions) * 100
                    logging.info(f"   Progress: {i+1}/{len(questions)} ({elapsed:.1f}%)")

            except Exception as e:
                logging.error(f"Error on question {i}: {e}")
                results.append(q)

        return results


def check_llama_cpp_installation():
    """Check if llama.cpp is available"""
    logging.info("🔍 Checking llama.cpp installation...")

    # Check Python bindings
    try:
        import llama_cpp
        logging.info("✅ llama-cpp-python is installed")
        return True
    except ImportError:
        logging.warning("⚠️  llama-cpp-python not found")

    # Check CLI
    try:
        result = subprocess.run(["llama-cli", "--version"], capture_output=True)
        if result.returncode == 0:
            logging.info("✅ llama-cli is available")
            return True
    except FileNotFoundError:
        pass

    logging.error("❌ llama.cpp not found!")
    logging.info("💡 Install with: pip install llama-cpp-python")
    return False


def find_gguf_model():
    """Find available GGUF model"""
    logging.info("🔍 Looking for GGUF model...")

    # Possible locations
    search_paths = [
        Path("/home/rocm-user/AMD_Hackathon/models"),
        Path("/home/rocm-user/models"),
        Path("./models"),
        Path.home() / ".cache" / "huggingface" / "hub"
    ]

    for search_path in search_paths:
        if not search_path.exists():
            continue

        # Look for GGUF files
        gguf_files = list(search_path.rglob("*.gguf"))
        if gguf_files:
            logging.info(f"✅ Found {len(gguf_files)} GGUF models in {search_path}")
            for f in gguf_files[:5]:  # Show first 5
                logging.info(f"   - {f.name}")
            return gguf_files[0]

    logging.warning("⚠️  No GGUF models found locally")
    logging.info("💡 Will need to download DeepSeek-V3 GGUF model")
    return None


def main():
    """Main pipeline"""
    start_time = time.time()

    logging.info("=" * 80)
    logging.info("🧠 CHAIN-OF-THOUGHT GENERATION WITH LLAMA.CPP (GGUF)")
    logging.info("=" * 80)

    # Check installation
    if not check_llama_cpp_installation():
        logging.error("❌ llama.cpp not available. Please install first:")
        logging.error("   pip install llama-cpp-python")
        logging.error("   or compile llama.cpp from source with ROCm support")
        return

    # Find model
    model_path = find_gguf_model()
    if model_path is None:
        logging.error("❌ No GGUF model found!")
        logging.info("💡 Please download DeepSeek-V3 GGUF first")
        logging.info("   We'll create a download script for this")
        return

    # Paths
    dataset_path = Path('data/comprehensive/full_50k_mcq.json')
    output_path = Path('data/enhanced/cot_enhanced_50k.json')

    # Load dataset
    logging.info(f"📂 Loading dataset from {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    logging.info(f"✅ Loaded {len(questions)} questions")

    # Initialize generator
    generator = LlamaCppCoTGenerator(str(model_path), n_gpu_layers=-1)

    # Generate CoT
    logging.info(f"🔄 Starting CoT generation...")
    logging.info(f"   Estimated time: 4-6 hours")
    enhanced = generator.generate_batch(questions)

    # Save
    logging.info(f"💾 Saving to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced, f, indent=2, ensure_ascii=False)

    # Stats
    with_cot = sum(1 for q in enhanced if 'chain_of_thought' in q and q['chain_of_thought'])
    avg_length = sum(len(q.get('chain_of_thought', '')) for q in enhanced) / len(enhanced)

    elapsed = time.time() - start_time
    logging.info("=" * 80)
    logging.info(f"✅ COMPLETE in {elapsed/3600:.2f} hours")
    logging.info(f"📊 Statistics:")
    logging.info(f"   Questions with CoT: {with_cot}/{len(enhanced)} ({with_cot/len(enhanced)*100:.1f}%)")
    logging.info(f"   Average CoT length: {avg_length:.0f} characters")
    logging.info(f"   Output: {output_path}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
