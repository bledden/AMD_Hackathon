#!/usr/bin/env python3
"""
Ensemble Voting System for 3-Model MCQ Answering
Combines Models #1 (LoRA), #2 (RSLoRA), #3 (DoRA) for maximum accuracy

Strategy:
- Probability-weighted voting (not simple majority)
- Each model outputs softmax probabilities for A, B, C, D
- Final answer = argmax(weighted average of probabilities)
- Target: 92-95% accuracy with ensemble

Individual Model Targets:
- Model #1 (LoRA r=64): 85-87%
- Model #2 (RSLoRA r=128): 88-92%
- Model #3 (DoRA r=64): 87-90%

Ensemble Expected:
- Diversity from different LoRA variants + ranks
- Probability voting reduces variance
- Target: 92-95% accuracy
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import numpy as np
from tqdm import tqdm
from unsloth import FastLanguageModel
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ModelEnsemble:
    """Ensemble of 3 LoRA variant models with probability-weighted voting"""

    def __init__(
        self,
        model1_path: str,
        model2_path: str,
        model3_path: str,
        weights: List[float] = [0.3, 0.4, 0.3]  # RSLoRA gets higher weight
    ):
        """
        Initialize ensemble with 3 models

        Args:
            model1_path: Path to Model #1 (LoRA r=64)
            model2_path: Path to Model #2 (RSLoRA r=128)
            model3_path: Path to Model #3 (DoRA r=64)
            weights: Voting weights [w1, w2, w3] (default: RSLoRA gets 0.4)
        """
        self.weights = np.array(weights)
        self.weights = self.weights / self.weights.sum()  # Normalize to sum to 1

        logging.info("🎯 Loading Ensemble Models...")
        logging.info(f"   Voting weights: Model #1: {self.weights[0]:.2f}, Model #2: {self.weights[1]:.2f}, Model #3: {self.weights[2]:.2f}")

        # Load Model #1 (LoRA)
        logging.info(f"\n📂 Loading Model #1 (LoRA r=64) from {model1_path}")
        self.model1, self.tokenizer1 = self._load_model(model1_path)
        logging.info("   ✅ Model #1 loaded")

        # Load Model #2 (RSLoRA)
        logging.info(f"\n📂 Loading Model #2 (RSLoRA r=128) from {model2_path}")
        self.model2, self.tokenizer2 = self._load_model(model2_path)
        logging.info("   ✅ Model #2 loaded")

        # Load Model #3 (DoRA)
        logging.info(f"\n📂 Loading Model #3 (DoRA r=64) from {model3_path}")
        self.model3, self.tokenizer3 = self._load_model(model3_path)
        logging.info("   ✅ Model #3 loaded")

        # Tokens for A, B, C, D
        self.answer_tokens = {
            'A': self.tokenizer1.encode('A', add_special_tokens=False)[0],
            'B': self.tokenizer1.encode('B', add_special_tokens=False)[0],
            'C': self.tokenizer1.encode('C', add_special_tokens=False)[0],
            'D': self.tokenizer1.encode('D', add_special_tokens=False)[0],
        }

        logging.info("\n✅ Ensemble initialized successfully!")
        logging.info(f"   Total VRAM: ~{torch.cuda.memory_allocated() / 1024**3:.2f}GB")

    def _load_model(self, model_path: str):
        """Load a single model with Unsloth"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)  # Enable inference mode
        return model, tokenizer

    def format_prompt(self, question: str, choices: Dict[str, str]) -> str:
        """Format question as prompt for Qwen2.5"""
        choices_text = "\n".join([f"{k}. {v}" for k, v in sorted(choices.items())])

        prompt = f"""<|im_start|>system
You are a helpful assistant that answers multiple choice questions accurately.<|im_end|>
<|im_start|>user
Question: {question}

Choices:
{choices_text}

Please select the correct answer (A, B, C, or D).<|im_end|>
<|im_start|>assistant
The correct answer is"""

        return prompt

    def get_answer_probabilities(self, model, tokenizer, prompt: str) -> np.ndarray:
        """
        Get probability distribution over A, B, C, D for a single model

        Returns:
            Array of [P(A), P(B), P(C), P(D)]
        """
        # Tokenize prompt
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        # Get logits for next token
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits

        # Extract logits for A, B, C, D tokens
        answer_logits = torch.tensor([
            logits[self.answer_tokens['A']].item(),
            logits[self.answer_tokens['B']].item(),
            logits[self.answer_tokens['C']].item(),
            logits[self.answer_tokens['D']].item(),
        ])

        # Convert to probabilities
        probs = F.softmax(answer_logits, dim=0).cpu().numpy()

        return probs

    def predict(self, question: str, choices: Dict[str, str]) -> Tuple[str, Dict[str, float]]:
        """
        Predict answer using ensemble voting

        Args:
            question: Question text
            choices: Dict of {A: choice_a, B: choice_b, C: choice_c, D: choice_d}

        Returns:
            (predicted_answer, confidence_scores)
        """
        # Format prompt
        prompt = self.format_prompt(question, choices)

        # Get probabilities from each model
        probs1 = self.get_answer_probabilities(self.model1, self.tokenizer1, prompt)
        probs2 = self.get_answer_probabilities(self.model2, self.tokenizer2, prompt)
        probs3 = self.get_answer_probabilities(self.model3, self.tokenizer3, prompt)

        # Weighted average of probabilities
        ensemble_probs = (
            self.weights[0] * probs1 +
            self.weights[1] * probs2 +
            self.weights[2] * probs3
        )

        # Get final answer
        answer_idx = np.argmax(ensemble_probs)
        answer = ['A', 'B', 'C', 'D'][answer_idx]

        # Return confidence scores
        confidence = {
            'A': ensemble_probs[0],
            'B': ensemble_probs[1],
            'C': ensemble_probs[2],
            'D': ensemble_probs[3],
            'individual_votes': {
                'model1': ['A', 'B', 'C', 'D'][np.argmax(probs1)],
                'model2': ['A', 'B', 'C', 'D'][np.argmax(probs2)],
                'model3': ['A', 'B', 'C', 'D'][np.argmax(probs3)],
            }
        }

        return answer, confidence


def validate_ensemble(
    ensemble: ModelEnsemble,
    val_data_path: str,
    num_samples: int = 100
):
    """Validate ensemble on holdout set"""
    logging.info(f"\n{'='*80}")
    logging.info(f"ENSEMBLE VALIDATION")
    logging.info(f"{'='*80}")

    # Load validation data
    with open(val_data_path) as f:
        val_data = json.load(f)

    # Validate on sample
    correct = 0
    total = 0
    detailed_results = []

    for item in tqdm(val_data[:num_samples], desc="Validating ensemble"):
        question = item['question']
        choices = item['choices']
        correct_answer = item['correct_answer']

        # Get ensemble prediction
        predicted, confidence = ensemble.predict(question, choices)

        # Check correctness
        is_correct = (predicted == correct_answer)
        if is_correct:
            correct += 1
        total += 1

        # Save detailed result
        detailed_results.append({
            'question': question,
            'correct_answer': correct_answer,
            'predicted_answer': predicted,
            'is_correct': is_correct,
            'confidence': confidence
        })

    accuracy = (correct / total) * 100

    logging.info(f"\n{'='*80}")
    logging.info(f"ENSEMBLE RESULTS (n={num_samples})")
    logging.info(f"{'='*80}")
    logging.info(f"Correct: {correct}/{total}")
    logging.info(f"Accuracy: {accuracy:.2f}%")
    logging.info(f"{'='*80}")

    # Analyze voting agreement
    agreement_all = sum(
        1 for r in detailed_results
        if len(set([
            r['confidence']['individual_votes']['model1'],
            r['confidence']['individual_votes']['model2'],
            r['confidence']['individual_votes']['model3']
        ])) == 1
    )
    agreement_any_2 = sum(
        1 for r in detailed_results
        if len(set([
            r['confidence']['individual_votes']['model1'],
            r['confidence']['individual_votes']['model2'],
            r['confidence']['individual_votes']['model3']
        ])) == 2
    )

    logging.info(f"\n📊 Voting Analysis:")
    logging.info(f"   All 3 models agree: {agreement_all}/{total} ({agreement_all/total*100:.1f}%)")
    logging.info(f"   Any 2 models agree: {agreement_any_2}/{total} ({agreement_any_2/total*100:.1f}%)")
    logging.info(f"   All disagree: {total-agreement_all-agreement_any_2}/{total} ({(total-agreement_all-agreement_any_2)/total*100:.1f}%)")

    return accuracy, detailed_results


def main():
    """Main validation pipeline"""
    logging.info("=" * 80)
    logging.info("🎯 ENSEMBLE VOTING SYSTEM - VALIDATION")
    logging.info("=" * 80)

    # Model paths
    model1_path = "/workspace/models/qwen2.5_72b_unsloth_curriculum/final_model"
    model2_path = "/workspace/models/model2_rslora_r128/final_model"
    model3_path = "/workspace/models/model3_dora_r64/final_model"

    # Validation data
    val_data_path = "/workspace/data/curriculum/val_5k.json"

    # Initialize ensemble
    # Give RSLoRA (Model #2) higher weight since it has highest expected accuracy
    ensemble = ModelEnsemble(
        model1_path=model1_path,
        model2_path=model2_path,
        model3_path=model3_path,
        weights=[0.3, 0.4, 0.3]  # Model #2 gets 40%
    )

    # Validate on first 100 questions
    accuracy, results = validate_ensemble(ensemble, val_data_path, num_samples=100)

    # Save results
    output_path = Path("/workspace/ensemble_validation_results.json")
    with open(output_path, 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'num_samples': len(results),
            'results': results
        }, f, indent=2)

    logging.info(f"\n✅ Results saved to {output_path}")
    logging.info(f"\n🎯 Final Ensemble Accuracy: {accuracy:.2f}%")

    if accuracy >= 92:
        logging.info("   🎉 TARGET ACHIEVED! (92%+ accuracy)")
    elif accuracy >= 87:
        logging.info("   ✅ Strong performance! (87%+ accuracy)")
    else:
        logging.info("   ⚠️  Below target - consider weight tuning")


if __name__ == "__main__":
    main()
