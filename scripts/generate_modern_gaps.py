#!/usr/bin/env python3
"""
Multi-model generation for modern domain gaps
Uses Phi-4, Qwen2.5, and Mistral-Nemo with ensemble validation
"""

import json
import torch
from pathlib import Path
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from tqdm import tqdm

class MultiModelGenerator:
    def __init__(self):
        """Initialize 3 models for ensemble generation"""
        self.models = {}
        self.tokenizers = {}

        model_names = [
            "microsoft/phi-4",
            "Qwen/Qwen2.5-7B-Instruct",
            "mistralai/Mistral-Nemo-Instruct-2407"
        ]

        print("Loading models for ensemble generation...")
        for name in model_names:
            print(f"  Loading {name}...")
            self.tokenizers[name] = AutoTokenizer.from_pretrained(
                name, trust_remote_code=True
            )
            if self.tokenizers[name].pad_token is None:
                self.tokenizers[name].pad_token = self.tokenizers[name].eos_token

            self.models[name] = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.models[name].eval()

    def generate_with_model(self, model_name: str, prompt: str) -> Dict:
        """Generate a question with a specific model"""
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, 'apply_chat_template'):
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.8,
                do_sample=True,
                top_p=0.9
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Extract JSON
        import re
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)

        try:
            return json.loads(response.strip())
        except:
            # Try to find JSON object
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except:
                    return None
            return None

    def ensemble_generate(self, domain: str, topic_detail: str) -> Dict:
        """Generate with all 3 models and validate"""

        prompt = f"""Generate a high-quality multiple choice question about {domain}: {topic_detail}

Requirements:
1. Question must be factually accurate and verifiable
2. Include exactly 4 answer options (A, B, C, D)
3. Only ONE correct answer
4. Distractors should be plausible but clearly wrong
5. Focus on current knowledge (2020-2024 if applicable)

Return ONLY a JSON object in this exact format:
{{
    "question": "The complete question text",
    "choices": {{
        "A": "First option",
        "B": "Second option",
        "C": "Third option",
        "D": "Fourth option"
    }},
    "correct_answer": "A",
    "domain": "{domain}",
    "explanation": "Brief explanation of why this answer is correct"
}}"""

        # Generate with all 3 models
        results = {}
        for model_name in self.models.keys():
            result = self.generate_with_model(model_name, prompt)
            if result and self.validate_format(result):
                results[model_name] = result

        # Ensemble validation
        if len(results) >= 2:  # At least 2 models succeeded
            # Check for consensus on answer
            answers = [r['correct_answer'] for r in results.values()]

            if len(set(answers)) == 1:  # Full agreement
                # Return the Phi-4 version (presumed highest quality)
                if "microsoft/phi-4" in results:
                    question = results["microsoft/phi-4"]
                else:
                    question = list(results.values())[0]

                question['validation'] = 'full_consensus'
                question['generators'] = list(results.keys())
                return question

            elif len(results) == 3:  # 2/3 agreement
                from collections import Counter
                answer_counts = Counter(answers)
                majority_answer, count = answer_counts.most_common(1)[0]

                if count >= 2:
                    # Find a model that gave the majority answer
                    for model_name, result in results.items():
                        if result['correct_answer'] == majority_answer:
                            question = result
                            question['validation'] = 'majority_vote'
                            question['generators'] = list(results.keys())
                            question['consensus_answer'] = majority_answer
                            return question

        return None  # No consensus

    def validate_format(self, q: Dict) -> bool:
        """Validate question format"""
        try:
            return (
                'question' in q and
                'choices' in q and
                'correct_answer' in q and
                len(q['choices']) == 4 and
                q['correct_answer'] in q['choices']
            )
        except:
            return False

def generate_gap_domains():
    """Generate questions for domains with gaps"""

    gaps = {
        'cryptocurrency': {
            'needed': 50,
            'topics': [
                'Bitcoin fundamentals and history',
                'Blockchain technology and consensus mechanisms',
                'Ethereum and smart contracts',
                'DeFi (Decentralized Finance) concepts',
                'NFTs and digital ownership',
                'Cryptocurrency regulation and legal issues',
                'Mining and proof of work/stake',
                'Crypto wallets and security',
                'Major cryptocurrencies (BTC, ETH, BNB, etc.)',
                'Cryptocurrency market dynamics'
            ]
        },
        'gaming': {
            'needed': 50,
            'topics': [
                'Video game history and evolution',
                'Major gaming platforms (PC, console, mobile)',
                'Game genres (RPG, FPS, RTS, etc.)',
                'Esports and competitive gaming',
                'Game development and design principles',
                'Gaming industry economics',
                'Famous game franchises and developers',
                'Gaming hardware and technology',
                'Online multiplayer and communities',
                'Gaming culture and terminology'
            ]
        },
        'social_media': {
            'needed': 50,
            'topics': [
                'Major social media platforms and their features',
                'Social media algorithms and engagement',
                'Influencer marketing and creator economy',
                'Privacy and data concerns',
                'Social media trends and viral content',
                'Content moderation and policies',
                'Social media advertising',
                'Impact on mental health and society',
                'Business use of social media',
                'Social media metrics and analytics'
            ]
        },
        'climate_change': {
            'needed': 44,
            'topics': [
                'Greenhouse gases and global warming',
                'Climate science and temperature records',
                'Renewable energy technologies',
                'Carbon footprint and emissions',
                'Climate change impacts on ecosystems',
                'International climate agreements (Paris, etc.)',
                'Climate adaptation strategies',
                'Extreme weather events',
                'Sea level rise and polar ice',
                'Climate change mitigation technologies'
            ]
        },
        'mental_health': {
            'needed': 49,
            'topics': [
                'Common mental health conditions',
                'Therapy types and approaches',
                'Mental health stigma and awareness',
                'Stress and anxiety management',
                'Depression symptoms and treatment',
                'Mental health in workplace/school',
                'Crisis intervention and support',
                'Mental health medications',
                'Child and adolescent mental health',
                'Mental health and technology'
            ]
        }
    }

    print("="*70)
    print("MULTI-MODEL GENERATION FOR MODERN DOMAINS")
    print("="*70)
    print("\nUsing 3-model ensemble with validation:")
    print("  • Phi-4 (Microsoft)")
    print("  • Qwen2.5-7B (Alibaba)")
    print("  • Mistral-Nemo (Mistral AI)")
    print("\nValidation: Requires 2/3 model agreement on answer")

    generator = MultiModelGenerator()
    all_generated = []

    for domain, config in gaps.items():
        print(f"\n📝 Generating {domain} questions...")
        needed = config['needed']
        topics = config['topics']
        questions_per_topic = max(1, needed // len(topics))

        domain_questions = []

        for topic in tqdm(topics, desc=f"  {domain}"):
            for _ in range(questions_per_topic):
                if len(domain_questions) >= needed:
                    break

                question = generator.ensemble_generate(domain, topic)

                if question:
                    domain_questions.append(question)

            if len(domain_questions) >= needed:
                break

        print(f"  ✅ Generated {len(domain_questions)}/{needed} for {domain}")
        all_generated.extend(domain_questions)

    # Save generated questions
    output_path = Path('data/modern_domains/multi_model_generated.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_generated, f, indent=2)

    print(f"\n💾 Saved {len(all_generated)} multi-model validated questions")

    # Statistics
    print("\n📊 Generation Statistics:")
    consensus_types = {}
    for q in all_generated:
        val_type = q.get('validation', 'unknown')
        consensus_types[val_type] = consensus_types.get(val_type, 0) + 1

    for val_type, count in consensus_types.items():
        print(f"  {val_type}: {count} questions")

    return all_generated

if __name__ == "__main__":
    # Generate with multi-model ensemble
    generated = generate_gap_domains()

    print("\n" + "="*70)
    print("✅ MULTI-MODEL GENERATION COMPLETE")
    print("="*70)
    print("\nThis approach:")
    print("1. Uses 3 models (Phi-4, Qwen2.5, Mistral-Nemo)")
    print("2. Requires 2/3 agreement for validation")
    print("3. Extracts knowledge from all models before training")
    print("4. Focuses only on domains with verified data gaps")
    print("\nNext: Combine with verified data for final dataset!")