# AMD Hackathon: Final Strategy Specifications

**Last Updated**: October 27, 2025
**Deadline**: Wednesday, October 29, 2025 @ 7:00 PM PT

---

## Table of Contents

- [Competition Requirements](#competition-requirements)
- [Hardware & Software Constraints](#hardware--software-constraints)
- [Our Model Architecture](#our-model-architecture)
- [Teacher Model Selection & Rationale](#teacher-model-selection--rationale)
- [Training Configuration](#training-configuration)
- [Novel Contribution Summary](#novel-contribution-summary)

---

## Competition Requirements

### Format

**Two Agents Required:**

1. **Q-Agent (Question Generator)**
   - Input: Topic string
   - Output: JSON with question, 4 choices (A/B/C/D), answer, explanation
   - **Speed limit**: < 10 seconds per question
   - **Correctness**: Oracle verifies topic, question, choices, answer

2. **A-Agent (Answerer)**
   - Input: Question + 4 choices
   - Output: JSON with answer letter (A/B/C/D), reasoning
   - **Speed limit**: < 6 seconds per answer
   - **Accuracy**: Compared against Q-Agent's correct answer

### Scoring

**Tournament Format**: Bracket-style, 2 innings per match, 20 questions each inning

**Your Score** = (Opponent's failed answers) + (Your correct answers)

**Example**:
- Your A-Agent answers 17/20 correctly: +17 points
- Opponent's A-Agent fails 6/20 of your questions: +6 points
- **Total**: 23 points per inning

### Critical Rules

1. ❌ **NO RAG** (Retrieval Augmented Generation)
2. ❌ **NO adversarial approaches** (jailbreaking, hallucination attacks)
3. ✅ **English only**
4. ✅ **Token limits**: Stay within max_tokens in yaml files
5. ✅ **Speed limits**: Strictly enforced (10s/6s)
6. ✅ **Submission files**:
   - `agents/question_agent.py`
   - `agents/question_model.py`
   - `agents/answer_agent.py`
   - `agents/answer_model.py`

### Submission Process

- **NO upload needed** - organizers collect directly from your Jupyter server
- Models must be loadable and generate correct JSON formats
- Invocation commands:
  ```bash
  python -m agents.question_agent --output_file "outputs/questions.json" --num_questions 20 --verbose
  python -m agents.answer_agent --input_file "outputs/filtered_questions.json" --output_file "outputs/answers.json" --verbose
  ```

---

## Hardware & Software Constraints

### Hardware (Provided by AMD/DigitalOcean)

**GPU Instance Specs:**
- **Platform**: DigitalOcean AMD Developer Cloud (amd.digitalocean.com)
- **GPU**: 1× AMD MI300X
  - **VRAM**: 192 GB
  - **Compute**: gfx942 architecture
- **CPU**: 20 vCPU
- **RAM**: 240 GB
- **Storage**:
  - Boot disk: 720 GB NVMe
  - Scratch disk: 5 TB NVMe
- **Cost**: $1.99/hour

**Current Disk Usage**: 54% utilized (325 GB available)

### Software Stack (Required)

**Base Image**:
- **MUST USE**: ROCm 6.4.0 (NOT PyTorch + ROCm 7.0.0)
- **Container**: `edaamd/aiac:latest`
- **Work directory**: `/workspace/AIAC/`

**Framework Requirements**:
- **Unsloth**: AMD ROCm-optimized training framework (REQUIRED for competition)
  - Version: 2024.11
  - 2× faster training on AMD MI300X
  - Built-in 4-bit quantization support
  - LoRA/RSLoRA optimizations

**Core Libraries**:
```python
unsloth==2024.11          # ROCm-optimized (REQUIRED)
torch==2.5.1+rocm6.2      # PyTorch with ROCm backend
transformers==4.46.3      # HuggingFace transformers
peft==0.17.1              # LoRA/RSLoRA support
trl==0.12.1               # Trainer utilities
bitsandbytes==0.45.0      # 4-bit quantization
llama-cpp-python          # GGUF inference (for teacher models)
```

**Key Constraint**: All training MUST use Unsloth framework to comply with competition rules

---

## Our Model Architecture

### Base Model

**Qwen2.5-72B-Instruct**
- **Size**: 72 billion parameters
- **Quantization**: 4-bit (bitsandbytes NF4)
- **VRAM footprint**: ~36 GB
- **HuggingFace**: `unsloth/Qwen2.5-72B-Instruct-bnb-4bit`

**Why Qwen2.5-72B?**
- ✅ **Best general knowledge**: 88.60% MMLU (tied #1 open-source)
- ✅ **Broad coverage**: Excels across STEM + Humanities + Social Sciences
- ✅ **Code debugging strength**: 84.2% Aider-Edit (best of all models)
- ✅ **Multilingual robustness**: 29 languages (safety net for edge cases)
- ✅ **Proven MCQ performance**: Designed for instruction following
- ✅ **Fits speed requirements**: 4-6s questions, 2-3s answers

### 4-Adapter Ensemble Architecture

```
┌─────────────────────────────────────────────────┐
│          QWEN2.5-72B BASE MODEL                 │
│         (88.60% MMLU - Tied #1)                 │
│         unsloth/Qwen2.5-72B-bnb-4bit            │
└─────────────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┬──────────────┐
        │             │             │              │
┌───────▼────────┐ ┌──▼──────────┐ ┌▼────────────┐ ┌▼───────────────┐
│ ADAPTER #1     │ │ ADAPTER #2  │ │ ADAPTER #3  │ │ ADAPTER #4     │
│                │ │             │ │             │ │                │
│ MATH           │ │ CODE        │ │ STEM        │ │ HUMANITIES     │
│ DISTILLED      │ │ DISTILLED   │ │ STANDARD    │ │ STANDARD       │
│                │ │             │ │             │ │                │
│ Teacher:       │ │ Teacher:    │ │ No Teacher  │ │ No Teacher     │
│ DeepSeek-R1    │ │ Qwen2.5     │ │ (Direct SFT)│ │ (Direct SFT)   │
│                │ │ Coder-32B   │ │             │ │                │
│                │ │             │ │             │ │                │
│ RSLoRA r=128   │ │ RSLoRA r=128│ │ RSLoRA r=128│ │ RSLoRA r=128   │
│ α=256          │ │ α=256       │ │ α=256       │ │ α=256          │
│ ~15K questions │ │ ~8K ques    │ │ ~13K ques   │ │ ~24K questions │
└────────────────┘ └─────────────┘ └─────────────┘ └────────────────┘
        │                 │               │               │
        └─────────────────┴───────────────┴───────────────┘
                            │
                    ┌───────▼────────┐
                    │  TIES-MERGING  │
                    │  (All r=128)   │
                    │  α/√r scaling  │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │ UNIFIED ADAPTER│
                    │   (~3 GB)      │
                    │                │
                    │ Q-AGENT: Gen   │
                    │ A-AGENT: Ans   │
                    └────────────────┘
```

**Why 4 Adapters?**
- **#1 & #2**: Address Qwen's measurable weaknesses via knowledge distillation
- **#3 & #4**: Cover domains where Qwen already excels (no distillation overhead)
- **TIES-merge**: Combine all 4 into single unified adapter (no ensemble latency)

---

## Teacher Model Selection & Rationale

### Adapter #1: Mathematics Distilled

**Teacher Model**: **DeepSeek-R1** (671B MoE, 37B active)

**Why DeepSeek-R1 for Math?**
- ✅ **Best open-source math model**: 97.3% MATH-500 (beats all others)
- ✅ **Competition math dominance**: 79.8% AIME 2024, 87.5% AIME 2025
- ✅ **Formal logic expertise**: 97.62% accuracy (highest)
- ✅ **Matches closed-source**: On par with OpenAI o1-1217
- ✅ **Verified gap**: Qwen is -11.3 points behind on MATH-500

**Benchmark Comparison**:
| Model | MATH-500 | AIME 2024 | Formal Logic |
|-------|----------|-----------|--------------|
| DeepSeek-R1 | **97.3%** | **79.8%** | **97.62%** |
| Qwen2.5-72B | 86.0% | Lower | Lower |
| **Gap** | **-11.3** | **~-15** | **Significant** |

**Model Access**:
- Format: GGUF (TQ1_0 quantization)
- Size: ~28 GB
- Inference: llama.cpp with ROCm
- Only loaded during teacher label generation, then unloaded

**Training Dataset** (~15K questions):
- Elementary mathematics
- High school mathematics
- College mathematics
- Abstract algebra
- Formal logic
- College physics (calculation-heavy)

---

### Adapter #2: Coding Distilled

**Teacher Model**: **Qwen2.5-Coder-32B-Instruct**

**Why Qwen2.5-Coder-32B for Code?**
- ✅ **Best open-source coding model**: 92.7% HumanEval (matches GPT-4o)
- ✅ **Top LiveCodeBench**: 70.7% (highest among open models)
- ✅ **SOTA code generation**: Beats DeepSeek-V3 (86.6%)
- ✅ **Same architecture**: Qwen family → easier cross-model distillation
- ✅ **Specialized expertise**: Pretrained on 5.5T tokens of code
- ✅ **Verified gap**: Qwen2.5-72B is -13.7 points behind on HumanEval

**Benchmark Comparison**:
| Model | HumanEval | LiveCodeBench | MBPP |
|-------|-----------|---------------|------|
| Qwen2.5-Coder-32B | **92.7%** | **70.7%** | **90.2%** |
| Qwen2.5-72B | 77.3% | 55.5% | 88.2% |
| DeepSeek-V3 | 86.6% | 36% | N/A |
| **Gap (Qwen)** | **-15.4** | **-15.2** | **-2.0** |

**Why NOT DeepSeek-V3?**
- ❌ Qwen2.5-Coder beats it on HumanEval (92.7% vs 86.6%)
- ❌ Qwen2.5-Coder beats it on LiveCodeBench (70.7% vs 36%)
- ✅ Qwen2.5-Coder shares architecture with our base (better distillation)

**Model Access**:
- Format: HuggingFace transformers
- Size: ~32B parameters (4-bit: ~16 GB)
- Inference: Direct PyTorch
- HuggingFace: `Qwen/Qwen2.5-Coder-32B-Instruct`

**Training Dataset** (~8K questions):
- Computer science (algorithms, data structures)
- Programming (code comprehension, theory)
- Software engineering (design patterns)

**Note**: Dataset shows only 2.5% coding questions, but these are high-value MCQs testing algorithmic reasoning (where Qwen2.5-Coder excels)

---

### Adapter #3: STEM Standard

**Teacher Model**: **None** (Direct supervised fine-tuning)

**Why No Teacher for STEM?**
- ✅ Qwen2.5-72B already **tied #1** in MMLU (88.60%)
- ✅ No single model dominates all STEM domains
- ✅ Dataset is 26% STEM (Biology 10.9%, Physics 10%, Chemistry 5.9%)
- ✅ Distillation overhead not worth it for area of strength

**Training Dataset** (~13K questions):
- Biology (largest category at 10.9%)
- Physics (10.0%)
- Chemistry (5.9%)
- Engineering (mechanical, electrical, civil)
- Earth science, astronomy
- Computer science theory (2.5% - absorbed here)

**Approach**: Standard SFT with RSLoRA r=128 on Qwen2.5-72B

---

### Adapter #4: Humanities Standard

**Teacher Model**: **None** (Direct supervised fine-tuning)

**Why No Teacher for Humanities?**
- ✅ Qwen2.5-72B already **tied #1** in MMLU (88.60%)
- ✅ Best open-source for humanities (law, philosophy, history, literature)
- ✅ Only GPT-5 is better (91.4%, but closed-source)
- ✅ Multilingual pre-training gives robust cultural knowledge

**Training Dataset** (~24K questions):
- History (world, US, European)
- Literature (analysis, comprehension)
- Philosophy (ethics, logic, metaphysics)
- Social science (psychology, sociology, economics)
- Geography (physical, cultural)

**Approach**: Standard SFT with RSLoRA r=128 on Qwen2.5-72B

---

## Training Configuration

### RSLoRA (Rank-Stabilized LoRA)

**All 4 adapters use identical LoRA config**:
```python
LoRA Configuration:
  rank (r): 128
  alpha (α): 256
  scaling: α/√r = 256/11.31 = 22.6  # RSLoRA stabilization
  dropout: 0.0
  target_modules: [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
  ]
  bias: "none"
  task_type: "CAUSAL_LM"
```

**Why RSLoRA over standard LoRA?**
- ✅ **α/√r scaling** (not α/r) prevents gradient collapse at high ranks
- ✅ **+6.5% performance gain** over standard LoRA (research-proven)
- ✅ **Stable training** at r=128 (standard LoRA unstable above r=64)
- ✅ **TIES-merge compatibility** (all adapters same rank/technique)

**Trainable Parameters per Adapter**:
- Total params: 74.4B
- Trainable: ~1.68B (2.26%)
- Frozen: 72.72B (97.74%)

### Training Hyperparameters

**Distilled Adapters (#1 Math, #2 Code)**:
```python
batch_size: 4
gradient_accumulation_steps: 4  # Effective batch = 16
learning_rate: 2e-4
lr_scheduler: "cosine"
warmup_ratio: 0.1
epochs: 3
max_grad_norm: 1.0
weight_decay: 0.01
fp16: False
bf16: True  # Better for MI300X

# Distillation-specific
temperature: 2.0  # Soften teacher logits
alpha: 0.3  # Weight hard targets
beta: 0.7  # Weight soft targets (KL divergence)
```

**Standard Adapters (#3 STEM, #4 Humanities)**:
```python
batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2e-4
lr_scheduler: "cosine"
warmup_ratio: 0.1
epochs: 3
max_grad_norm: 1.0
weight_decay: 0.01
bf16: True
```

### TIES-Merging Configuration

**Algorithm**: TIES (Trim, Elect Sign, Disjoint Merge)

**Parameters**:
```python
trim_threshold: 0.2  # Remove bottom 20% by magnitude
merge_method: "ties"
weights: [0.25, 0.25, 0.25, 0.25]  # Equal contribution
```

**Process**:
1. **Trim**: Remove low-magnitude parameters (noise reduction)
2. **Elect Sign**: Majority voting on parameter signs (conflict resolution)
3. **Disjoint Merge**: Average agreeing params, keep non-conflicting
4. **Output**: Single unified adapter (~3 GB)

---

## Novel Contribution Summary

### "Cross-Architecture Knowledge Distillation via LoRA Merging"

**What makes this novel:**

1. **Cross-Architecture Distillation**
   - Traditional: Same family (GPT-4 → GPT-3.5)
   - Ours: Different architectures (DeepSeek-R1 → Qwen, Qwen-Coder → Qwen)
   - Transfers domain expertise while preserving base strengths

2. **Targeted Distillation Only Where Needed**
   - Research-backed: Identified quantified gaps (-11.3 math, -15.4 code)
   - Surgical approach: Only distill weak areas
   - Efficient: No distillation overhead in strong areas (STEM, Humanities)

3. **LoRA Adapter Distillation**
   - Not full model distillation (expensive, slow)
   - Train lightweight adapters (1.68B = 2.26% of total)
   - Fast training, fast inference, memory efficient

4. **RSLoRA for High-Rank Stability**
   - Standard LoRA: α/r (breaks at r>64)
   - RSLoRA: α/√r (stable at r=128)
   - +6.5% gain, enables TIES-merge at high rank

5. **TIES-Merging of Distilled Knowledge**
   - Combine 4 specialist adapters → 1 unified adapter
   - Math + Code + STEM + Humanities in single model
   - No ensemble overhead (3-5GB adapter swap, not 140GB model reload)

**Competitive Advantages**:
- ✅ Grounded in research (not trial-and-error)
- ✅ Surgically addresses weaknesses (not blind overfit)
- ✅ Novel approach (competitors won't expect this)
- ✅ Comprehensive (covers all MCQ domains)
- ✅ Efficient (fits in Unsloth framework and time constraints)

---

## Expected Performance

### Baseline (Qwen2.5-72B without fine-tuning)

| Domain | Baseline | Source |
|--------|----------|--------|
| MMLU (General) | 88.60% | Official benchmark |
| MATH-500 | 86.0% | Official benchmark |
| HumanEval | 77.3% | Official benchmark |
| LiveCodeBench | 55.5% | Official benchmark |

### After Distillation + TIES-Merge (Projected)

| Domain | Adapter | Baseline | Target | Gain |
|--------|---------|----------|--------|------|
| **Mathematics** | #1 Distilled | 86.0% | **94-96%** | +8-10% |
| **Coding** | #2 Distilled | 77.3% | **85-90%** | +8-13% |
| **STEM** | #3 Standard | 88.6% | **90-92%** | +2-4% |
| **Humanities** | #4 Standard | 88.6% | **90-92%** | +2-4% |
| **Overall MCQ** | Unified | ~82% | **86-88%** | +4-6% |

### Tournament Performance (Estimated)

**Q-Agent (Question Generation)**:
- Creativity: 8/10 (balanced, domain-appropriate)
- Difficulty: 7/10 (challenging but fair)
- Correctness: 95%+ (oracle verified)
- Speed: ~4-6s per question (well under 10s limit)

**A-Agent (Answering)**:
- Accuracy: **86-88%** (vs 82% baseline Qwen)
- Reasoning quality: High (CoT-trained from teacher models)
- Speed: ~2-3s per answer (well under 6s limit)

**Win Probability Scenarios**:
- vs Weak opponents (75-80% A-Agent): **90%+ win rate**
- vs Average opponents (80-85% A-Agent): **70-80% win rate**
- vs Strong opponents (85-90% A-Agent): **50-60% win rate**

---

## Summary for External Review

**Our Approach**: Take a strong generalist base model (Qwen2.5-72B, 88.60% MMLU - tied #1 open-source) and make it even stronger by injecting specialist knowledge from the best open models in math (DeepSeek-R1: 97.3% MATH-500) and coding (Qwen2.5-Coder-32B: 92.7% HumanEval).

**How We Do It**: Efficiently through LoRA adapters trained with knowledge distillation, using the Unsloth framework (required by competition) on AMD MI300X GPUs. We combine everything using TIES-Merging, a cutting-edge model merging technique.

**Novel Contribution**: Cross-Architecture Knowledge Distillation via LoRA Merging - surgically transferring domain expertise from specialist models to our generalist base, only in areas where we've identified quantified performance gaps.

**Expected Result**: A Q&A system that generates high-quality questions and answers them with expert-level accuracy (86-88%) across all topics – math, coding, STEM, and humanities – exactly what's needed to win the AMD x Unsloth Synthetic Data Hackathon.

**Technical Compliance**:
- ✅ Uses Unsloth framework (required)
- ✅ Fits on single MI300X (192GB VRAM, 59GB used in inference)
- ✅ Meets speed requirements (<10s questions, <6s answers)
- ✅ ROCm 6.4.0 compatible
- ✅ Outputs correct JSON formats
- ✅ No RAG, no adversarial approaches, English only
