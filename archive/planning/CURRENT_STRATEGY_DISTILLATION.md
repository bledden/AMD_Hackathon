# AMD Hackathon: Knowledge Distillation Strategy

**Competition**: AMD x Unsloth Synthetic Data Hackathon
**Deadline**: Wednesday, October 29, 2025 @ 7 PM PT
**Track**: Synthetic-Data Track (Q&A Agents)

---

## Table of Contents

- [Competition Requirements](#competition-requirements)
- [Current Research Findings](#current-research-findings)
- [Strategic Decision: Base Model](#strategic-decision-base-model)
- [Ensemble Architecture](#ensemble-architecture)
- [Knowledge Distillation Framework](#knowledge-distillation-framework)
- [Adapter Specifications](#adapter-specifications)
- [Novel Contribution](#novel-contribution)
- [Technical Implementation](#technical-implementation)
- [Expected Performance](#expected-performance)
- [Resources](#resources)

---

## Competition Requirements

### Core Format

**Two Agents Per Team:**

1. **Q-Agent (Question Generator)**
   - Takes a topic as input
   - Generates multiple-choice question with 4 choices (A/B/C/D)
   - Provides correct answer and explanation
   - **Speed requirement**: Under 10 seconds per question

2. **A-Agent (Answerer)**
   - Takes a question + 4 choices as input
   - Selects correct answer (A, B, C, or D)
   - Provides reasoning for answer
   - **Speed requirement**: Under 6 seconds per answer

### Required JSON Formats

**Q-Agent Output**:
```json
{
  "topic": "Topic of the Question",
  "question": "Full question text",
  "choices": [
    "A) <choice A text>",
    "B) <choice B text>",
    "C) <choice C text>",
    "D) <choice D text>"
  ],
  "answer": "correct choice letter only (A, B, C, or D)",
  "explanation": "brief explanation within 100 words"
}
```

**A-Agent Output**:
```json
{
  "answer": "correct choice letter only (A, B, C, or D)",
  "reasoning": "brief reasoning within 100 words"
}
```

### Critical Rules

1. **NO RAG** - No Retrieval Augmented Generation
2. **NO adversarial approaches** - No jailbreaking, hallucination attacks
3. **English only** - Both agents must use English
4. **Token limits** - Stay within max_tokens in yaml files
5. **Speed limits** - Strictly enforced (10s questions, 6s answers)
6. **Submission format** - 4 files in `agents/`:
   - `question_agent.py`
   - `question_model.py`
   - `answer_agent.py`
   - `answer_model.py`

### Infrastructure

**Hardware**:
- Platform: DigitalOcean AMD MI300X
- GPU: 1x MI300X - 192GB VRAM
- CPU: 20 vCPU - 240GB RAM
- Storage: 720GB Boot + 5TB Scratch
- Cost: $1.99/hr

**Software**:
- Image: ROCm 6.4.0
- Container: `edaamd/aiac:latest`
- Framework: Unsloth (ROCm-optimized)
- Libraries: PyTorch, Transformers, TRL, PEFT

---

## Current Research Findings

### Benchmark Comparison

| Benchmark | **Qwen2.5-72B** | **DeepSeek-V3** | **DeepSeek-R1** | Analysis |
|-----------|-----------------|-----------------|-----------------|----------|
| **MMLU** (General Knowledge) | **88.60%** | 88.5% | N/A | Qwen tied for 1st |
| **MATH-500** (Mathematics) | 86.0% | 89.0% | **97.3%** | **-11.3 gap for Qwen** |
| **AIME 2024** (Competition Math) | Lower | N/A | **79.8%** | **-15.3 gap for Qwen** |
| **AIME 2025** | N/A | N/A | **87.5%** | DeepSeek-R1 dominates |
| **Formal Logic** | Lower | N/A | **97.62%** | DeepSeek-R1 strongest |
| **LiveCodeBench** (Coding) | 28% | **36%** | N/A | **-8 gap for Qwen** |
| **HumanEval-Mul** (Code Tests) | 77.3% | **82.6%** | N/A | DeepSeek-V3 leads |
| **Aider-Edit** (Code Debug) | **84.2%** | Lower | N/A | **Qwen strongest** |
| **Languages** | **29** | 3 | 3 | Qwen multilingual |

### Key Insights

**Qwen2.5-72B Strengths**:
- General knowledge (MMLU 88.60% - tied #1)
- Code editing/debugging (Aider-Edit 84.2%)
- Multilingual robustness (29 languages)
- Broad domain coverage

**Qwen2.5-72B Weaknesses** (quantified gaps):
1. **Mathematics**: -11.3 points on MATH-500 vs DeepSeek-R1
2. **Advanced Coding**: -8 points on LiveCodeBench vs DeepSeek-V3
3. **Formal Logic**: Significantly lower than DeepSeek-R1's 97.62%

**Teacher Models Identified**:
- **DeepSeek-R1**: Mathematics champion (97.3% MATH-500, 87.5% AIME 2025)
- **DeepSeek-V3**: Coding champion (36% LiveCodeBench, 82.6% HumanEval)

---

## Strategic Decision: Base Model

### Decision: KEEP Qwen2.5-72B as Base Model

**Rationale**:

1. **Tournament is MCQ-focused, not pure math**
   - MMLU performance (88.60% - tied 1st) is most relevant
   - General knowledge matters more than specialized math ability
   - Qwen's breadth covers all potential question domains

2. **Qwen's breadth > DeepSeek's depth**
   - Need solid performance across STEM + Humanities + Social Sciences
   - DeepSeek-R1 dominates math but lacks general knowledge data
   - DeepSeek-V3 excels at coding but lower on humanities

3. **Existing investment**
   - STEM specialist training already 1/7 complete with Qwen
   - Switching base models would lose this progress
   - Time constraint (46 hours remaining) makes switching risky

4. **Multilingual robustness**
   - 29-language support provides safety net for diverse questions
   - More robust embeddings from multilingual pre-training
   - Better generalization to edge cases

5. **Clear path to improvement**
   - Measurable gaps (-11.3 math, -8 coding) are addressable
   - Knowledge distillation can transfer DeepSeek expertise
   - TIES-merging enables combining specialist adapters

**Conclusion**: Qwen is the optimal base for a tournament requiring breadth. We will surgically address its weaknesses via distillation from domain experts.

---

## Ensemble Architecture

### Architecture Diagram

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
│ Math Distilled │ │ Code Distill│ │ STEM Std    │ │ Humanities Std │
│                │ │             │ │             │ │                │
│ Teacher:       │ │ Teacher:    │ │ No Teacher  │ │ No Teacher     │
│ DeepSeek-R1    │ │ DeepSeek-V3 │ │ (Direct SFT)│ │ (Direct SFT)   │
│ (97.3% MATH)   │ │ (36% LCB)   │ │             │ │                │
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
                    │                │
                    │ Q-AGENT: Gen   │
                    │ A-AGENT: Ans   │
                    └────────────────┘
```

### Why This Architecture?

1. **Unified base model** (Qwen2.5-72B)
   - Load once at tournament start
   - Fast adapter swapping (3-5GB vs 140GB model reload)
   - All adapters are compatible (same base, same rank, same technique)

2. **RSLoRA for all adapters** (rank r=128)
   - Rank-Stabilized LoRA uses α/√r scaling (not α/r)
   - Enables high-rank training without gradient collapse
   - Research shows +6.5% gain over standard LoRA
   - All adapters use r=128 for perfect TIES-merge compatibility

3. **Strategic adapter split**:
   - **Distilled adapters (#1, #2)**: Address Qwen's measurable weaknesses
   - **Standard adapters (#3, #4)**: Cover areas where Qwen is already strong

4. **TIES-Merging for unified model**:
   - Trim: Remove low-magnitude parameters (noise reduction)
   - Elect Sign: Majority voting on parameter signs (handle conflicts)
   - Disjoint Merge: Average agreeing params, keep non-conflicting (preserve all knowledge)
   - Result: Single adapter with combined expertise

---

## Knowledge Distillation Framework

### What is Knowledge Distillation?

**Definition**: Training a student model to mimic a teacher model's outputs, including:
- Hard targets (final predictions: A, B, C, or D)
- Soft targets (logits/probabilities: confidence distributions)
- Reasoning patterns (chain-of-thought explanations)

**Why it works**:
- Teacher's confidence distribution contains more information than just the answer
- Example: Teacher might output [0.6, 0.3, 0.08, 0.02] showing "A is most likely, B is plausible"
- Student learns the teacher's reasoning process, not just memorization

### Our Cross-Architecture Distillation

**Novel approach**: Training Qwen adapters to learn from DeepSeek models

**Process**:
1. Load teacher model (DeepSeek-R1 or DeepSeek-V3)
2. Generate teacher outputs for filtered dataset:
   - Chain-of-thought reasoning
   - Answer confidence (logits)
   - Explanation text
3. Train Qwen adapter to replicate teacher's outputs:
   - Loss = α * CrossEntropy(hard) + (1-α) * KL-Divergence(soft)
   - α = 0.3 (weight soft targets higher for knowledge transfer)
4. Save Qwen adapter (teacher model no longer needed)

**Why this is novel**:
- Most distillation: Same architecture (e.g., GPT-4 → GPT-3.5)
- Our approach: Cross-architecture (DeepSeek → Qwen)
- Transfers domain expertise while keeping Qwen's strengths
- Enables TIES-merging of specialized knowledge

---

## Adapter Specifications

### Adapter #1: Math Distilled

**Purpose**: Transfer DeepSeek-R1's mathematics expertise to Qwen

**Configuration**:
- Base: Qwen2.5-72B
- Teacher: DeepSeek-R1 (97.3% MATH-500)
- LoRA: RSLoRA r=128, α=256 (α/√r = 256/11.31 = 22.6)
- Trainable params: ~1.68B (2.26% of 74.4B total)

**Dataset** (~15K questions):
- `elementary_mathematics` (MCQ questions)
- `high_school_mathematics`
- `college_mathematics`
- `abstract_algebra`
- `formal_logic`
- `college_physics` (calculation-heavy only)

**Training process**:
1. Load DeepSeek-R1 via llama.cpp (GGUF format for memory efficiency)
2. For each math question:
   - Generate DeepSeek-R1 chain-of-thought reasoning
   - Extract logits for A/B/C/D
   - Save as augmented training example
3. Train Qwen adapter with distillation loss:
   - Hard target: Correct answer
   - Soft target: DeepSeek's logits
   - CoT target: Reasoning text
4. Save adapter: `math_distilled_rslora_r128/`

**Target improvement**: +8-10% on MATH-500 (86% → 94-96%)

---

### Adapter #2: Code Distilled

**Purpose**: Transfer DeepSeek-V3's coding expertise to Qwen

**Configuration**:
- Base: Qwen2.5-72B
- Teacher: DeepSeek-V3 (36% LiveCodeBench)
- LoRA: RSLoRA r=128, α=256

**Dataset** (~8K questions):
- `computer_science` (algorithms, data structures)
- `programming` (code comprehension)
- `software_engineering` (design patterns, architecture)

**Training process**: Same as Adapter #1, using DeepSeek-V3 as teacher

**Target improvement**: +4-6% on LiveCodeBench (28% → 32-34%)

---

### Adapter #3: STEM Standard

**Purpose**: Cover general STEM topics where Qwen is already competitive

**Configuration**:
- Base: Qwen2.5-72B
- Teacher: None (direct supervised fine-tuning)
- LoRA: RSLoRA r=128, α=256

**Dataset** (~13K questions):
- `physics` (conceptual, not calculation-heavy)
- `chemistry` (organic, inorganic, general)
- `biology` (cell biology, genetics, evolution)
- `engineering` (mechanical, electrical, civil)
- `earth_science` (geology, meteorology)

**Training process**: Standard SFT with MCQ dataset

**Why no distillation**: Qwen already strong in general STEM (MMLU 88.60%)

---

### Adapter #4: Humanities Standard

**Purpose**: Cover humanities and social sciences

**Configuration**:
- Base: Qwen2.5-72B
- Teacher: None (direct supervised fine-tuning)
- LoRA: RSLoRA r=128, α=256

**Dataset** (~24K questions):
- `history` (world, US, European)
- `literature` (analysis, comprehension)
- `philosophy` (ethics, logic, metaphysics)
- `social_science` (psychology, sociology, economics)
- `geography` (physical, cultural)

**Training process**: Standard SFT with MCQ dataset

**Why no distillation**: Qwen tied #1 in MMLU (88.60%), already dominates humanities

---

## Novel Contribution

### "Cross-Architecture Knowledge Distillation via LoRA Merging"

**What makes this novel:**

1. **Cross-architecture distillation**
   - Traditional: Same model family (GPT-4 → GPT-3.5)
   - Ours: Different architectures (DeepSeek → Qwen)
   - Transfers domain expertise while preserving base model strengths

2. **Targeted distillation only where needed**
   - Research-backed: Only distill in areas with quantified gaps
   - Math: -11.3 points → distill from DeepSeek-R1
   - Code: -8 points → distill from DeepSeek-V3
   - General: Already strong → no distillation overhead

3. **LoRA adapter distillation**
   - Not full model distillation (expensive, slow)
   - Train lightweight adapters (1.68B params = 2.26% of total)
   - Fast training, fast inference, memory efficient

4. **TIES-Merging of distilled knowledge**
   - Combine 4 specialist adapters into unified adapter
   - Math + Code + STEM + Humanities in single model
   - No ensemble overhead at inference time

5. **RSLoRA for high-rank stability**
   - Standard LoRA: α/r scaling (breaks at high ranks)
   - RSLoRA: α/√r scaling (stable at r=128)
   - +6.5% performance gain over standard LoRA

**Competitive advantages**:
- Grounded in research (not just "try everything")
- Surgically addresses weaknesses (not blind overfit)
- Novel approach (competitors won't expect this)
- Comprehensive (covers all MCQ domains)
- Efficient (fits in time/compute constraints)

---

## Technical Implementation

### Training Pipeline

**Phase 1: Adapter #3 (STEM Standard)** - ✅ IN PROGRESS
- Status: Chunk 1/7 complete
- Loss: 0.62 → 0.56 (good convergence)
- Time: ~2 hours remaining
- Output: `stem_standard_rslora_r128/`

**Phase 2: Generate Math Teacher Labels** (~3 hours)
1. Load DeepSeek-R1 via llama.cpp
2. Process 15K math questions
3. Generate CoT reasoning + logits
4. Save augmented dataset: `data/math_distilled_dataset.json`

**Phase 3: Train Adapter #1 (Math Distilled)** (~4 hours)
1. Load Qwen2.5-72B + RSLoRA r=128
2. Train with distillation loss:
   ```python
   loss = 0.3 * F.cross_entropy(logits, hard_labels) + \
          0.7 * F.kl_div(F.log_softmax(student_logits / T, dim=-1),
                         F.softmax(teacher_logits / T, dim=-1))
   # T = temperature (2.0 for smoothing)
   ```
3. Output: `math_distilled_rslora_r128/`

**Phase 4: Generate Code Teacher Labels** (~2 hours)
1. Load DeepSeek-V3 via llama.cpp
2. Process 8K coding questions
3. Save augmented dataset: `data/code_distilled_dataset.json`

**Phase 5: Train Adapter #2 (Code Distilled)** (~3 hours)
1. Same process as Phase 3
2. Output: `code_distilled_rslora_r128/`

**Phase 6: Train Adapter #4 (Humanities Standard)** (~4 hours)
1. Standard SFT on 24K humanities questions
2. Output: `humanities_standard_rslora_r128/`

**Phase 7: TIES-Merge All Adapters** (~1 hour)
1. Load all 4 adapters
2. Apply TIES algorithm:
   - Trim threshold: 0.2 (remove bottom 20% by magnitude)
   - Sign election: Majority voting
   - Disjoint merge: Average agreeing, keep non-conflicting
3. Output: `unified_adapter_rslora_r128/`

**Phase 8: Build Tournament Agents** (~4 hours)
1. Create `agents/question_agent.py`:
   - Load Qwen2.5-72B + unified adapter
   - Temperature: 0.6 (moderate creativity)
   - Max tokens: 200
2. Create `agents/answer_agent.py`:
   - Load Qwen2.5-72B + unified adapter
   - Temperature: 0.2 (high precision)
   - Max tokens: 150
3. Test invocation commands
4. Verify JSON output formats
5. Verify speed (<10s question, <6s answer)

**Phase 9: Validation** (~6 hours)
1. Generate 100 questions, verify quality
2. Answer 500 test questions, measure accuracy
3. Run mock tournament
4. Fine-tune generation parameters
5. Final speed tests

---

### Hardware Utilization

**Model Loading**:
- Qwen2.5-72B (4-bit): ~36GB VRAM
- RSLoRA adapter: ~3GB VRAM
- Inference buffer: ~20GB
- Total: ~59GB / 192GB (31% utilization)

**Training**:
- Batch size: 4 (fit easily)
- Gradient checkpointing: Enabled
- Peak VRAM: ~80GB / 192GB (42% utilization)

**Teacher Models** (temporary, GGUF format):
- DeepSeek-R1 GGUF: ~28GB (TQ1_0 quantization)
- DeepSeek-V3 GGUF: ~85GB (TQ1_0 quantization)
- Only loaded during label generation, then unloaded

---

### Software Stack

**Core Libraries**:
```python
unsloth==2024.11  # ROCm-optimized training
torch==2.5.1+rocm6.2
transformers==4.46.3
peft==0.17.1  # LoRA/RSLoRA support
trl==0.12.1  # Trainer utilities
bitsandbytes==0.45.0  # 4-bit quantization
```

**Distillation Tools**:
```python
llama-cpp-python  # GGUF model inference
scipy  # KL-divergence calculation
```

**Merging Tools**:
```python
# Custom TIES implementation
from peft import PeftModel, get_peft_model
import torch.nn.functional as F
```

---

## Expected Performance

### Baseline (Qwen2.5-72B without fine-tuning)

| Metric | Score | Source |
|--------|-------|--------|
| MMLU | 88.60% | Official benchmark |
| MATH-500 | 86.0% | Official benchmark |
| LiveCodeBench | 28% | Official benchmark |

### After Distillation + TIES-Merge (Projected)

| Domain | Adapter | Baseline | Target | Gain |
|--------|---------|----------|--------|------|
| **Mathematics** | #1 Distilled | 86.0% | **94-96%** | +8-10% |
| **Coding** | #2 Distilled | 28% | **32-34%** | +4-6% |
| **STEM** | #3 Standard | 88% | **90-92%** | +2-4% |
| **Humanities** | #4 Standard | 88.60% | **90-92%** | +2-4% |
| **Overall MCQ** | Unified | ~82% | **86-88%** | +4-6% |

### Tournament Performance (Estimated)

**Q-Agent (Question Generation)**:
- Creativity: 8/10 (balanced, domain-appropriate)
- Difficulty: 7/10 (challenging but fair)
- Correctness: 95%+ (oracle verified)
- Speed: ~4-6s per question (well under 10s limit)

**A-Agent (Answer)**:
- Accuracy: **86-88%** (vs 82% baseline)
- Reasoning quality: High (CoT training)
- Speed: ~2-3s per answer (well under 6s limit)

**Win Conditions**:
- Strong at both Q&A (balanced agents)
- High accuracy on opponent's questions (distilled reasoning)
- Generate fair but challenging questions (not too hard)
- Consistent performance across all domains (TIES-merged expertise)

---

## Resources

### Official Documentation

- **Hackathon**: https://docs.unsloth.ai/new/unsloth-amd-pytorch-synthetic-data-hackathon
- **AMD Cloud**: https://devcloud.amd.com/
- **Registration**: https://forms.gle/RPV7fURLNHDjz2yr9

### Technical References

- **RSLoRA Paper**: "Rank-Stabilized LoRA: Robust Adapter Training" (2024)
- **TIES-Merging**: "TIES-Merging: Resolving Interference in Task Arithmetic" (NeurIPS 2023)
- **Knowledge Distillation**: "Distilling the Knowledge in a Neural Network" (Hinton et al.)
- **Unsloth Docs**: https://github.com/unslothai/unsloth
- **Meta Synthetic Data Kit**: https://github.com/meta-llama/synthetic-data-kit/

### Model Cards

- **Qwen2.5-72B**: https://huggingface.co/Qwen/Qwen2.5-72B-Instruct
- **DeepSeek-R1**: https://huggingface.co/deepseek-ai/DeepSeek-R1
- **DeepSeek-V3**: https://huggingface.co/deepseek-ai/DeepSeek-V3

### Our Implementation Files

**Training Scripts**:
- [scripts/train_stem_specialist_rslora.py](scripts/train_stem_specialist_rslora.py) - Adapter #3 (IN PROGRESS)
- [scripts/train_math_distilled_rslora.py](scripts/train_math_distilled_rslora.py) - Adapter #1 (TODO)
- [scripts/train_code_distilled_rslora.py](scripts/train_code_distilled_rslora.py) - Adapter #2 (TODO)
- [scripts/train_humanities_rslora.py](scripts/train_humanities_rslora.py) - Adapter #4 (TODO)

**Distillation Scripts**:
- [scripts/generate_math_teacher_labels.py](scripts/generate_math_teacher_labels.py) - DeepSeek-R1 CoT
- [scripts/generate_code_teacher_labels.py](scripts/generate_code_teacher_labels.py) - DeepSeek-V3 CoT

**Merging Script**:
- [scripts/merge_specialists_ties.py](scripts/merge_specialists_ties.py) - TIES implementation

**Agent Scripts** (TODO):
- `AIAC/agents/question_agent.py` - Q-Agent entry point
- `AIAC/agents/question_model.py` - Q-Agent model loader
- `AIAC/agents/answer_agent.py` - A-Agent entry point
- `AIAC/agents/answer_model.py` - A-Agent model loader

---

## Summary

**Strategy**: Research-driven knowledge distillation from domain experts to address Qwen's specific weaknesses

**Base Model**: Qwen2.5-72B (88.60% MMLU, tied #1 for general knowledge)

**Ensemble**: 4 RSLoRA adapters (r=128) → TIES-merged into unified adapter
1. Math Distilled (DeepSeek-R1 teacher)
2. Code Distilled (DeepSeek-V3 teacher)
3. STEM Standard (direct SFT)
4. Humanities Standard (direct SFT)

**Novel Contribution**: Cross-Architecture Knowledge Distillation via LoRA Merging

**Target Performance**: 86-88% MCQ accuracy (vs 82% baseline Qwen)

**Status**: Adapter #3 training (1/7 chunks complete), remaining phases planned

**Competitive Edge**: Grounded in research, surgically addresses weaknesses, novel approach competitors won't expect
