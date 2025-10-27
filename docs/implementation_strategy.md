# AMD Hackathon Implementation Strategy
## Chain-of-Thought + Curriculum Learning + Distillation

**Last Updated**: 2025-10-26
**Deadline**: Wednesday, Oct 29, 2025 @ 7:00 PM PT (~36 hours)

---

## 🎯 Goal
Create Q&A agents that achieve **94-96% accuracy** through:
1. **Chain-of-Thought Training** (+3-5%)
2. **Curriculum Learning** (+1-2%)
3. **Comprehensive Dataset** (+2-3%)

---

## 📋 Pipeline Overview

### Phase 1: Chain-of-Thought Generation (4-6 hours)
**Status**: In Progress ⏳

**Teacher Model**: DeepSeek-V3.1 (1-bit GGUF quantization)
- **Why**: 90% MMLU, best reasoning quality
- **Format**: GGUF 1-bit (IQ1_M) via llama.cpp
- **VRAM**: ~50-60GB (fits easily in 192GB)
- **Speed**: 4-6 hours for 50K questions
- **ROCm**: Native support via llama.cpp

**Process**:
```python
# For each question in 50K dataset:
1. Load question + choices
2. Generate step-by-step reasoning (DeepSeek-V3.1)
3. Add 'chain_of_thought' field to question
4. Save enhanced dataset
```

**Output**: `/data/enhanced/cot_enhanced_50k.json`

**Scripts**:
- [download_deepseek_gguf.py](../scripts/download_deepseek_gguf.py) - Setup GGUF model
- [generate_cot_llama_cpp.py](../scripts/generate_cot_llama_cpp.py) - Generate CoT

---

### Phase 2: Curriculum Ordering (1-2 hours)
**Status**: Pending ⏸️

**Assessor Model**: Phi-4 (14B parameters)
- **Why**: 84% MMLU, efficient difficulty scoring
- **VRAM**: ~28GB
- **Speed**: 1-2 hours for 50K questions
- **Architecture**: Different from Qwen (diverse perspective)

**Process**:
```python
# For each question with CoT:
1. Load question + choices + CoT
2. Assess difficulty score 1-10 (Phi-4)
3. Sort questions by difficulty
4. Add curriculum metadata
```

**Output**: `/data/enhanced/curriculum_ordered_50k.json`

**Curriculum Structure**:
- Easy (1-4): First 30% of training
- Medium (4-7): Middle 50% of training
- Hard (7-10): Final 20% of training

**Scripts**:
- [create_curriculum_phi4.py](../scripts/create_curriculum_phi4.py)

---

### Phase 3: Fine-Tuning Qwen3-235B (12 hours)
**Status**: Pending ⏸️

**Student Model**: Qwen3-235B-Instruct
- **Why**: State-of-the-art Q&A performance
- **VRAM**: ~180GB with quantization
- **Final Performance**: 94-96% accuracy target

**Process**:
```python
# Training configuration:
1. Load curriculum-ordered dataset (easy → hard)
2. Use Unsloth for 2x speedup + 70% less VRAM
3. Fine-tune with LoRA adapters
4. Progressive learning: adjust LR as difficulty increases
```

**Training Hyperparameters**:
- **Epochs**: 3
- **Batch Size**: 2-4 (gradient accumulation)
- **Learning Rate**: 2e-5 → 1e-5 (curriculum decay)
- **Quantization**: 4-bit (QLo RA)
- **LoRA Rank**: 64
- **Context Length**: 2048 tokens

**Output**: `/models/qwen3_235b_cot_curriculum/`

**Scripts**:
- TBD: `train_qwen3_235b.py`

---

### Phase 4: Testing & Deployment (2 hours)
**Status**: Pending ⏸️

**Benchmarking**:
1. Accuracy on held-out test set (10% of 50K = 5K questions)
2. Latency testing (target: <60ms per question)
3. VRAM usage monitoring

**Deployment**:
1. Export fine-tuned model
2. Create Q-Agent (question answering)
3. Create A-Agent (answer validation)
4. Test tournament interface

**Scripts**:
- TBD: `test_model.py`
- TBD: `deploy_agents.py`

---

## 🔧 Technical Implementation

### Why GGUF for CoT Generation?

**GGUF/llama.cpp** vs **BitsAndBytes**:

| Feature | GGUF/llama.cpp | BitsAndBytes |
|---------|---------------|--------------|
| ROCm Support | ✅ Native | ⚠️ Limited |
| True 1-bit | ✅ Yes (IQ1_M) | ❌ No |
| VRAM Usage | 50-60GB | 80-100GB |
| Speed | Fast (C++) | Slower (Python) |
| Use Case | **Inference** | Training + Inference |

**Decision**: Use GGUF for CoT generation (inference only), then switch to PyTorch for fine-tuning.

---

### Why This Strategy Beats Competitors?

**Baseline Approach** (Most competitors):
```
Download MMLU → Fine-tune Qwen3 → 90% accuracy
```

**Our Approach**:
```
50K diverse questions → CoT generation (DeepSeek-V3.1) →
Curriculum ordering (Phi-4) → Fine-tune Qwen3-235B → 94-96% accuracy
```

**Competitive Advantages**:
1. **CoT**: Models learn reasoning process, not just answers
2. **Curriculum**: Progressive learning improves retention
3. **Scale**: 50K vs typical 10K questions
4. **Diversity**: Multiple sources (MMLU, TriviaQA, CommonsenseQA, LogiQA)

**Expected Edge**: +4-6% accuracy over baseline

---

## 📊 Current Status

### ✅ Completed
- [x] Dataset collection (50,000 questions)
- [x] Topic coverage analysis (95%+ coverage, 42 domains)
- [x] Answer format verification (100% A/B/C/D format)
- [x] Infrastructure setup (MI300X, ROCm, PyTorch)
- [x] Strategy designed (CoT + Curriculum + Comprehensive data)

### ⏳ In Progress
- [ ] Installing llama-cpp-python with ROCm support (~10 min)
- [ ] Downloading DeepSeek-V3.1 GGUF (~30 min for 50GB)

### ⏸️ Pending
- [ ] Generate CoT for 50K questions (4-6 hours)
- [ ] Create curriculum ordering (1-2 hours)
- [ ] Fine-tune Qwen3-235B (12 hours)
- [ ] Test and deploy agents (2 hours)

**Total Remaining**: ~20 hours
**Time Available**: ~36 hours
**Buffer**: 16 hours ✅

---

## 🚨 Risk Mitigation

### Risk 1: Model Download Fails
**Mitigation**: Multiple GGUF sources (bartowski, unsloth repos)
**Fallback**: Use Q2_K or Q4_K quantization if IQ1_M unavailable

### Risk 2: llama.cpp ROCm Issues
**Mitigation**: Pre-built binaries available, compiled with HIPBLAS
**Fallback**: Use CPU inference (slower but works)

### Risk 3: Fine-Tuning Timeout
**Mitigation**: 16-hour buffer, can reduce dataset to 30K if needed
**Fallback**: Early stopping at 2 epochs instead of 3

### Risk 4: Phi-4 Unavailable
**Mitigation**: Heuristic difficulty scoring as fallback
**Fallback**: Sort by question/choice length + complexity keywords

---

## 📝 Next Steps

1. **Wait for llama-cpp-python installation** (5-10 min)
2. **Monitor DeepSeek GGUF download** (~30 min)
3. **Start CoT generation** (4-6 hours)
4. **While CoT runs**: Prepare Phi-4 curriculum script
5. **While curriculum runs**: Prepare Qwen3-235B training script
6. **Sequential execution**: CoT → Curriculum → Training → Testing

**Estimated Completion**: Tuesday, Oct 28, 2025 @ 11:00 PM PT
**Deadline**: Wednesday, Oct 29, 2025 @ 7:00 PM PT
**Safety Margin**: 20 hours ✅

---

## 📚 References

- [DeepSeek-V3 Paper](https://arxiv.org/abs/2401.14196)
- [Qwen3 Technical Report](https://qwenlm.github.io/blog/qwen3/)
- [Phi-4 Release](https://huggingface.co/microsoft/phi-4)
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [Curriculum Learning](https://arxiv.org/abs/2009.11032)
- [llama.cpp ROCm Support](https://github.com/ggerganov/llama.cpp/blob/master/docs/backend/ROCm.md)
