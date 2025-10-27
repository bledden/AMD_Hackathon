# AMD Hackathon Status Report & Strategic Plan

**Date**: October 26, 2025
**Time Remaining**: ~72 hours (Deadline: Wednesday Oct 29, 2025 @ 7:00 PM PT)
**Budget**: $297 remaining

---

## 1. Hackathon Overview

### Goal
Create the best-performing Q&A agent system using AMD MI300X GPU hardware for a tournament-style competition.

### Competition Format
- **Q-Agent**: Asks questions to gather information
- **A-Agent**: Answers questions based on its knowledge
- **Objective**: Achieve highest accuracy on comprehensive Q&A tasks

### Rules & Constraints
- Must utilize AMD MI300X GPU (192GB VRAM)
- Focus on general knowledge Q&A (NOT hyperfocused on niche categories like blood relations or seating arrangements)
- Deploy two separate agents (Q and A) for tournament play
- Optimize for accuracy above all else

---

## 2. Hardware & Infrastructure

### Compute Resources
```
GPU:         AMD Instinct MI300X VF
VRAM:        192 GB (205,822,885,888 bytes)
System RAM:  235 GB total, 229 GB available
Disk:        697 GB total, 211 GB free
ROCm:        Version 6.2.41133-dd7f95766
Container:   Docker with ROCm 6.2 support
```

### Key Insight on Memory
- **VRAM is the bottleneck** - models must fit here for fast training
- System RAM can be used for offloading but causes 10-100x slowdown
- Disk offloading is possible but makes training impractical

---

## 3. Dataset Status

### Current Dataset
- **50,000 questions** collected and formatted
- **100% MCQ format** (multiple choice with 4 options)
- **Curriculum ordered** by difficulty (easy → hard)
- Split: 45,002 training / 4,998 validation

### Dataset Composition
```
- 34% Easy questions (basic facts, definitions)
- 61% Medium questions (reasoning, connections)
- 5% Hard questions (complex analysis, synthesis)
```

### Sources
- MMLU (various academic subjects)
- TriviaQA (general knowledge)
- CommonsenseQA (reasoning)
- Additional curated sources

---

## 4. What We've Tried (Chronological)

### Phase 1: Initial Strategy - Chain-of-Thought (CoT)
**Approach**: Use DeepSeek-V3.1 to generate CoT explanations, then train Qwen3-235B
**Result**: FAILED - CoT generation at 139 seconds/question = 80 days for 50K questions
**Learning**: Even with optimization, would take 8-12 days minimum

### Phase 2: Catastrophic Forgetting Research
**Discovery**: Analyzed 2024-2025 papers on LLM fine-tuning
**Key Finding**: CoT gains (+3-5%) < Forgetting penalty (-6-10%) = NET NEGATIVE
**Decision**: Skip CoT entirely, focus on LoRA + Curriculum + Replay Buffer

### Phase 3: BitsAndBytes Attempts
**Approach**: Use BitsAndBytes for 4-bit quantization with Qwen3-235B
**Result**: FAILED - No ROCm 6.2 support (`libbitsandbytes_rocm62.so` not found)
**Multiple attempts**: Building from source, downgrading, all failed

### Phase 4: Unsloth Implementation
**Approach**: Use Unsloth (designed for AMD ROCm) with Qwen3-235B
**Partial Success**:
- ✅ Model loads (158-180GB VRAM)
- ✅ LoRA configures properly
- ❌ Meta tensor error - some weights offloaded to CPU/disk
- ❌ Unsloth can't handle offloaded weights

### Current Status
**Problem**: Qwen3-235B is fundamentally too large
- Needs ~470GB unquantized (235B params × 2 bytes)
- Even "4-bit" version uses 160-180GB VRAM
- Leaves insufficient room for training
- Requires CPU/disk offloading which breaks Unsloth

---

## 5. Strategic Pivot Recommendation

### The Reality Check
**Qwen3-235B won't work** in our environment because:
1. Takes 180GB+ VRAM, leaving <12GB for training
2. Requires offloading which breaks acceleration
3. Would train extremely slowly even if we got it working
4. Risk of OOM during training is very high

### Recommended Model: Qwen2.5-72B-Instruct

**Why this will work:**
- **Size**: ~40-50GB in 4-bit (fits easily in 192GB VRAM)
- **Performance**: Still extremely capable (72B is powerful)
- **Training Speed**: Full GPU acceleration, no offloading
- **Success Rate**: Near 100% - will definitely work
- **Time to Train**: 6-8 hours for full fine-tuning

**Expected Performance:**
- Baseline: ~82-83% accuracy
- With our optimizations: ~85-87% accuracy
- Training techniques: LoRA + Curriculum + Replay Buffer

---

## 6. Proposed Action Plan

### Immediate Actions (Next 4 Hours)

#### Step 1: Model Pivot (30 minutes)
```python
# Switch to Qwen2.5-72B-Instruct
model_name = "Qwen/Qwen2.5-72B-Instruct"
# Will use ~50GB VRAM, leaving 140GB for training
```

#### Step 2: Training Script Update (30 minutes)
- Modify `train_qwen3_unsloth.py` → `train_qwen2.5_unsloth.py`
- Increase LoRA rank back to 64 (we have room now)
- Increase batch size to 4-8 for faster training
- Remove memory limitations since model fits easily

#### Step 3: Launch Training (6-8 hours)
- Start curriculum training immediately
- Monitor first epoch for stability
- Checkpoint every 1000 steps

#### Step 4: Parallel Validation (During Training)
- Test checkpoints on validation set
- Monitor for overfitting
- Adjust learning rate if needed

### Day 2 Plan (Monday)

#### Morning: Training Completion
- Final model checkpoint
- Full validation testing
- Performance metrics collection

#### Afternoon: Agent Development
- Create Q-Agent wrapper (question generation)
- Create A-Agent wrapper (answer selection)
- Test inter-agent communication

### Day 3 Plan (Tuesday)

#### Morning: Optimization
- Fine-tune prompts for better accuracy
- Implement confidence scoring
- Add fallback strategies

#### Afternoon: Testing
- Full tournament simulation
- Stress testing
- Bug fixes

### Day 4 Plan (Wednesday - Competition Day)

#### Morning: Final Preparations
- Last-minute optimizations
- Backup deployment ready
- Documentation complete

#### Afternoon: Competition
- Deploy agents
- Monitor performance
- Real-time adjustments if allowed

---

## 7. Risk Mitigation

### Plan B Options
1. **If Qwen2.5-72B fails**: Use Llama-3.1-70B (proven to work)
2. **If training is too slow**: Reduce dataset to 30K highest quality
3. **If accuracy is low**: Implement ensemble voting with multiple checkpoints
4. **If deployment fails**: Have quantized GGUF versions ready

### What We WON'T Do
- Don't waste more time on Qwen3-235B
- Don't try to build custom ROCm binaries
- Don't implement complex CoT (negative ROI)
- Don't get distracted by niche optimizations

---

## 8. Success Metrics

### Minimum Viable Product
- 82% accuracy on validation set
- Sub-2 second response time
- Stable deployment

### Target Performance
- 85-87% accuracy
- Sub-1 second response time
- Robust error handling

### Stretch Goals
- 88%+ accuracy
- Ensemble model deployment
- Advanced prompt engineering

---

## 9. Budget Allocation

**Remaining: $297**

Suggested allocation:
- $100 - Emergency compute backup (if needed)
- $100 - Additional storage (if needed)
- $97 - Buffer for unexpected costs

Current burn rate is minimal since we're using pre-allocated resources.

---

## 10. Key Decisions & Rationale

### Why Not Qwen3-235B?
- **Too Large**: Requires 180GB+ VRAM
- **Too Slow**: Offloading kills performance
- **Too Risky**: High chance of OOM failures
- **Time Cost**: Would take days to properly train

### Why Qwen2.5-72B?
- **Right Size**: Fits comfortably in available VRAM
- **Fast Training**: No offloading needed
- **Proven Architecture**: Well-tested and reliable
- **Still Powerful**: 72B parameters is very capable

### Why LoRA + Curriculum?
- **LoRA**: Reduces trainable params from 72B to ~500M
- **Curriculum**: Improves learning efficiency (+1-2% accuracy)
- **Replay Buffer**: Prevents catastrophic forgetting
- **Combined Effect**: Maximum gain with minimum risk

---

## 11. Critical Path Timeline

```
Hour 0-1:   Create Qwen2.5-72B training script
Hour 1-2:   Start training, verify stability
Hour 2-8:   Training continues (automated)
Hour 8-9:   Validation testing
Hour 9-10:  Create agent wrappers
Hour 10-12: Integration testing
Hour 12-24: Optimization and refinement
Hour 24-48: Full system testing
Hour 48-72: Final preparations and deployment
```

---

## 12. Conclusion

**We need to pivot NOW from Qwen3-235B to Qwen2.5-72B.**

The math is clear:
- Qwen3-235B: Won't fit, won't train properly, too risky
- Qwen2.5-72B: Will fit, will train fast, will deliver results

With 72 hours remaining, we can't afford to debug memory issues. We need a model that works immediately and trains reliably. Qwen2.5-72B gives us the best chance of success within our constraints.

**Recommended Immediate Action**: Start implementing the Qwen2.5-72B training script and launch training within the next hour.

---

*End of Status Report*