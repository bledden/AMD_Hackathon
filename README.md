# AMD Hackathon - Q&A Agent Tournament 🏆

**Status**: ✅ **TOURNAMENT SUBMISSION COMPLETE**
**Deadline**: Wednesday, October 29, 2025 @ 7:00 PM PT
**Team**: Blake Ledden + Claude (Anthropic)
**Final Result**: 92% Accuracy with Qwen2.5-7B-Instruct

---

## 🎯 Final Solution

**Model**: Qwen2.5-7B-Instruct with Timeout Protection
**Performance**:
- ✅ **Accuracy**: 92% (46/50 on validation set)
- ✅ **Speed**: Average 0.228s, guaranteed <6s with timeout fallback
- ✅ **Compliance**: Meets all tournament requirements
- ✅ **Architecture**: Pure baseline model (no adapters/fine-tuning)

**Key Innovation**: Threading-based timeout protection guarantees <6s response time with graceful fallback to statistical best-guess ("B") for slow questions.

---

## 📚 Complete Documentation

### [COMPLETE_JOURNEY_DOCUMENTATION.md](COMPLETE_JOURNEY_DOCUMENTATION.md)

**Comprehensive 20-page document covering our entire journey:**

- **5 Training Attempts** (4 failures → 1 breakthrough)
  - Attempt 1: Reasoning distillation (0-3% accuracy)
  - Attempt 2: Simple Q→A fine-tuning (2% accuracy - mode collapse)
  - Attempt 3: Targeted domain training (0% accuracy - mode collapse)
  - Attempt 4: Ultra-minimal training (73.5% accuracy - no improvement)
  - Attempt 5: Qwen2.5-7B baseline (92% accuracy - SUCCESS!)

- **Technical Deep-Dives**
  - Mode collapse in adapter training: What it is, why it happens, how we detected it
  - Why DeepSeek-R1-32B (32B params, 61-73% accuracy) lost to Qwen2.5-7B (7B params, 92% accuracy)
  - Timeout protection implementation with threading
  - Why reasoning chains failed in fine-tuning

- **Decision Analysis**
  - Every pivot point with rationale
  - What drove each strategic choice
  - Lessons learned from each failure

- **Performance Statistics**
  - Complete training time breakdown
  - Accuracy comparisons across all attempts
  - Speed analysis and outlier detection

---

## 🚀 Tournament Agents

### Answer Agent: [AIAC/agents/answer_model.py](AIAC/agents/answer_model.py)
```python
# Qwen2.5-7B-Instruct with timeout protection
- Model: /workspace/models/qwen2.5_7b_instruct
- Timeout: 5.5 seconds (guarantees <6s requirement)
- Fallback: Returns "B" if generation exceeds timeout
- Accuracy: 92% validated
```

### Question Agent: [AIAC/agents/question_model.py](AIAC/agents/question_model.py)
```python
# Pre-generated question pool selector
- Response time: <1s
- No generation needed (reads from pool)
- Random selection with no-repeat tracking
```

---

## 🎓 Key Lessons Learned

### 1. Model Selection > Model Size
- Qwen2.5-7B (7B params): 92% accuracy
- DeepSeek-R1-32B (32B params): 61-73% accuracy
- **Lesson**: Task-specific pre-training (instruction-tuning) beats raw parameter count

### 2. Mode Collapse in Adapter Training
- All fine-tuning attempts (2-4) suffered mode collapse
- Adapters learned to output constant tokens ("10000000") instead of task patterns
- **Lesson**: Pre-trained baseline often beats fine-tuned adapters under time pressure

### 3. Timeout Protection is Essential
- ~1% of questions exceed time limits unpredictably
- No correlation with length, complexity, or input size
- **Lesson**: Graceful degradation (timeout + fallback) beats optimization alone

### 4. Reasoning Chains ≠ Better Performance
- Training with CoT reasoning chains → model rambles endlessly (0-3% accuracy)
- Simple Q→A format → still failed due to mode collapse
- **Lesson**: CoT is great for prompting, terrible for MCQ fine-tuning

### 5. Fast Iteration Under Pressure
- 4 training attempts in 3 hours taught us what NOT to do
- Final hour pivot to Qwen2.5-7B found 92% solution
- **Lesson**: Fail fast, pivot quickly, always have fallback plans

---

## 📊 Training Attempts Summary

| Attempt | Approach | Time | Accuracy | Result | Key Issue |
|---------|----------|------|----------|--------|-----------|
| 1 | Reasoning Distillation | 2h | 0-3% | ❌ | Endless rambling |
| 2 | Simple Q→A (5K) | 33m | 2% | ❌ | Mode collapse |
| 3 | Targeted (6K) | 11m | 0% | ❌ | Mode collapse |
| 4 | Ultra-Minimal (100) | 2.6m | 73.5% | ⚠️ | No improvement |
| 5 | **Qwen2.5-7B Baseline** | **0m** | **92%** | **✅** | **Success!** |

**Total training time invested**: ~3 hours
**Final solution training time**: 0 minutes (baseline model)

---

## 🔧 Technical Stack

**Infrastructure**:
- **Server**: AMD MI300X (192GB VRAM)
- **OS**: ROCm 6.2.41133
- **Container**: Docker `rocm`
- **Location**: 129.212.186.194

**Software**:
- **Framework**: Transformers 4.57.1
- **Model Loading**: AutoModelForCausalLM
- **Precision**: bfloat16
- **Device**: auto (GPU)

**Deployment**:
- **AIAC Directory**: `/workspace/AIAC/agents/`
- **Model Path**: `/home/rocm-user/AMD_Hackathon/models/qwen2.5_7b_instruct`
- **Interface**: Module-based (`python -m AIAC.agents.answer_model`)

---

## 📈 Performance Metrics

### Answer Agent (Qwen2.5-7B)
- **Validation Set**: 50 questions (random sample)
- **Accuracy**: 92.0% (46/50 correct)
- **Average Response Time**: 0.228s
- **Median Response Time**: <0.2s
- **Max Response Time**: 9.13s (handled by timeout)
- **Guaranteed Max**: <6s (with timeout protection)
- **Timeout Fallback**: "B" (statistically most common)

### Question Agent
- **Pool Size**: Configurable (`/workspace/question_pool.json`)
- **Selection Method**: Random with no-repeat tracking
- **Response Time**: <1s (instant file read)
- **Compliance**: <10s requirement easily met

---

## 🎯 What Worked

1. **Pre-trained instruction-tuned model** (Qwen2.5-7B) over fine-tuned adapter
2. **Smaller specialized model** (7B) over larger general model (32B)
3. **Timeout protection** with graceful fallback over perfect optimization
4. **Fast pivots** when approaches failed (research new model > debug failed adapter)
5. **Built-in sanity checks** to detect mode collapse immediately

---

## 💡 What We Would Do Differently

### With More Time:
1. Test multiple baseline models on Day 1 (not final hour)
2. Skip fine-tuning entirely if strong baselines exist
3. Build timeout protection from the start
4. Simplify architecture (no ensemble if 92% achievable with single model)

### With Less Time:
1. Test 3-5 pre-trained models immediately
2. Pick best performer
3. Add timeout wrapper
4. Submit
5. **(Don't spend hours on fine-tuning)**

---

## 📁 Repository Structure

```
AMD_Hackathon/
├── AIAC/
│   └── agents/
│       ├── __init__.py
│       ├── answer_model.py       # 92% accuracy Qwen2.5-7B + timeout
│       └── question_model.py     # Question pool selector
│
├── COMPLETE_JOURNEY_DOCUMENTATION.md  # Full story (20 pages)
├── README.md                          # This file
│
├── scripts/
│   ├── test_qwen7b_quick.py          # Validation test (50 questions)
│   ├── train_*.py                     # Training attempts 1-4 (failed)
│   └── analyze_failures.py           # Baseline analysis
│
└── tournament_server_qwen7b.py       # HTTP API (optional deployment)
```

---

## 🏆 Competition Strategy

**Our Approach**:
- Prioritize **accuracy** (92% achieved)
- Guarantee **compliance** (<6s with timeout)
- Ensure **reliability** (graceful fallback, no crashes)
- Maintain **simplicity** (baseline model, no complex ensemble)

**Why This Wins**:
1. **High accuracy** beats most fine-tuned approaches
2. **Speed compliance** guaranteed (some competitors may violate)
3. **Reliability** matters in tournament play (no edge cases)
4. **Simplicity** reduces failure modes

---

## 📞 Connection & Testing

### SSH Access
```bash
ssh amd-hackathon
# Enter passphrase when prompted
```

### Test Answer Agent
```bash
ssh amd-hackathon "docker exec rocm bash -c 'cd /workspace && python3 -m AIAC.agents.answer_model'"
```

### Check Tournament Server
```bash
ssh amd-hackathon "docker exec rocm curl http://localhost:5000/health"
# Expected: {"status":"ready","model":"Qwen2.5-7B-Instruct","accuracy":"92%"}
```

---

## 🎓 Research & Citations

**Models**:
- Qwen2.5-7B-Instruct: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- DeepSeek-R1-Distill-Qwen-32B: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

**Key Insights**:
- Model selection matters more than size for structured tasks
- Instruction-tuning pre-training provides strong MCQ baselines
- Mode collapse is common in aggressive adapter fine-tuning
- Timeout protection is essential for unpredictable generation latency

---

## 📄 License & Attribution

**Team**: Blake Ledden + Claude (Anthropic)
**Event**: AMD Hackathon - Q&A Agent Tournament
**Date**: October 29, 2025
**Hardware**: AMD MI300X GPU (192GB VRAM)

**Achievement**: 92% accuracy solution deployed under 3-hour deadline pressure after 4 failed training attempts.

---

## 🔗 Quick Links

- **Full Journey**: [COMPLETE_JOURNEY_DOCUMENTATION.md](COMPLETE_JOURNEY_DOCUMENTATION.md)
- **Answer Agent**: [AIAC/agents/answer_model.py](AIAC/agents/answer_model.py)
- **Question Agent**: [AIAC/agents/question_model.py](AIAC/agents/question_model.py)
- **Tournament Server**: [tournament_server_qwen7b.py](tournament_server_qwen7b.py)

---

**Status**: ✅ Tournament-ready and submitted
**Result**: 92% accuracy, <6s guaranteed, fully compliant
**Lessons**: 5 attempts, 4 failures, 1 breakthrough - documented for future reference
