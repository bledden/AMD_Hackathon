# AMD Hackathon Q&A Agent - Detailed Project Plan

**Competition**: AMD Hackathon - Q&A Agent Tournament
**Timeline**: October 24-29, 2025 (5 days)
**Budget**: $30-50 of DigitalOcean AMD credits
**Platform**: Single MI300X GPU (192GB VRAM, $1.99/hr)

---

## Project Objectives

### Primary Goal
Build a fine-tuned Q&A agent that can:
1. Generate creative, challenging questions on a specific theme
2. Answer questions accurately and concisely
3. Compete effectively in bracket-style tournament

### Success Criteria
- **Minimum**: Working agent that participates without errors
- **Target**: Win at least one tournament round
- **Stretch**: Advance multiple rounds, top performance

---

## Technical Architecture

### Stack
```
Base Model: LLaMA 3 8B (recommended) or Mistral 7B
    ↓
Optimization: Unsloth (2x faster, 80% less memory)
    ↓
Fine-tuning: QLoRA/LoRA (4-bit quantization)
    ↓
Training: SFT on Q&A dataset (500-2000 pairs)
    ↓
Deployment: PyTorch 2.6.0 + ROCm 7.0.0 on MI300X
```

### Key Technologies
- **Unsloth**: Fast fine-tuning with AMD MI300X support
- **PyTorch**: Deep learning framework
- **ROCm**: AMD GPU acceleration
- **Transformers/TRL**: Training and inference
- **LoRA/QLoRA**: Parameter-efficient fine-tuning

---

## 5-Day Implementation Plan

### Day 1 (Friday, Oct 24) - SETUP & BASELINE
**Time**: 3-4 hours | **Cost**: ~$6-8 | **Status**: READY TO START

#### Tasks
1. **Deploy MI300X Instance** (30 min)
   - Log into amd.digitalocean.com
   - Select: Single MI300X + PyTorch 2.6.0 + ROCm 7.0.0
   - Configure SSH key
   - Launch instance
   - Document IP and credentials

2. **Environment Setup** (30 min)
   ```bash
   # Install Unsloth
   pip install "unsloth[rocm] @ git+https://github.com/unslothai/unsloth.git"

   # Install dependencies
   pip install transformers datasets trl accelerate peft bitsandbytes

   # Verify GPU
   rocm-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Choose Theme & Model** (1 hour)
   - **Recommended themes**: Science, History, Technology, Space
   - **Model choice**: LLaMA 3 8B (best balance)
   - Test model loading with Unsloth
   - Verify 4-bit quantization works

4. **Baseline Testing** (1-2 hours)
   - Load model in inference mode
   - Test basic question generation
   - Test basic answer generation
   - Document baseline quality
   - Estimate fine-tuning time

#### Deliverables
- [ ] Running MI300X instance
- [ ] Unsloth installed and verified
- [ ] Base model loaded successfully
- [ ] Baseline performance documented
- [ ] Theme selected

---

### Day 2 (Saturday, Oct 25) - DATASET & FIRST FINE-TUNE
**Time**: 4-6 hours | **Cost**: ~$8-12 | **Status**: PENDING

#### Tasks
1. **Dataset Strategy Selection** (1 hour)
   Choose one approach:

   **Option A: Existing Dataset** (Fastest)
   - Download SQuAD 2.0 or Natural Questions
   - Filter for chosen theme
   - Format for instruction tuning
   - Pros: Quick, high quality
   - Cons: May not match theme perfectly

   **Option B: Synthetic Generation** (More control)
   - Use GPT-4/Claude API to generate Q&A pairs
   - Create 500-1000 theme-specific pairs
   - Focus on tournament-relevant questions
   - Pros: Perfect theme match, creative questions
   - Cons: Requires API access/credits

   **Option C: Hybrid** (Recommended)
   - Start with 300-500 existing Q&A pairs
   - Augment with 200-300 synthetic pairs
   - Best of both approaches

2. **Dataset Creation** (2 hours)
   ```python
   # Target format
   [
       {
           "instruction": "Generate a challenging question about [theme]",
           "input": "",
           "output": "What is the relationship between...?"
       },
       {
           "instruction": "Answer: What is photosynthesis?",
           "input": "",
           "output": "Photosynthesis is the process..."
       }
   ]
   ```

   - Create 500-1000 question generation examples
   - Create 500-1000 question answering examples
   - Balance difficulty levels (easy, medium, hard)
   - Validate formatting
   - Split: 90% train, 10% validation

3. **First Fine-tune** (2-3 hours)
   - Configure QLoRA (r=16, 4-bit)
   - Set training parameters:
     - Batch size: 2
     - Gradient accumulation: 4
     - Learning rate: 2e-4
     - Steps: 60-100 (adjust based on dataset size)
   - Start training
   - Monitor loss curve
   - Save checkpoint

4. **Initial Evaluation** (1 hour)
   - Test question generation (10 samples)
   - Test answer accuracy (20 questions)
   - Compare to baseline
   - Document improvements/issues

#### Deliverables
- [ ] Dataset of 500-2000 Q&A pairs
- [ ] First fine-tuned model checkpoint
- [ ] Training loss curves
- [ ] Initial evaluation results
- [ ] Notes on what to improve

---

### Day 3 (Sunday, Oct 26) - OPTIMIZATION
**Time**: 4-6 hours | **Cost**: ~$8-12 | **Status**: PENDING

#### Tasks
1. **Dataset Refinement** (2 hours)
   Based on Day 2 evaluation:
   - Add more diverse question types
   - Balance easy/medium/hard questions
   - Improve answer formatting
   - Add edge cases
   - Consider synthetic data augmentation (see AMD blog post)
   - Target: 1000-2000 total pairs

2. **Hyperparameter Tuning** (1 hour)
   - Adjust learning rate if needed
   - Modify LoRA rank (try r=8 or r=32)
   - Tune temperature for generation
   - Adjust max sequence length
   - Document all changes

3. **Second Fine-tune** (2-3 hours)
   - Use improved dataset
   - Use refined hyperparameters
   - Longer training if needed (100-200 steps)
   - Save multiple checkpoints
   - Compare checkpoints

4. **Comprehensive Evaluation** (1 hour)
   - Question quality assessment (30 samples)
   - Answer accuracy test (50 questions)
   - Difficulty calibration
   - Response consistency
   - Document best checkpoint

#### Deliverables
- [ ] Refined dataset (1000-2000 pairs)
- [ ] Second fine-tuned model (improved)
- [ ] Checkpoint comparison
- [ ] Comprehensive evaluation report
- [ ] Best model selected

---

### Day 4 (Monday, Oct 27) - TOURNAMENT OPTIMIZATION
**Time**: 3-4 hours | **Cost**: ~$6-8 | **Status**: PENDING

#### Tasks
1. **Tournament Strategy Development** (1 hour)

   **Question Generation Strategy**:
   - Difficulty sweet spot: Hard but not impossible
   - Question types that test understanding, not just recall
   - Edge cases in the domain
   - Prompt engineering:
     ```python
     "Generate a challenging but fair question about [topic]
     that tests deep understanding rather than simple recall."
     ```

   **Answer Strategy**:
   - Accuracy > eloquence
   - Concise answers (50-100 words ideal)
   - Confidence calibration (low temp = 0.1-0.3)
   - Cite reasoning when relevant

2. **Inference Pipeline** (1 hour)
   - Create robust inference script
   - Add error handling
   - Implement temperature tuning
   - Test with various inputs
   - Optimize for speed

3. **Final Fine-tune** (1-2 hours, optional)
   - If needed based on testing
   - Focus on weak areas
   - Quick iteration
   - Save final checkpoint

4. **Documentation** (30 min)
   - Approach writeup
   - Model card
   - Tournament strategy notes
   - Submission preparation

#### Deliverables
- [ ] Tournament-optimized inference pipeline
- [ ] Strategy document
- [ ] Final model checkpoint
- [ ] Submission-ready package

---

### Day 5 (Tuesday, Oct 28) - POLISH & TESTING
**Time**: 2-3 hours | **Cost**: ~$4-6 | **Status**: PENDING

#### Tasks
1. **Integration Testing** (1 hour)
   - End-to-end tournament simulation
   - Test question generation → answering flow
   - Verify response times
   - Check error handling
   - Stress test

2. **Final Adjustments** (1 hour)
   - Fix any bugs found
   - Final prompt tuning
   - Response formatting
   - Add fallback mechanisms

3. **Submission Preparation** (30 min)
   - Package model and code
   - Write submission notes
   - Prepare deployment instructions
   - Final documentation review

4. **Buffer Time** (30 min)
   - Handle unexpected issues
   - Last-minute testing
   - Backup checkpoint

#### Deliverables
- [ ] Fully tested agent
- [ ] Submission package
- [ ] Deployment documentation
- [ ] Ready for competition

---

### Day 6 (Wednesday, Oct 29) - COMPETITION DAY
**Time**: Variable | **Cost**: As needed | **Status**: PENDING

#### Tasks
1. **Pre-Competition**
   - Submit agent
   - Verify deployment
   - Final systems check

2. **During Competition**
   - Monitor performance
   - Track tournament progress
   - Note strengths/weaknesses

3. **Post-Competition**
   - Analyze results
   - Gather feedback
   - Document learnings
   - Celebrate! 🎉

---

## Key Implementation Details

### Dataset Format Template
```python
# Question Generation Examples
{
    "instruction": "Generate a [difficulty] question about [topic] that [requirement]",
    "input": "",
    "output": "[Generated question]"
}

# Question Answering Examples
{
    "instruction": "Answer this question accurately and concisely: [question]",
    "input": "",
    "output": "[Answer]"
}
```

### Training Configuration
```python
from unsloth import FastLanguageModel
from trl import SFTTrainer

# Model loading
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True
)

# LoRA config
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing=True
)

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=100,
    fp16=False,
    bf16=True,
    logging_steps=1,
    optim="adamw_8bit",
    output_dir="outputs"
)
```

### Inference Template
```python
FastLanguageModel.for_inference(model)

# Generate question
question_prompt = f"""Generate a challenging question about {theme}.

Question:"""

question = model.generate(
    tokenizer(question_prompt, return_tensors="pt").to("cuda"),
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9
)

# Generate answer
answer_prompt = f"""Answer this question accurately:

Question: {question}

Answer:"""

answer = model.generate(
    tokenizer(answer_prompt, return_tensors="pt").to("cuda"),
    max_new_tokens=256,
    temperature=0.3,
    top_p=0.95
)
```

---

## Risk Management

### Technical Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU OOM | High | Use 4-bit quantization, reduce batch size, gradient checkpointing |
| Slow training | Medium | Unsloth optimization, smaller dataset, fewer steps |
| Poor Q&A quality | High | Better dataset, more training, prompt engineering |
| Deployment issues | Medium | Test early, document setup, have backup plan |

### Budget Risks
| Scenario | Cost | Plan |
|----------|------|------|
| Optimal path | $30-40 | Follow 5-day plan, stop instance when idle |
| Extended training | $45-55 | Additional fine-tuning iterations |
| Instance left running | $100+ | Set alerts, use auto-stop, monitor closely |

### Mitigation Strategies
1. **Save checkpoints frequently** - Can resume if interrupted
2. **Stop instance when not using** - Prevents runaway costs
3. **Test locally first** - Validate code before GPU time
4. **Have fallback dataset** - Don't depend on external APIs
5. **Document everything** - Easy to pick up next day

---

## Cost Tracking

### Estimated Costs
- Day 1 (4 hrs): $8
- Day 2 (6 hrs): $12
- Day 3 (6 hrs): $12
- Day 4 (4 hrs): $8
- Day 5 (3 hrs): $6
- Competition (2 hrs): $4
- **Total**: ~$50

### Cost Optimization Tips
1. Develop scripts locally, only use GPU for training/inference
2. Use tmux/screen to persist sessions
3. Stop instance between work sessions
4. Use DigitalOcean snapshots for backups
5. Monitor with: `watch -n 60 "echo Hours: $(($(date +%s) - START_TIME) / 3600)"`

---

## Success Metrics

### Technical Metrics
- **Training**: Loss < 1.0, smooth convergence
- **Questions**: Diverse, creative, challenging but fair
- **Answers**: >80% accuracy on validation set
- **Inference**: <5 seconds per Q&A pair

### Competition Metrics
- **Primary**: Win at least 1 round
- **Secondary**: Top 50% of participants
- **Stretch**: Top 25% or finals appearance

### Learning Metrics
- Understand Unsloth optimization
- Experience with AMD MI300X + ROCm
- Q&A agent architecture knowledge
- Tournament strategy insights

---

## Resources & References

### Essential Links
- **Unsloth GitHub**: https://github.com/unslothai/unsloth
- **AMD Blog Post**: https://www.amd.com/en/developer/resources/technical-articles/2025/10x-model-fine-tuning-using-synthetic-data-with-unsloth.html
- **DigitalOcean AMD**: https://amd.digitalocean.com
- **ROCm Docs**: https://rocm.docs.amd.com/

### Datasets
- **SQuAD 2.0**: https://rajpurkar.github.io/SQuAD-explorer/
- **Natural Questions**: https://ai.google.com/research/NaturalQuestions
- **TriviaQA**: https://nlp.cs.washington.edu/triviaqa/

### Community Support
- **Unsloth Discord**: https://discord.gg/unsloth
- **AMD ROCm Issues**: https://github.com/ROCm/ROCm/issues
- **Hugging Face Forums**: https://discuss.huggingface.co/

---

## Next Steps

### Immediate Actions (Today)
1. ✅ Initialize git repository
2. ✅ Create project structure
3. ⬜ Set up directory structure
4. ⬜ Write setup scripts
5. ⬜ Create dataset pipeline
6. ⬜ Write training script
7. ⬜ Write inference script

### Before Deployment
1. Review competition rules and format
2. Choose theme carefully (competitive advantage)
3. Test all scripts locally with dummy data
4. Prepare troubleshooting guide
5. Have backup dataset ready

### During Competition
1. Monitor agent performance
2. Take notes on what works/doesn't work
3. Engage with other participants
4. Gather ideas for improvements

---

## Post-Competition Plans

### If Successful
- Write up approach and share with community
- Refine agent for potential future competitions
- Explore RL approaches with remaining credits
- Apply learnings to other projects

### If Unsuccessful
- Analyze what went wrong
- Identify improvement areas
- Still valuable learning experience
- Consider iterating with remaining credits

### Long-term
- Experience applies to dendritic research project
- AMD MI300X platform familiarity
- Unsloth optimization techniques
- Q&A agent architecture patterns

---

## Notes

### Key Principles
1. **Keep it simple** - SFT with good dataset beats complex RL
2. **Start small, iterate fast** - Test quickly, improve continuously
3. **Quality over quantity** - Better dataset > more training
4. **Monitor costs** - Stop instance when not using
5. **Have fun** - It's a competition, enjoy the learning!

### Remember
- This is a 5-day sprint, not a 2-month research project
- Perfect is the enemy of good
- Ship working > ship perfect
- Learn from experience
- Connect with community

---

**Let's build an awesome Q&A agent! 🚀**
