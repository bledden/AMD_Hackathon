# AMD x Unsloth Synthetic Data Hackathon - Complete Briefing

**Purpose**: This document provides complete context for external reviewers (AI models, team members, advisors) to understand the hackathon challenge and help make strategic decisions.

**Last Updated**: October 27, 2025
**Deadline**: Wednesday, October 29, 2025 @ 7:00 PM PT (48 hours remaining)
**Current Status**: Decision phase - choosing between 1, 2, or 3 model training approaches

---

## Table of Contents

1. [Competition Overview](#competition-overview)
2. [Challenge Description](#challenge-description)
3. [Technical Requirements](#technical-requirements)
4. [Rules & Restrictions](#rules--restrictions)
5. [Scoring & Winning Criteria](#scoring--winning-criteria)
6. [Our Current Situation](#our-current-situation)
7. [Decision Points](#decision-points)
8. [What We Need From You](#what-we-need-from-you)

---

## Competition Overview

### Event Details

**Competition**: AMD x Unsloth Synthetic Data Hackathon
**Track**: Synthetic-Data Track (Q&A Agents)
**Organizers**: AMD, Unsloth AI, PyTorch
**Platform**: DigitalOcean AMD Developer Cloud
**Format**: Bracket-style tournament (teams compete head-to-head)

### Prizes (Synthetic-Data Track)

- 🥇 **1st Place**: $3,000 + 1,200 hrs GPU credits + Trophy
- 🥈 **2nd Place**: $1,500 + 600 hrs GPU credits + Trophy
- 🥉 **3rd Place**: $900 + 300 hrs GPU credits + Trophy

### Timeline

- **Registration**: Opened September 20, 2025
- **Competition Starts**: Saturday, October 26, 2025
- **Submission Deadline**: **Wednesday, October 29, 2025 @ 7:00 PM PT** (STRICT)
- **Finals/Awards**: October 30, 2025 (San Francisco)

---

## Challenge Description

### The Game: "20 Questions" Tournament

Build AI agents that compete in a Q&A tournament styled like "20 Questions":

**Two Agents Required:**

1. **Q-Agent (Question Generator)**
   - Input: A topic (string)
   - Output: A challenging multiple-choice question with 4 options (A/B/C/D)
   - Goal: Create questions that stump the opponent

2. **A-Agent (Answerer)**
   - Input: A question with 4 choices
   - Output: The correct answer (A, B, C, or D)
   - Goal: Answer as many questions correctly as possible

### Tournament Structure

**Each match has 2 innings:**

**Inning 1:**
- Your Q-Agent generates 20 questions → Opponent's A-Agent tries to answer
- Opponent's Q-Agent generates 20 questions → Your A-Agent tries to answer

**Inning 2:**
- (Same format repeats)

**Total per match:** 40 questions asked, 40 questions answered

### Scoring System

**Your Score = Q-Agent Points + A-Agent Points**

- **Q-Agent Points**: Number of questions opponent FAILS to answer (max 20 per inning)
- **A-Agent Points**: Number of questions you CORRECTLY answer (max 20 per inning)

**Example Tournament:**

**Inning 1:**
- Your Q-Agent: Opponent answers 12/20 → You score **8 points** (20 - 12)
- Your A-Agent: You answer 16/20 correctly → You score **16 points**

**Inning 2:**
- Your Q-Agent: Opponent answers 14/20 → You score **6 points** (20 - 14)
- Your A-Agent: You answer 15/20 correctly → You score **15 points**

**Your Total Score**: 8 + 16 + 6 + 15 = **45 points**
**Maximum Possible**: 80 points (stump all 40, answer all 40)

**Winner**: Highest score advances in bracket

---

## Technical Requirements

### Hardware Requirements

**CRITICAL CONSTRAINT**: AMD platform only allows **1 droplet per team**

**Droplet Specifications:**
- **Platform**: DigitalOcean AMD Developer Cloud (amd.digitalocean.com)
- **Image**: **ROCm 6.4.0** (NOT PyTorch + ROCm 7.0.0) ⚠️
- **GPU**: Single MI300X
  - 1 GPU - 192 GB VRAM
  - 20 vCPU - 240 GB RAM
  - Boot disk: 720 GB NVMe
  - Scratch disk: 5 TB NVMe
- **Cost**: $1.99/hour
- **IPv6**: Not required (use IPv4 only)

### Software Requirements

**Docker Container (REQUIRED):**
```bash
docker run -it --rm \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size 8G \
  -w /workspace \
  --name rocm-jupyter \
  edaamd/aiac:latest
```

**Key Libraries:**
- Unsloth (for fast fine-tuning with ROCm)
- PyTorch with ROCm 6.4.0 support
- Transformers, TRL, PEFT
- Python (all code must be in Python)

**Development Environment:**
- JupyterLab (accessed via port forwarding)
- Must work within `AIAC/` folder structure provided in container

---

## Rules & Restrictions

### MUST FOLLOW (Hard Restrictions)

#### 1. **JSON Output Formats (EXACT FORMAT REQUIRED)**

**Q-Agent Output:**
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
  "answer": "A",
  "explanation": "brief explanation within 100 words"
}
```

**A-Agent Output:**
```json
{
  "answer": "B",
  "reasoning": "brief reasoning within 100 words"
}
```

**CRITICAL**:
- Only `answer` must be a single letter: A, B, C, or D
- Organizers extract ONLY the `answer` key for evaluation
- Invalid format = 0 points (automatic loss)
- `explanation` and `reasoning` are optional but recommended

---

#### 2. **Speed Limits (STRICTLY ENFORCED)**

- **Question generation**: Under **10 seconds per question**
- **Answer generation**: Under **6 seconds per answer**
- **Enforcement**: Requests exceeding limits will be **ignored/not considered**

This is a HARD LIMIT. If your agent times out, you get 0 points for that question.

---

#### 3. **NO RAG (Retrieval Augmented Generation)**

- Cannot use external knowledge bases
- Cannot use web search during inference
- Cannot use vector databases or retrieval systems
- Pure model inference only

---

#### 4. **NO Adversarial Approaches**

- No jailbreaking
- No prompt injection attacks
- No making models hallucinate intentionally
- No exploiting opponent's weaknesses through adversarial prompts

---

#### 5. **English Only**

- Both Q-Agent and A-Agent must use English
- No multilingual approaches
- No code-switching

---

#### 6. **Token Limits**

- Must respect `max_tokens` limits in `agent.yaml` and `qgen.yaml`
- Other parameters (temperature, top_p, etc.) can be changed
- Exceeding token limits may result in disqualification

---

#### 7. **Topics**

- Questions must be general Q&A (broad topics)
- Example topics from starter code: "Seating Arrangements", "Blood Relations"
- **These are examples only** - NOT restricted to these topics
- Can use any topic suitable for general knowledge Q&A

---

#### 8. **File Structure (REQUIRED)**

Must have **four `.py` files** in `agents/` directory:
- `question_agent.py`
- `question_model.py`
- `answer_agent.py`
- `answer_model.py`

Missing any file = potential disqualification

---

#### 9. **Submission Deadline**

- **Wednesday, October 29, 2025 @ 7:00 PM PT**
- NO late submissions accepted
- NO changes after deadline (may result in disqualification)

---

### MUST NOT DO

❌ Submit after deadline
❌ Use RAG or external knowledge retrieval
❌ Use adversarial techniques
❌ Output invalid JSON formats
❌ Exceed speed limits (10s/6s)
❌ Use non-English responses
❌ Exceed token limits in YAML configs
❌ Miss any required Python files
❌ Use languages other than Python

---

## Scoring & Winning Criteria

### What Determines the Winner?

**50% Weight: A-Agent Performance**
- Accuracy (correctly answering opponent's questions)
- Broad knowledge across many topics
- Strong reasoning ability
- Edge case handling

**50% Weight: Q-Agent Performance**
- Difficulty (stumping opponent)
- Question creativity
- Plausible distractors (wrong answers that seem right)
- Strategic topic selection

### Key Metrics

**A-Agent Quality Indicators:**
- Accuracy: 85-90% = Excellent, 75-85% = Good, <75% = Weak
- Reasoning: Clear logical explanations
- Consistency: Similar performance across topics
- Speed: <3 seconds per answer = Safe, 4-5s = Risky, >5s = Danger

**Q-Agent Quality Indicators:**
- Stump Rate: 40%+ = Excellent, 30-40% = Good, <30% = Weak
- Creativity: Novel question types, unexpected angles
- Distractor Quality: Wrong answers that experts might pick
- Correctness: 100% accuracy (wrong answers = instant loss)

### Tournament Bracket

- Single elimination (lose once = out)
- Seeding based on qualification rounds
- Final winner takes 1st place ($3,000)
- Runner-up takes 2nd place ($1,500)
- Semi-final losers share 3rd place ($900)

---

## Our Current Situation

### What We've Done So Far

✅ **Research Phase Complete**
- Analyzed competition requirements thoroughly
- Selected 3 candidate models (Phi-4 14B, Qwen3 8B, Mistral NeMo 12B)
- Validated all models have Unsloth support
- Confirmed speed requirements can be met

✅ **Infrastructure Understanding**
- ROCm 6.4.0 requirement confirmed with organizers
- Docker container setup understood
- AIAC folder structure documented

✅ **Strategy Documentation**
- Created comprehensive project plan
- Documented 3-agent parallel strategy
- Prepared training configs for all models

❌ **Not Yet Done**
- Droplet not created (waiting on decision)
- No training started
- No datasets prepared
- No code written

### Current Constraint: Single Droplet

**Original Plan**: Train 3 models in parallel on 3 droplets
**Reality**: AMD platform only allows 1 droplet per team
**Impact**: Must train models sequentially instead of parallel

### Time Remaining

**Current Time**: Sunday, October 27, 2025 (morning)
**Deadline**: Wednesday, October 29, 2025 @ 7:00 PM PT
**Hours Remaining**: ~60 hours (2.5 days)

**Time Budget Breakdown:**
- Setup & verification: 4-6 hours
- Dataset preparation: 4-8 hours
- Training: 30-100 hours (depends on # of models)
- Testing & evaluation: 10-30 hours
- Submission prep: 4-6 hours

**Challenge**: Must fit everything within 60-hour window

---

## Decision Points

### Critical Decision: How Many Models to Train?

**Budget Available**: $300 in GPU credits
**Budget Target**: $200-250 (save remainder for other projects)

We must choose between:

---

### **Option A: Train 3 Models** (~$240 total)

**Models:**
1. Phi-4 14B (Foundation) - 35 hours = $70
2. Qwen3 8B (Challenger) - 35 hours = $70
3. Mistral NeMo 12B (Specialist) - 30 hours = $60
4. Testing - 20 hours = $40

**Timeline:**
- Saturday evening: Start Phi-4
- Sunday evening: Start Qwen3
- Monday evening: Start Mistral NeMo
- Tuesday evening: All done, begin testing
- Wednesday: Final testing, select winner, submit

**Pros:**
✅ Maximum strategy diversity (3 approaches)
✅ Best hedge (if one fails, have 2 backups)
✅ Complete comparison data
✅ 85% probability at least one model places
✅ Can submit best performer

**Cons:**
❌ Undertrained (2.5 epochs vs ideal 3)
❌ Tight testing window (only 20 hours)
❌ Highest cost ($240)
❌ Complex to manage
❌ Risk of not finishing in time

**Expected Outcome:**
- Phi-4: 22-25 points/round (good but undertrained)
- Qwen3: 22-25 points/round (good but undertrained)
- Mistral NeMo: 22-24 points/round (undertrained)
- Win probability: 85%
- Expected prize: $1,500

---

### **Option B: Train 2 Models** (~$240 total) ⭐ **CURRENT RECOMMENDATION**

**Models:**
1. Phi-4 14B (Foundation) - 50 hours = $100
2. Qwen3 8B (Challenger) - 50 hours = $100
3. Testing - 20 hours = $40

**Timeline:**
- Saturday evening: Start Phi-4
- Monday morning: Start Qwen3
- Tuesday evening: Both done, begin testing
- Wednesday: Testing, select winner, submit

**Pros:**
✅ Fully trained models (3 complete epochs)
✅ Good hedge (2 different strategies)
✅ Exactly $100/model budget target
✅ 75% probability at least one model places
✅ Adequate testing time (20 hours)
✅ Covers foundation + creative approaches

**Cons:**
❌ No domain specialist (Mistral NeMo)
❌ Only 2 comparison points
❌ Less comprehensive
❌ If both fail, no third backup

**Expected Outcome:**
- Phi-4: 24-27 points/round (well-trained)
- Qwen3: 24-27 points/round (well-trained)
- Win probability: 75%
- Expected prize: $1,350

---

### **Option C: Train 1 Model** (~$200 total)

**Model:**
1. Phi-4 14B only
   - First training: 30 hours = $60
   - Evaluation: 10 hours = $20
   - Second training: 30 hours = $60
   - Testing: 30 hours = $60

**Timeline:**
- Saturday evening: Start Phi-4 (run 1)
- Sunday evening: Evaluate, refine dataset
- Monday evening: Start Phi-4 (run 2)
- Tuesday evening: Done, begin extensive testing
- Wednesday: Final testing, submit

**Pros:**
✅ Lowest cost (~$200)
✅ Can do 2 full training iterations
✅ 30+ hours testing time (most refinement)
✅ Most polished single model
✅ Learn from first run mistakes
✅ Simplest to manage

**Cons:**
❌ All eggs in one basket
❌ No backup if Phi-4 wrong for task
❌ No comparative data
❌ Can't test different strategies
❌ High risk

**Expected Outcome:**
- Phi-4 (iteration 1): 23-26 points/round
- Phi-4 (iteration 2): 25-28 points/round (refined)
- Win probability: 60%
- Expected prize: $1,000

---

## Model Comparison (Context for Decision)

### Phi-4 14B (Microsoft, 2025)

**Strengths:**
- Best reasoning (matches GPT-4o-mini on benchmarks)
- Excellent A-Agent potential (85-90% accuracy)
- Broad knowledge base
- Full Unsloth support with bug fixes
- Fits in <15GB VRAM

**Weaknesses:**
- Q-Agent may be predictable (30-40% stump rate)
- Larger model (14B = slower training)
- Less creative than Qwen3

**Best For:**
- Answering questions correctly (A-Agent focus)
- Reliable, consistent performance
- Foundation strategy

**Estimated Performance:**
- A-Agent: 17-18/20 correct (85-90%)
- Q-Agent: 6-8/20 stumps (30-40%)
- **Total: 23-26 points/round**

---

### Qwen3 8B (Alibaba, 2025)

**Strengths:**
- Creative question generation (35-45% stump rate)
- Hybrid reasoning architecture
- Dynamic 2.0 quantization (best quality)
- 8x longer context support
- Newest model (2025)

**Weaknesses:**
- Slightly lower A-Agent accuracy (80-85%)
- Less proven than Phi-4
- May generate too-difficult questions

**Best For:**
- Creative Q-Agent questions
- Stumping opponents
- Challenger strategy

**Estimated Performance:**
- A-Agent: 16-17/20 correct (80-85%)
- Q-Agent: 7-9/20 stumps (35-45%)
- **Total: 23-26 points/round**

---

### Mistral NeMo 12B (Mistral, 2024)

**Strengths:**
- Best JSON compliance (98% accuracy)
- Domain specialist (tech/science focus)
- Superior instruction following
- Most memory efficient (12GB)
- 128K context window

**Weaknesses:**
- Slower inference (74 tok/s vs 100+)
- Narrower knowledge (domain-focused)
- 2024 model (not 2025)
- Lower A-Agent accuracy on general topics

**Best For:**
- Domain-specific questions
- Perfect JSON output
- Exploiting opponent weaknesses

**Estimated Performance:**
- A-Agent: 16-17/20 correct (82-87%)
- Q-Agent: 7-9/20 stumps (35-45%)
- **Total: 23-26 points/round**

**Special Note:**
If opponent is weak in tech/science, NeMo's stump rate can reach 50%+ (10+/20), making it the winner.

---

## What We Need From You

### Questions for Review

1. **Strategy Selection:**
   - Which option (A/B/C) gives us the best chance to win?
   - Is it worth training 3 models with less training quality?
   - Or better to train 2 models fully?

2. **Model Selection:**
   - If training 2 models, is Phi-4 + Qwen3 the right pair?
   - Should we include Mistral NeMo despite being a specialist?
   - Is Phi-4's reasoning advantage worth the slower training?

3. **Risk Assessment:**
   - What's the biggest risk we're not considering?
   - Should we prioritize A-Agent accuracy or Q-Agent creativity?
   - Is 20 hours enough testing time?

4. **Technical Concerns:**
   - Are there any hackathon requirements we're missing?
   - Is our understanding of the scoring system correct?
   - Any red flags in our approach?

5. **Time Management:**
   - Can we realistically train 3 models by Wednesday 7 PM?
   - Should we reduce epochs from 3 to 2.5 to fit more models?
   - Is sequential training the right approach?

---

## Key Facts Summary

### MUST KNOW

✅ **Deadline**: Wednesday, October 29 @ 7 PM PT (STRICT)
✅ **Platform Limit**: Only 1 droplet allowed (sequential training required)
✅ **Image**: ROCm 6.4.0 (NOT PyTorch + ROCm 7.0.0)
✅ **Docker**: Must use `edaamd/aiac:latest` container
✅ **JSON Format**: Exact format required (invalid = 0 points)
✅ **Speed Limits**: 10s for questions, 6s for answers (exceed = ignored)
✅ **No RAG**: Pure model inference only
✅ **File Structure**: 4 required Python files in `agents/` directory
✅ **Scoring**: (Questions that stump opponent) + (Questions you answer correctly)

### MUST NOT DO

❌ Use RAG or external retrieval
❌ Exceed speed limits (10s/6s)
❌ Output invalid JSON
❌ Submit after 7 PM PT Wednesday
❌ Use adversarial techniques
❌ Use non-English responses

---

## Our Ask

Please review this briefing and help us decide:

1. **Which option (A/B/C) should we choose?**
2. **Are there any risks or requirements we're missing?**
3. **Is our model selection (Phi-4/Qwen3/NeMo) appropriate?**
4. **Any strategic insights for maximizing our winning chance?**

**Provide your recommendation with reasoning.**

---

## Appendix: Quick Reference

### Competition Format
- Game: "20 Questions" tournament (bracket elimination)
- Agents: Q-Agent (generates questions) + A-Agent (answers questions)
- Rounds: 2 innings × 20 questions = 40 questions per match
- Scoring: (Stumped opponent) + (Correct answers) = Total score

### Technical Stack
- Platform: DigitalOcean AMD (1 MI300X droplet only)
- Image: ROCm 6.4.0
- Container: edaamd/aiac:latest
- Library: Unsloth for fine-tuning
- Models: Phi-4 14B, Qwen3 8B, Mistral NeMo 12B (candidates)

### Budget & Time
- Budget: $300 available, $200-250 target
- Cost: $1.99/hour
- Time: 60 hours until deadline
- Training: 30-50 hours per model
- Testing: 20-30 hours needed

### Decision Options
- **A**: 3 models, $240, 2.5 epochs, 85% win chance
- **B**: 2 models, $240, 3 epochs, 75% win chance ⭐
- **C**: 1 model, $200, 3 epochs × 2 runs, 60% win chance

---

**Thank you for reviewing! We need your input to make the final decision.** 🚀
