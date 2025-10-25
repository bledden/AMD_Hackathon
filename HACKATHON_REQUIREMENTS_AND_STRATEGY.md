# AMD Hackathon: Complete Requirements, Guidelines & Strategy

**Competition**: AMD x Unsloth Synthetic Data Hackathon
**Deadline**: Wednesday, October 29, 2025 @ 7 PM PT
**Platform**: AMD MI300X GPU via DigitalOcean
**Track**: Synthetic-Data Track (Q&A Agents)

---

## Table of Contents

- [Competition Overview](#competition-overview)
- [Prizes](#prizes)
- [Technical Requirements](#technical-requirements)
- [Competition Format](#competition-format)
- [Required JSON Formats](#required-json-formats)
- [Restrictions & Rules](#restrictions--rules)
- [Submission Requirements](#submission-requirements)
- [Infrastructure Setup](#infrastructure-setup)
- [Our Three-Agent Strategy](#our-three-agent-strategy)
- [Model Selection & Rationale](#model-selection--rationale)
- [Timeline & Execution Plan](#timeline--execution-plan)
- [Resources](#resources)

---

## Competition Overview

### Challenge Description

Build AI agents that compete in a Q&A tournament:
- **Q-Agent**: Generates challenging multiple-choice questions
- **A-Agent**: Answers multiple-choice questions accurately

### Competition Format: "20 Questions" Game

- **Two teams compete** in a bracket-style tournament
- **Two innings** per match
- Each inning: 20 questions exchanged between teams
- **Scoring**:
  - Your Q-Agent scores points when opponent's A-Agent fails to answer
  - Your A-Agent scores points when it correctly answers opponent's questions
- **Winner**: Team with highest combined score across both innings

### Example Scoring (from Slide 18)

**Inning 1:**
- Team A Q-Agent: 5/20 (opponent answered 15)
- Team A A-Agent: 15/20 (answered correctly)

**Inning 2:**
- Team A A-Agent: 7/20 (answered correctly)
- Team A Q-Agent: 13/20 (opponent answered 7)

**Team A Score**: 12 (5 + 7)
**Team B Score**: 28 (15 + 13)
**Result**: Team B Wins!

---

## Prizes

### Synthetic-Data Track (Our Track)

1. **1st Place**: $3,000 + 1,200 hrs GPU credits + Trophy
2. **2nd Place**: $1,500 + 600 hrs GPU credits + Trophy
3. **3rd Place**: $900 + 300 hrs GPU credits + Trophy

### OpenEnv RL Track (Alternative Track)

- **Early Submission Bonus** (Sunday 7 PM PT): $500 GPU credits + Ray-Ban Meta (2 teams)
- **Regular Awards**: $2,500, $1,000, $500 in GPU credits

---

## Technical Requirements

### Hardware Requirements

**GPU Instance:**
- **Platform**: DigitalOcean AMD Developer Cloud (amd.digitalocean.com)
- **Image**: **ROCm 6.4.0** (NOT PyTorch + ROCm 7.0.0)
- **Size**: Single MI300X
  - 1 GPU - 192 GB VRAM
  - 20 vCPU - 240 GB RAM
  - Boot disk: 720 GB NVMe
  - Scratch disk: 5 TB NVMe
  - **Cost**: $1.99/hr

**IMPORTANT**: Slide 11 warns - "Only start one instance or you won't have enough credits for submission"
- This refers to the **provided credits for one droplet**
- We're using **our own budget** for 3 droplets (approved approach)

### Software Stack

**Docker Container:**
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
- Unsloth (for fast fine-tuning)
- PyTorch with ROCm support
- Transformers, TRL, PEFT
- Meta's Synthetic Data Kit (encouraged)

**Development Environment:**
- JupyterLab (accessed via port forwarding)
- Work within `AIAC/` folder structure
- All code must be in Python

---

## Competition Format

### AIAC Overview (Slide 17)

**Two Agents Per Team:**
1. **Q-Agent** (Question Generator)
   - Takes a topic as input
   - Generates multiple-choice question with 4 choices (A/B/C/D)
   - Provides correct answer and explanation

2. **A-Agent** (Answerer)
   - Takes a question + 4 choices as input
   - Selects correct answer (A, B, C, or D)
   - Provides reasoning for answer

**Game Flow:**
```
Inning 1:
  Team A Q-Agent generates 20 questions
    → Team B A-Agent attempts to answer
  Team B Q-Agent generates 20 questions
    → Team A A-Agent attempts to answer

Inning 2:
  (Same format, roles may switch)

Final Score = Total correct answers across both innings
```

---

## Required JSON Formats

### Q-Agent Output Format

**Given**: A topic string

**Must produce**:
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
  "explanation": "brief explanation within 100 words for why the answer is correct"
}
```

**Evaluation Process:**
- Organizers extract **only** `Question` and `Choices` keys
- Feed to opponent's A-Agent
- `Topic`, `Question`, `Choices`, and `Answer` verified for correctness by Oracle
- Having `explanation` is a plus but not required if answer is correct

### A-Agent Output Format

**Given**: Question text + 4 choices

**Must produce**:
```json
{
  "answer": "correct choice letter only (A, B, C, or D)",
  "reasoning": "brief reasoning within 100 words for why the answer is correct"
}
```

**Evaluation Process:**
- Organizers extract **only** the `Answer` key
- Compare with correct answer from opponent's question
- Having `reasoning` is a plus but not required if answer is correct

**CRITICAL**: Only responses following these exact formats will be considered!

---

## Restrictions & Rules

### Hard Restrictions (Slide 25)

1. **NO last-minute submission changes**
   - Deadline is **strict**: Wednesday, October 29 @ 7 PM PT
   - Any changes after deadline **may disqualify** submission

2. **RAG is NOT allowed**
   - No Retrieval Augmented Generation techniques
   - No external knowledge bases or search

3. **No adversarial approaches**
   - No jailbreaking
   - No making models hallucinate
   - No prompt injection attacks

4. **English only**
   - Both Q-Agent and A-Agent must use English
   - No multilingual approaches

5. **Token limits**
   - Stay within `max_tokens` limits in `agent.yaml` and `qgen.yaml`
   - Other parameters can be changed

6. **Topics**
   - Questions must pertain to topics in general Q&A
   - Example topics provided: Seating Arrangements, Blood Relations (but not limited to these)

7. **Speed requirements** (CRITICAL!)
   - **Question generation**: Under 10 seconds per question
   - **Answer generation**: Under 6 seconds per answer
   - Requests exceeding limits will be **ignored/not considered**

### Guidelines (Slide 26)

- Follow **rules** and **formats** strictly
- Focus on **quality**: ensure Question, Choices, and Answer correctness
- Give **equal importance** to Q-Agent and A-Agent
- Ensure **four `.py` files** exist in `agents/`:
  - `question_agent.py`
  - `question_model.py`
  - `answer_agent.py`
  - `answer_model.py`
- Do NOT cross the restrictions - all will be enforced
- Ask questions in Discord if unclear

### Ground Rules (Slide 27)

- Be respectful of participants
- Use AMD GPUs (MI300X access provided)
- Use Synthetic Data (Meta's SDK encouraged)
- Maximize GPU memory usage (192 GB is a lot!)
- Have fun and drink chai! ☕

---

## Submission Requirements

### Folder Structure (Slide 21, 24)

**All work must be in `AIAC/` folder**:
```
AIAC/
├── agents/
│   ├── question_agent.py     # Q-Agent entry point
│   ├── question_model.py     # Q-Agent model implementation
│   ├── answer_agent.py       # A-Agent entry point
│   └── answer_model.py       # A-Agent model implementation
├── outputs/
│   ├── questions.json        # Generated questions
│   └── answers.json          # Generated answers
├── assets/
│   ├── sample_answer.json
│   ├── sample_questions.json
│   ├── topics.json
│   └── topics_example.json
└── [your model files]
    ├── model.safetensors
    ├── *.pt, *.pth files
    └── [other checkpoints]
```

### Submission Process

1. **No upload needed**
   - Organizers collect code **directly from your Jupyter server**
   - They will access your droplet

2. **Agent invocation**:
   ```bash
   # Q-Agent generation
   python -m agents.question_agent \
     --output_file "outputs/questions.json" \
     --num_questions 20 \
     --verbose

   # A-Agent generation
   python -m agents.answer_agent \
     --input_file "outputs/filtered_questions.json" \
     --output_file "outputs/answers.json" \
     --verbose
   ```

3. **Model checkpoints**:
   - Ensure model files (`.safetensors`, `.pt`, `.pth`) are loadable
   - Must generate expected output files when invoked

4. **Output files**:
   - `outputs/questions.json` - Q-Agent output
   - `outputs/answers.json` - A-Agent output
   - Must follow exact JSON formats specified

5. **Testing**:
   - Test submission by running commands in "Getting Started" section
   - Verify outputs match required formats

---

## Infrastructure Setup

### Step-by-Step Setup

#### 1. Create Account & Register Team
- Create account: https://devcloud.amd.com/
- Fill team form (one per team): https://forms.gle/RPV7fURLNHDjz2yr9

#### 2. Create GPU Instance
- Navigate to Create → GPU Droplets
- Enter promo code: `amdhack`
- **Image**: ROCm 6.4.0 (under Quick Start)
- **Size**: Single MI300X ($1.99/hr)
- **SSH**: Add your SSH key
- **Hostname**: Descriptive name (e.g., `agent-1-foundation`)

#### 3. Access Instance
```bash
ssh root@<public_ip>
```

#### 4. Start Docker Container
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

⚠️ **IMPORTANT**: Work is NOT persistent if Jupyter/Docker is killed. **Back up regularly!**

#### 5. Port Forward Jupyter (Local Terminal)
```bash
ssh -N -v -L localhost:8889:localhost:8889 root@<public_ip>
```

#### 6. Access Jupyter
Open in browser: `http://127.0.0.1:8889/lab?token=<token>`
(Token shown in Docker container logs)

#### 7. Navigate to AIAC Folder
- Open `AIAC/` folder in JupyterLab
- Read `README.ipynb`
- Review example files in `assets/`

---

## Our Three-Agent Strategy

### Why Three Agents?

**Strategic Advantages:**
1. **Maximize winning probability** - Test 3 different approaches simultaneously
2. **Comprehensive submission** - Show full experimental process (impresses judges)
3. **Hedge against uncertainty** - If one strategy fails, others may succeed
4. **Time efficient** - Parallel execution (4 days) vs sequential (12+ days)
5. **Direct comparison** - Empirically determine best approach

**Budget:**
- 3 droplets × $1.99/hr × ~80 hours = $180-240 typical
- Available budget: $600 ($300 now + $300 after Wednesday)
- Well within budget, plenty of headroom

### Three-Agent Overview

| Agent | Model | Size | Strategy | Role | Cost |
|-------|-------|------|----------|------|------|
| **1: Foundation** | LLaMA 3.1 8B Instruct | 8B | Conservative, balanced | Safety net | $120-160 |
| **2: Challenger** | Qwen 2.5 7B Instruct | 7B | Aggressive, creative | High-reward | $120-160 |
| **3: Hybrid** | Mistral 7B v0.3 | 7B | Specialized, efficient | Domain expert | $80-120 |

**Total**: $180-250 (parallel execution over 4 days)

---

## Model Selection & Rationale

### Agent 1: "Foundation" - LLaMA 3.1 8B Instruct

**Model**: `unsloth/llama-3.1-8b-instruct-bnb-4bit`

#### Why This Model?

**Technical Strengths:**
- ✅ **Latest stable from Meta** (released July 2024)
- ✅ **Instruction-tuned** - pre-trained for following instructions
- ✅ **Proven for Q&A** - widely tested in community
- ✅ **Fast inference** - easily meets 6-10s time limits
- ✅ **Strong MCQ performance** - good at multiple-choice format
- ✅ **Low hallucination rate** - produces accurate answers
- ✅ **8B sweet spot** - balance of speed and capability
- ✅ **Well-documented** - extensive Unsloth support

**Strategic Role:**
- **Conservative approach** - proven, reliable, safe
- **Balanced Q&A** - equal focus on questions and answers
- **Accuracy-first** - prioritize correct answers over creativity
- **Your safety net** - guaranteed solid submission

#### Training Strategy

**Dataset:**
- Curated high-quality MCQ pairs (500-1000 examples)
- Source: Mix of existing datasets + manual curation
- Focus: Clarity, correctness, balanced difficulty
- Format: Multiple-choice Q&A in required JSON format

**Fine-tuning:**
- Approach: Supervised Fine-Tuning (SFT) with LoRA
- LoRA rank: 16 (standard)
- Learning rate: 2e-4
- Training steps: 100-150
- Batch size: 2-4 (fits in 192GB easily)

**Generation Parameters:**
- **Q-Agent** (question generation):
  - Temperature: 0.5-0.6 (moderate creativity)
  - Top-p: 0.9
  - Max tokens: ~200
- **A-Agent** (answering):
  - Temperature: 0.1-0.3 (high precision)
  - Top-p: 0.95
  - Max tokens: ~150

#### Expected Performance

**Strengths:**
- ✅ Answer accuracy: 88-90%
- ✅ Question quality: Clear, well-structured
- ✅ Consistency: Reliable, predictable
- ✅ Speed: Well under time limits (~4-5s question, ~2-3s answer)
- ✅ Hallucination: Very low rate

**Weaknesses:**
- ❌ Question creativity: 7/10 (moderate, not exceptional)
- ❌ Question difficulty: May generate predictable questions
- ❌ Risk: Opponents may answer easily

**Best Scenario for Winning:**
- Tournament heavily weights answer accuracy
- Consistent performance matters more than creativity
- Opponents have weak A-Agents

#### Resource Requirements

- **Training time**: 60-80 hours
- **Cost**: ~$120-160 ($1.99/hr × 60-80 hrs)
- **GPU memory**: ~80GB VRAM (40% of 192GB)
- **Inference speed**: ~4-5s per question, ~2-3s per answer

---

### Agent 2: "Challenger" - Qwen 2.5 7B Instruct

**Model**: `unsloth/qwen2.5-7b-instruct-bnb-4bit`

#### Why This Model?

**Technical Strengths:**
- ✅ **Newest model** (released October 2024 - cutting edge!)
- ✅ **Strong reasoning** - Qwen 2.5 series excels at complex logic
- ✅ **Creative generation** - generates novel, interesting questions
- ✅ **7B for speed** - must meet 6-10s limits (14B would be too slow)
- ✅ **Instruction-tuned** - follows MCQ format well
- ✅ **Multilingual base** - though we'll use English only
- ✅ **Competitive edge** - newer = less tested by competitors
- ✅ **Good at "hard" questions** - can generate challenging MCQs

**Why NOT 14B?**
- ❌ 14B would be slower (~8-12s per generation)
- ❌ Might exceed 10s question generation limit
- ❌ 7B is fast enough while maintaining quality

**Strategic Role:**
- **Aggressive approach** - generate HARD questions to stump opponents
- **High risk, high reward** - newer model, less predictable
- **Creative Q-Agent** - novel question phrasing and topics
- **Strong A-Agent** - handle difficult questions from opponents
- **Potential domination** - if it works well, could sweep tournament

#### Training Strategy

**Dataset:**
- Large-scale synthetic data (2000+ examples)
- Generation: Use GPT-4/Claude API to create challenging MCQs
- Focus: Difficult questions that test reasoning (not just recall)
- Emphasis: Creative phrasing, diverse topics, edge cases
- Format: Multiple-choice Q&A in required JSON format

**Synthetic Data Generation Approach:**
```
Prompt to GPT-4/Claude:
"Generate challenging multiple-choice questions that:
- Test reasoning and understanding, not just memorization
- Have 4 plausible choices (A/B/C/D)
- Include subtle distinctions between choices
- Cover diverse topics (science, history, logic, etc.)
- Are difficult but fair (expert-level knowledge)"
```

**Fine-tuning:**
- Approach: SFT with larger LoRA rank
- LoRA rank: 32 (larger for more capacity)
- Learning rate: 2e-4
- Training steps: 150-200 (more data = more steps)
- Batch size: 2-4

**Generation Parameters:**
- **Q-Agent** (question generation):
  - Temperature: 0.7-0.8 (high creativity)
  - Top-p: 0.9
  - Max tokens: ~250
  - Focus: Generate HARD questions
- **A-Agent** (answering):
  - Temperature: 0.3-0.4 (balance precision and fluency)
  - Top-p: 0.95
  - Max tokens: ~200
  - Focus: Logical reasoning for difficult questions

#### Expected Performance

**Strengths:**
- ✅ Question creativity: 9/10 (most creative)
- ✅ Question difficulty: Generates hardest questions
- ✅ Reasoning capability: Strong on complex questions
- ✅ Novelty: Less predictable, unique approaches
- ✅ Opponent failures: Likely to stump opponents frequently

**Weaknesses:**
- ❌ Answer accuracy: 82-85% (slight trade-off for creativity)
- ❌ Hallucination risk: Higher with aggressive generation
- ❌ Unpredictability: Newer model, less tested
- ❌ Question fairness: Might generate questions that are TOO hard

**Best Scenario for Winning:**
- Tournament rewards question difficulty and creativity
- Opponents have strong A-Agents (need hard questions to beat them)
- Scoring heavily weights opponent failures
- Novel approaches impress judges

#### Resource Requirements

- **Training time**: 60-80 hours
- **Cost**: ~$120-160 ($1.99/hr × 60-80 hrs)
- **GPU memory**: ~90GB VRAM (45% of 192GB)
- **Inference speed**: ~5-7s per question, ~3-4s per answer

---

### Agent 3: "Hybrid" - Mistral 7B v0.3

**Model**: `unsloth/mistral-7b-v0.3-bnb-4bit`

#### Why This Model?

**Technical Strengths:**
- ✅ **Most efficient** - best performance per parameter
- ✅ **Fastest training** - 40-60 hrs vs 60-80 hrs (allows multiple iterations!)
- ✅ **Fast inference** - easily meets time limits (~3-4s question, ~2s answer)
- ✅ **Strong reasoning** - Mistral is known for logical tasks
- ✅ **v0.3 is latest** (released 2024)
- ✅ **Domain specialization** - smaller model = focused expertise
- ✅ **Cost-effective** - cheaper to train = more experimentation possible
- ✅ **Quick iterations** - can try multiple training runs

**Strategic Role:**
- **Specialist approach** - deep expertise in chosen domains
- **Fast iteration** - try different strategies due to speed
- **Efficiency champion** - maximum performance per parameter
- **Domain domination** - if tournament focuses on your domains, you win
- **Backup plan** - if time is short, this trains fastest

#### Training Strategy

**Dataset:**
- Domain-specialized (pick 1-2 focus areas, go deep)
- Potential domains:
  - **Science** (physics, chemistry, biology)
  - **Technology** (programming, AI, systems)
  - **Logic & Reasoning** (puzzles, deduction, math)
  - **History** (world events, causes, effects)
- Deep coverage: 1000+ examples in chosen domains
- Format: Multiple-choice Q&A in required JSON format

**Domain Selection Strategy:**
- Choose domains where:
  - You can generate high-quality training data
  - Tournament is likely to include these topics
  - Mistral's reasoning strengths apply

**Fine-tuning:**
- Approach: Focused SFT with standard LoRA
- LoRA rank: 16 (standard)
- Learning rate: 2e-4
- Training steps: 80-100 (faster convergence)
- Batch size: 4-8 (smaller model = larger batches)
- **Multiple runs possible**: Can iterate 2-3 times due to speed

**Generation Parameters:**
- **Q-Agent** (question generation):
  - Temperature: 0.6 (moderate, domain-appropriate)
  - Top-p: 0.9
  - Max tokens: ~200
  - Focus: Expert-level domain questions
- **A-Agent** (answering):
  - Temperature: 0.2 (high precision for facts)
  - Top-p: 0.95
  - Max tokens: ~150
  - Focus: Accurate domain knowledge

#### Expected Performance

**Strengths:**
- ✅ Domain expertise: 90%+ accuracy in specialized areas
- ✅ Speed: Fastest of all 3 agents
- ✅ Iteration: Can try multiple approaches
- ✅ Cost-effective: Cheapest to train
- ✅ Reasoning: Strong on logic-based questions

**Weaknesses:**
- ❌ Breadth: Struggles outside chosen domains
- ❌ General knowledge: Weaker on diverse topics
- ❌ Question creativity: 8/10 overall (9/10 in domain)
- ❌ Risk: If tournament is broad, disadvantaged

**Best Scenario for Winning:**
- Tournament has domain-specific rounds (e.g., "Science Round")
- Topics cluster around your chosen domains
- Allows domain selection
- Speed and efficiency matter (can iterate quickly)

#### Resource Requirements

- **Training time**: 40-60 hours (FASTEST)
- **Cost**: ~$80-120 ($1.99/hr × 40-60 hrs) (CHEAPEST)
- **GPU memory**: ~60GB VRAM (30% of 192GB)
- **Inference speed**: ~3-4s per question, ~2s per answer (FASTEST)

---

## Model Comparison Matrix

### Performance Comparison

| Metric | Agent 1 (LLaMA 3.1 8B) | Agent 2 (Qwen 2.5 7B) | Agent 3 (Mistral 7B) |
|--------|------------------------|------------------------|----------------------|
| **Release Date** | July 2024 | October 2024 | 2024 |
| **Model Size** | 8B parameters | 7B parameters | 7B parameters |
| **Quantization** | 4-bit (BNB) | 4-bit (BNB) | 4-bit (BNB) |
| **Question Creativity** | 7/10 | 9/10 | 8/10 (9 in domain) |
| **Answer Accuracy** | 88-90% | 82-85% | 85-90% (90+ in domain) |
| **Speed (Question)** | 4-5s | 5-7s | 3-4s |
| **Speed (Answer)** | 2-3s | 3-4s | 2s |
| **Training Time** | 60-80 hrs | 60-80 hrs | 40-60 hrs |
| **Training Cost** | $120-160 | $120-160 | $80-120 |
| **GPU Memory** | ~80GB | ~90GB | ~60GB |
| **Risk Level** | Low | Medium-High | Low-Medium |
| **Hallucination Rate** | Low | Medium | Low |
| **Breadth** | Broad | Very Broad | Narrow (Deep) |
| **Reliability** | 9/10 | 7/10 | 8/10 |

### Strategic Comparison

| Aspect | Agent 1 | Agent 2 | Agent 3 |
|--------|---------|---------|---------|
| **Question Strategy** | Balanced difficulty | Hard questions | Domain-expert questions |
| **Answer Strategy** | Accuracy-first | Creative + accurate | Precision in domain |
| **Dataset Type** | Curated quality | Synthetic scale | Specialized depth |
| **Tournament Style** | Defensive | Aggressive | Specialist |
| **Best Against** | Weak opponents | Strong A-Agents | Domain-focused tournaments |
| **Worst Against** | Creative opponents | Accuracy-focused scoring | Broad general knowledge |

### When Each Agent Wins

**Agent 1 (Foundation) wins when:**
- Tournament heavily weights answer accuracy
- Consistency and reliability matter most
- Opponents have weak A-Agents (our accuracy shines)
- Time constraints favor faster models
- Risk minimization is key

**Agent 2 (Challenger) wins when:**
- Tournament rewards question creativity and difficulty
- Scoring heavily weights opponent failures
- Opponents have strong A-Agents (need hard questions)
- Novelty and innovation impress judges
- Willing to trade some accuracy for creativity

**Agent 3 (Hybrid) wins when:**
- Tournament has domain-specific rounds
- Topics cluster around chosen specialization
- Speed and efficiency are rewarded
- Multiple iterations possible
- Can identify optimal domains early

---

## Are These The Optimal Models?

### ✅ YES - Here's Why:

**1. All models are latest available (2024)**
- LLaMA 3.1: July 2024
- Qwen 2.5: October 2024 (newest!)
- Mistral v0.3: 2024

**2. All meet speed requirements**
- 7-8B size range
- 4-bit quantization
- All well under 10s/6s limits

**3. Diverse strategies**
- Conservative (Agent 1)
- Aggressive (Agent 2)
- Specialist (Agent 3)

**4. Proven for Q&A tasks**
- All have strong instruction-following
- All good at multiple-choice format
- All tested by community

**5. Within budget**
- Total: $180-250
- Available: $600
- Plenty of headroom

### Alternative Models Considered

**Why we DIDN'T choose these:**

| Model | Why Not? |
|-------|----------|
| **Qwen 2.5 14B** | Too slow (8-12s per generation), might exceed 10s limit |
| **LLaMA 3 70B** | Way too slow, won't fit time limits, overkill |
| **Phi-3 Medium** | Not as proven for Q&A, smaller context window |
| **Gemma 2 9B** | Good but not better than our choices |
| **GPT-2/smaller** | Not powerful enough for competitive performance |
| **Yi 34B** | Too large, too slow for time constraints |

### Potential Alternatives (if you want to change)

**If you want MORE creativity:**
- Agent 2: `unsloth/mistral-nemo-12b-instruct-bnb-4bit` (newer, but might be slow)

**If you want MORE speed:**
- Agent 1: `unsloth/llama-3-8b-instruct-bnb-4bit` (slightly faster, equally good)

**If you want NEWEST tech (risky):**
- Agent 2: `unsloth/qwen2.5-14b-instruct-bnb-4bit` (newest, but speed risk)

### Final Recommendation

**KEEP THE CURRENT PLAN:**
- Agent 1: LLaMA 3.1 8B Instruct (safe, proven, reliable)
- Agent 2: Qwen 2.5 7B Instruct (newest, creative, competitive edge)
- Agent 3: Mistral 7B v0.3 (efficient, specialist, fast iteration)

This combination gives you:
- ✅ Latest 2024 models
- ✅ Speed compliance (all 7-8B)
- ✅ Diverse strategies
- ✅ Optimal performance/cost
- ✅ Maximum winning probability

---

## Timeline & Execution Plan

### Saturday, October 26 - Day 1: Parallel Setup

**Morning (4-6 hours wall time):**

1. **Deploy all 3 MI300X droplets** (1 hour)
   - Agent 1: `agent-1-foundation-llama3` (ROCm 6.4.0)
   - Agent 2: `agent-2-challenger-qwen` (ROCm 6.4.0)
   - Agent 3: `agent-3-hybrid-mistral` (ROCm 6.4.0)

2. **SSH into all 3, start Docker containers** (30 min)
   - Run `edaamd/aiac:latest` container on each
   - Port-forward Jupyter on each (8889, 8890, 8891)

3. **Verify AIAC folder structure** (30 min)
   - Check `AIAC/` exists
   - Review `README.ipynb`
   - Examine example files

**Afternoon (4-6 hours wall time):**

4. **Dataset preparation** (parallel on local machine)
   - Agent 1: Curate 500-1000 quality MCQ pairs
   - Agent 2: Generate 2000+ synthetic MCQs (GPT-4/Claude)
   - Agent 3: Build 1000+ domain-specific MCQs

5. **Upload datasets to all 3 agents**

6. **Configure training scripts**
   - Set model names in each agent
   - Set LoRA parameters
   - Set training steps

**End of Day:**
- 3 droplets running with Docker
- Datasets prepared and uploaded
- Ready to start training Sunday morning
- **Cost**: ~$24-36 (3 × 4-6hrs × $1.99)

---

### Sunday, October 27 - Day 2: First Fine-Tunes

**Morning (2 hours prep):**

7. **Final training setup**
   - Verify Unsloth installation on all 3
   - Test loading base models
   - Verify datasets load correctly

**Afternoon/Evening (8-10 hours training):**

8. **Start fine-tuning on all 3 agents** (parallel)
   - Agent 1: 100-150 steps (~8 hours)
   - Agent 2: 150-200 steps (~8 hours)
   - Agent 3: 80-100 steps (~6 hours)

9. **Monitor training** (check every 2 hours)
   - Watch loss curves
   - Check for errors
   - Verify GPU utilization

**Late Evening (2 hours):**

10. **Initial testing**
    - Generate 10 sample questions from each
    - Answer 20 test questions with each
    - Compare quality
    - Identify issues

**End of Day:**
- 3 fine-tuned models (v1)
- Initial performance data
- Notes on what to improve
- **Cost**: ~$60-72 (3 × 10hrs × $1.99)

---

### Monday, October 28 - Day 3: Iteration & Optimization

**Morning (2 hours):**

11. **Analyze Day 2 results**
    - Which agent performed best?
    - What issues did we find?
    - How to improve datasets?

12. **Dataset improvements**
    - Agent 1: Add more diverse examples
    - Agent 2: Fix any hallucination patterns
    - Agent 3: Deepen domain coverage

**Afternoon (8-10 hours training):**

13. **Second fine-tune** (parallel on all 3)
    - Improved datasets
    - Adjusted hyperparameters if needed
    - Longer training if beneficial

14. **Continuous monitoring**

**Evening (2 hours):**

15. **Comprehensive evaluation**
    - Generate 50 questions from each
    - Answer 100 questions with each
    - Calculate accuracy scores
    - Document performance

**End of Day:**
- 3 optimized models (v2)
- Performance metrics
- Clear winner emerging
- **Cost**: ~$48-60 (3 × 8-10hrs × $1.99)

---

### Tuesday, October 29 - Day 4: Final Tuning & Testing

**Morning (4-6 hours):**

16. **Final refinements**
    - Quick iterations on best 1-2 agents
    - Fine-tune generation parameters (temperature, etc.)
    - Optimize for speed (under 10s/6s limits)

17. **Mock tournament**
    - Agent 1 vs Agent 2 (20 questions each way)
    - Agent 2 vs Agent 3 (20 questions each way)
    - Agent 1 vs Agent 3 (20 questions each way)
    - Calculate scores

**Afternoon (4-6 hours):**

18. **Integration with AIAC infrastructure**
    - Ensure models work with their command structure
    - Test: `python -m agents.question_agent`
    - Test: `python -m agents.answer_agent`
    - Verify JSON output formats

19. **Final testing**
    - Generate 20 questions (timing each)
    - Answer 20 questions (timing each)
    - Verify all under time limits
    - Check output files

20. **Documentation**
    - Document each agent's approach
    - Prepare selection rationale
    - Write up comparative analysis

**End of Day:**
- 3 production-ready agents
- Complete performance data
- Mock tournament results
- Integration verified
- **Cost**: ~$48-72 (3 × 8-12hrs × $1.99)

---

### Wednesday, October 29 - Day 5: Evaluation & Submission

**Morning (3-4 hours, NO GPU COST):**

21. **Final evaluation** (on local machine)
    - Compare all 3 agents systematically
    - Score on:
      - Question quality (40%)
      - Answer accuracy (40%)
      - Mock tournament performance (20%)

22. **Select winner**
    - Choose best performer
    - Document why it won
    - Prepare all 3 agents' data

**Afternoon (2-3 hours):**

23. **Prepare submission**
    - Winner's code in `AIAC/agents/`
    - All model checkpoints loaded
    - Test final invocation
    - Verify outputs

24. **Final checks**
    - Run submission commands one last time
    - Verify `outputs/questions.json`
    - Verify `outputs/answers.json`
    - Check JSON formats

25. **Submit by 7 PM PT**
    - Organizers collect from Jupyter server
    - Keep instance running
    - Don't make changes after deadline

**Evening:**
26. **Cleanup**
    - Download all results locally
    - Stop/destroy all 3 droplets
    - Calculate final costs
    - Document learnings

**Cost**: $0 (local evaluation only)

---

### Total Estimated Costs

**By Day:**
- Saturday: $24-36
- Sunday: $60-72
- Monday: $48-60
- Tuesday: $48-72
- Wednesday: $0

**Total**: $180-240 typical, up to $437 maximum

**Budget Status:**
- Available: $600
- Planned: $180-240
- Remaining: $360-420 for dendritic research
- ✅ Well within budget

---

## Resources

### Official Documentation

- **Hackathon Docs**: https://docs.unsloth.ai/new/unsloth-amd-pytorch-synthetic-data-hackathon
- **AMD Developer Cloud**: https://devcloud.amd.com/
- **Team Registration**: https://forms.gle/RPV7fURLNHDjz2yr9

### Technical Resources

- **AMD Blog Post**: https://www.amd.com/en/developer/resources/technical-articles/2025/10x-model-fine-tuning-using-synthetic-data-with-unsloth.html
- **Unsloth Notebooks**: https://github.com/unslothai/notebooks
- **Synthetic Data Kit**: https://github.com/meta-llama/synthetic-data-kit/
- **OpenEnv RL Example**: https://github.com/unslothai/notebooks/blob/main/nb/OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game_BF16.ipynb

### Community Support

- **Discord**: Join #challenge-news for announcements
- **Office Hours**:
  - Saturday 4 PM PT
  - Sunday 4 PM PT
- **Talks**:
  - Saturday 10 AM PT: Opening talk
  - Saturday 10:30 AM PT: Daniel on RL with OpenEnv
  - Sunday 10 AM PT: Sanyam on Synthetic Data best practices

### Our Resources

- **GitHub Repo**: https://github.com/bledden/AMD_Hackathon
- **Strategy Docs**:
  - [README.md](README.md) - Project overview
  - [THREE_AGENT_STRATEGY_EXPLAINED.md](THREE_AGENT_STRATEGY_EXPLAINED.md) - Detailed strategy
  - [PROJECT_PLAN.md](PROJECT_PLAN.md) - Original single-agent plan
  - [QUICKSTART.md](QUICKSTART.md) - Quick start guide
  - [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) - Deployment steps
  - [docs/COMPETITION_STRATEGY.md](docs/COMPETITION_STRATEGY.md) - Tournament tactics

---

## Quick Reference Commands

### Docker & Jupyter

```bash
# SSH to instance
ssh root@<public_ip>

# Start ROCm Jupyter container
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

# Port-forward Jupyter (local terminal)
ssh -N -v -L localhost:8889:localhost:8889 root@<public_ip>

# Access in browser
http://127.0.0.1:8889/lab?token=<token>
```

### Agent Invocation

```bash
# Generate questions (Q-Agent)
python -m agents.question_agent \
  --output_file "outputs/questions.json" \
  --num_questions 20 \
  --verbose

# Generate answers (A-Agent)
python -m agents.answer_agent \
  --input_file "outputs/filtered_questions.json" \
  --output_file "outputs/answers.json" \
  --verbose
```

### Monitoring

```bash
# Check GPU
rocm-smi

# Monitor training
watch -n 30 'rocm-smi; tail -20 /workspace/AIAC/training.log'

# Check outputs
cat outputs/questions.json | jq '.'
cat outputs/answers.json | jq '.'
```

---

## Critical Reminders

### Must Do:
- ✅ Use ROCm 6.4.0 image (NOT PyTorch + ROCm 7.0.0)
- ✅ Work within `AIAC/` folder structure
- ✅ Follow exact JSON formats
- ✅ Meet speed limits (10s questions, 6s answers)
- ✅ Test invocation commands
- ✅ Back up work regularly (Docker is not persistent!)
- ✅ Submit before 7 PM PT Wednesday

### Must NOT Do:
- ❌ Change anything after deadline
- ❌ Use RAG or adversarial approaches
- ❌ Exceed token limits in yaml files
- ❌ Generate non-English output
- ❌ Create questions that take >10s
- ❌ Create answers that take >6s

---

**We're ready to build three exceptional Q&A agents! Let's win this! 🏆**
