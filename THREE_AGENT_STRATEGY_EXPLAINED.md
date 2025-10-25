# AMD Hackathon: Three-Agent Parallel Strategy - Detailed Explanation

**Created**: October 25, 2025
**Competition Deadline**: Wednesday, October 29, 2025
**Strategy**: Deploy 3 different AI agents in parallel, test all, submit best performer

---

## Executive Summary

This document explains the rationale behind deploying **three parallel fine-tuned language models** instead of a single agent for the AMD Hackathon Q&A tournament competition. Each agent uses a different base model and strategy to maximize chances of winning while demonstrating comprehensive experimental methodology.

---

## Competition Context

### AMD Hackathon Challenge
Build Q&A agents that compete in bracket-style tournaments where agents:
1. Generate challenging questions for opponents
2. Answer opponents' questions accurately

**Winning Strategy**: Balance between asking hard questions (to stump opponents) and answering accurately (to score points).

### Tournament Format
- **Bracket-style elimination** or round-robin
- Each round: Your agent asks a question + answers opponent's question
- **Scoring**: Based on question difficulty and answer accuracy
- **Winner**: Advances through tournament brackets

---

## Why Three Agents Instead of One?

### 1. Maximize Winning Probability
- **Single agent**: 100% commitment to one approach
  - If it fails, you lose
  - If strategy is suboptimal, you're stuck

- **Three agents**: Test multiple strategies simultaneously
  - If Agent 1 fails, Agents 2 & 3 may succeed
  - Choose best performer after testing
  - Higher overall success probability

**Statistical Advantage**: 3 independent attempts > 1 attempt

### 2. Comprehensive Submission
Most competitors will submit:
- 1 agent
- Basic documentation
- "This is my agent"

**Your submission**:
- 3 agents with full experimental data
- Systematic comparison methodology
- "We tested 3 approaches, here's why Agent 2 won"
- Shows engineering rigor and scientific method

**Judges love this**: Even if you don't win tournament, comprehensive approach impresses.

### 3. Hedge Against Uncertainty
Tournament format may favor different strategies:
- **If tournament rewards creative questions** → Agent 2 (Challenger) wins
- **If tournament rewards accurate answers** → Agent 1 (Foundation) wins
- **If tournament has domain-specific rounds** → Agent 3 (Hybrid) wins

You don't know which until competition day. **Three agents = prepared for all scenarios.**

### 4. Time Efficiency
**Sequential approach** (one agent at a time):
- Agent 1: 4 days training + testing
- Agent 2: 4 days training + testing
- Agent 3: 4 days training + testing
- **Total**: 12 days (miss deadline!)

**Parallel approach** (all 3 simultaneously):
- All agents: 4 days training + testing concurrently
- **Total**: 4 days (meet deadline!)

**Cloud computing = parallelization advantage**

### 5. Direct Comparison Data
Training 3 agents on same task provides valuable insights:
- Which model architecture is best for Q&A?
- Does model size (7B vs 8B vs 14B) matter?
- Which training strategy (curated vs synthetic data) works better?

**Research value**: Even if you don't win, you've learned something valuable.

---

## The Three Agents: Detailed Breakdown

### Agent 1: "Foundation" - The Safe Baseline

#### Model Choice
**LLaMA 3.1 8B Instruct** (`unsloth/llama-3.1-8b-instruct-bnb-4bit`)

#### Why This Model?
1. **Proven track record**: Meta's latest stable release (2024)
2. **Instruction-tuned**: Already fine-tuned for following instructions
3. **Widely adopted**: Extensive community testing and validation
4. **Strong Q&A performance**: Known to excel at question-answering tasks
5. **Reliable**: Low risk of unexpected behavior or failures

#### Training Strategy
- **Dataset**: Curated high-quality Q&A pairs (500-1000 examples)
  - Sourced from SQuAD 2.0, manual curation
  - Focus on clarity and accuracy
  - Balanced question types (factual, conceptual, application)

- **Fine-tuning approach**: Conservative SFT with LoRA
  - LoRA rank: 16 (standard)
  - Learning rate: 2e-4 (proven)
  - Training steps: 100-150

- **Generation parameters**:
  - Question temperature: 0.5-0.6 (moderate creativity)
  - Answer temperature: 0.1-0.3 (high precision)

#### Expected Performance
**Strengths**:
- ✅ High answer accuracy (85-90%)
- ✅ Consistent, reliable performance
- ✅ Clear, well-structured questions
- ✅ Low hallucination rate

**Weaknesses**:
- ❌ Questions may be predictable
- ❌ Less creative question generation
- ❌ Opponents may answer easily

#### Strategic Role
**Your safety net**: If Agents 2 & 3 fail or underperform, Agent 1 guarantees a solid submission. You can confidently submit this and compete reasonably well.

**Best scenario for winning**: Tournament heavily weighs answer accuracy over question difficulty.

#### Resource Requirements
- **Training time**: 60-80 hours
- **Cost**: ~$120-160
- **GPU memory**: ~80GB VRAM (well within 192GB)

---

### Agent 2: "Challenger" - The High-Risk Play

#### Model Choice
**Qwen 2.5 14B** (`unsloth/qwen2.5-14b-instruct-bnb-4bit`)

#### Why This Model?
1. **Cutting-edge**: Released October 2024, newest technology
2. **Larger capacity**: 14B parameters = more knowledge, better reasoning
3. **Strong reasoning**: Qwen 2.5 series excels at complex problem-solving
4. **Creative generation**: Known for generating novel, interesting outputs
5. **Competitive edge**: Newer model = potential advantage

#### Training Strategy
- **Dataset**: Large-scale synthetic data (2000+ examples)
  - Generated using GPT-4/Claude via API
  - Focus on challenging, creative questions
  - Diverse topics and difficulty levels
  - Emphasis on reasoning over memorization

- **Fine-tuning approach**: Aggressive SFT with larger LoRA
  - LoRA rank: 32 (larger for 14B model)
  - Learning rate: 2e-4 (standard)
  - Training steps: 150-200 (more data = more steps)

- **Generation parameters**:
  - Question temperature: 0.7-0.8 (high creativity)
  - Answer temperature: 0.3-0.4 (balance precision and fluency)

#### Expected Performance
**Strengths**:
- ✅ Generates very challenging questions
- ✅ Creative, novel phrasing
- ✅ Strong reasoning for complex answers
- ✅ May stump opponents frequently

**Weaknesses**:
- ❌ May sacrifice answer accuracy (80-85%)
- ❌ Higher hallucination risk
- ❌ Questions might be too hard (unfair)
- ❌ Less predictable behavior

#### Strategic Role
**Your high-reward gamble**: If tournament rewards creative, difficult questions and Agent 2 maintains decent answer accuracy, this could dominate. High risk because newer model is less tested.

**Best scenario for winning**: Tournament rewards question difficulty and creativity, opponents struggle with hard questions.

#### Resource Requirements
- **Training time**: 60-80 hours
- **Cost**: ~$120-160
- **GPU memory**: ~120GB VRAM (still within 192GB with 4-bit quantization)

---

### Agent 3: "Hybrid" - The Domain Specialist

#### Model Choice
**Mistral 7B v0.3** (`unsloth/mistral-7b-v0.3-bnb-4bit`)

#### Why This Model?
1. **Efficiency**: Smaller model = faster training (2-3x iterations possible)
2. **Strong per-parameter performance**: Mistral is highly efficient
3. **Fast iteration**: Can test multiple configurations
4. **Domain focus**: Smaller model + focused training = deep expertise
5. **Cost-effective**: Cheaper to train, can afford more experimentation

#### Training Strategy
- **Dataset**: Domain-specialized (choose ONE: science, technology, or history)
  - Deep coverage of chosen domain (1000+ examples)
  - Expert-level questions in domain
  - Domain-specific datasets + curated examples
  - Narrow but deep knowledge

- **Fine-tuning approach**: Focused SFT with standard LoRA
  - LoRA rank: 16 (standard)
  - Learning rate: 2e-4 (standard)
  - Training steps: 80-100 (faster convergence)
  - **Multiple training runs possible** due to speed

- **Generation parameters**:
  - Question temperature: 0.6 (moderate, domain-appropriate)
  - Answer temperature: 0.2 (high precision for facts)

#### Expected Performance
**Strengths**:
- ✅ Dominates in chosen domain
- ✅ Expert-level questions in specialty
- ✅ Highly accurate domain answers (90%+)
- ✅ Fast training = can iterate multiple times

**Weaknesses**:
- ❌ Struggles outside domain
- ❌ If tournament is general knowledge, disadvantaged
- ❌ Smaller model = less general capability

#### Strategic Role
**Your specialist**: If tournament has domain-specific rounds or focuses on your chosen area, Agent 3 wins decisively. Also serves as fastest-training backup.

**Best scenario for winning**: Tournament focuses on your domain (e.g., "Science Round", "Technology Round") or allows domain selection.

#### Resource Requirements
- **Training time**: 40-60 hours (fastest!)
- **Cost**: ~$80-120 (cheapest!)
- **GPU memory**: ~60GB VRAM (lowest)

---

## Comparative Analysis

### Model Comparison Matrix

| Aspect | Agent 1 (LLaMA 3.1 8B) | Agent 2 (Qwen 2.5 14B) | Agent 3 (Mistral 7B) |
|--------|------------------------|------------------------|----------------------|
| **Answer Accuracy** | 85-90% (High) | 80-85% (Good) | 90%+ in domain (Excellent) |
| **Question Creativity** | 6/10 (Moderate) | 9/10 (Excellent) | 7/10 in domain (Good) |
| **Reliability** | 9/10 (Excellent) | 7/10 (Good) | 8/10 (Very Good) |
| **Training Speed** | 60-80 hrs (Moderate) | 60-80 hrs (Moderate) | 40-60 hrs (Fast) |
| **Cost** | $120-160 | $120-160 | $80-120 |
| **Risk Level** | Low | High | Medium |
| **Hallucination Rate** | Low | Medium | Low |
| **Breadth** | Broad | Very Broad | Narrow |
| **Depth** | Moderate | Moderate | Deep (in domain) |

### Strategy Comparison

| Strategy Type | Agent 1 | Agent 2 | Agent 3 |
|---------------|---------|---------|---------|
| **Question Strategy** | Balanced difficulty | Hard questions | Domain-expert questions |
| **Answer Strategy** | Accuracy-first | Balance creativity/accuracy | Precision in domain |
| **Dataset Strategy** | Curated quality | Synthetic scale | Specialized depth |
| **Tournament Style** | Defensive | Aggressive | Specialist |

---

## Decision Framework: Selecting the Winner (Wednesday)

### Evaluation Criteria

#### 1. Question Quality (40% weight)
Evaluate 50 generated questions per agent:
- **Difficulty**: Challenging but fair (not impossible)
- **Creativity**: Novel phrasing, interesting angles
- **Clarity**: Unambiguous, well-structured
- **Diversity**: Multiple topics and types
- **Answerability**: Can be answered with knowledge

**Scoring**: 1-10 scale for each criterion, average across 50 questions

#### 2. Answer Accuracy (40% weight)
Test each agent on 100 questions (mix of easy, medium, hard):
- **Correctness**: Factually accurate
- **Completeness**: Covers key points
- **Conciseness**: No unnecessary content
- **Confidence**: Doesn't hallucinate or hedge excessively

**Scoring**: % correct (human evaluation + automated checks)

#### 3. Tournament Strategy (20% weight)
Simulate mock tournament rounds:
- Agent 1 vs Agent 2 (10 rounds)
- Agent 2 vs Agent 3 (10 rounds)
- Agent 1 vs Agent 3 (10 rounds)

**Scoring**: Win/loss ratio across all matchups

### Selection Process

**Step 1**: Calculate weighted scores
```
Score = (Question Quality × 0.4) + (Answer Accuracy × 0.4) + (Tournament Win% × 0.2)
```

**Step 2**: Compare to baseline (no fine-tuning)
- All 3 agents should significantly outperform baseline
- If one doesn't, investigate why

**Step 3**: Select winner
- Highest score = primary submission
- Document why it won
- Include all 3 in submission package

**Step 4**: Prepare submission
- Winner's model and config
- All 3 agents' results
- Comparative analysis
- Selection rationale

---

## Resource Allocation & Budget

### Budget Breakdown

#### Total Available
- Current credits: $300
- Additional credits (after Wednesday): $300
- **Total**: $600

#### AMD Hackathon Allocation
- Agent 1: $120-160 (60-80 hrs)
- Agent 2: $120-160 (60-80 hrs)
- Agent 3: $80-120 (40-60 hrs)
- **Subtotal**: $180-250 typical, up to $437 maximum

#### Remaining for Dendritic Research
- After hackathon: $350-420
- Plus new credits: $300
- **Total for dendritic**: $650-720

✅ **Well within budget, plenty of headroom**

### Time Allocation

#### Saturday (Setup Day)
- Deploy all 3 droplets: 1 hour
- Install dependencies (parallel): 30 min wall time (1.5 hrs GPU time each)
- Dataset preparation (parallel): 2-4 hours wall time
- **Total wall time**: 4-6 hours
- **Total GPU time**: 3 × 4-6 = 12-18 hours
- **Cost**: $24-36

#### Sunday (First Training)
- Fine-tune all 3 (parallel): 8-10 hours wall time
- Testing: 2 hours
- **Total wall time**: 10-12 hours
- **Total GPU time**: 3 × 10 = 30 hours
- **Cost**: $60-72

#### Monday (Iteration)
- Dataset improvements: 2 hours (local)
- Second fine-tune (parallel): 8-10 hours wall time
- **Total wall time**: 10-12 hours
- **Total GPU time**: 3 × 10 = 30 hours
- **Cost**: $48-60

#### Tuesday (Final Tuning)
- Final refinements (parallel): 4-6 hours wall time
- Mock tournaments: 4 hours wall time
- **Total wall time**: 8-10 hours
- **Total GPU time**: 3 × 8 = 24 hours
- **Cost**: $48-72

#### Wednesday (Evaluation)
- Local evaluation: 3-4 hours (no GPU cost)
- Select winner: 1 hour
- Prepare submission: 2 hours
- **Cost**: $0

**Total Estimated Cost**: $180-240

---

## Risk Assessment & Mitigation

### Risk 1: One or More Agents Fail

**Probability**: Low (15-20%)
- Each uses proven technology (Unsloth + MI300X)
- Different models = independent failures

**Impact**: Medium
- Still have 1-2 working agents
- Can submit best of what works

**Mitigation**:
- Test early (Sunday evening)
- Agent 1 is safest (prioritize if time limited)
- Agent 3 trains fastest (backup plan)

### Risk 2: Budget Overrun

**Probability**: Low (10%)
- Budget is conservative ($180-240 typical, $437 max)
- Plenty of headroom ($600 available)

**Impact**: Low
- Still within total budget
- Just leaves less for dendritic research

**Mitigation**:
- Track costs daily
- Can stop Agent 2 or 3 early if needed
- Agent 1 must complete (highest priority)

### Risk 3: Time Runs Short

**Probability**: Medium (30%)
- Unexpected debugging
- Slower training than estimated

**Impact**: Medium
- May not finish all 3 agents

**Mitigation**:
- **Priority order**: Agent 1 > Agent 3 > Agent 2
- Agent 3 trains fastest (40-60 hrs vs 60-80 hrs)
- Can stop Agent 2 early (riskiest anyway)
- Agent 1 completion = guaranteed submission

### Risk 4: All Agents Perform Poorly

**Probability**: Very Low (<5%)
- Unsloth is proven technology
- All 3 approaches tested by community

**Impact**: High
- No competitive submission

**Mitigation**:
- Compare to baseline early
- At least one should beat baseline
- Worst case: Submit best of 3 anyway

---

## Expected Outcomes

### Minimum Success (90% probability)
- All 3 agents train successfully
- All beat baseline (no fine-tuning)
- At least one generates coherent Q&A
- Have working submission

### Good Success (60% probability)
- Clear differentiation between agents
- Winner performs well in tournament
- Advances past first round
- Comprehensive submission impresses

### Excellent Success (30% probability)
- Winner advances multiple rounds
- Demonstrates clear strategy advantage
- Gets positive judge/organizer feedback
- Potentially places in competition

### Outstanding Success (10% probability)
- Wins tournament or places top 3
- Comprehensive submission wins "Best Engineering" or similar
- Gets showcased by AMD/organizers

---

## Why This Approach is Superior

### Compared to Single-Agent Approach

**Traditional Approach** (what most competitors do):
1. Pick one model
2. Train it
3. Hope it works
4. Submit

**Problems**:
- ❌ No backup if approach fails
- ❌ No comparison data
- ❌ Can't optimize strategy
- ❌ Miss better alternative

**Our Three-Agent Approach**:
1. Test 3 different models
2. Train all in parallel
3. Compare systematically
4. Submit best performer

**Advantages**:
- ✅ Backup if one fails
- ✅ Rich comparison data
- ✅ Can pick optimal strategy
- ✅ Find best model empirically
- ✅ Impressive submission regardless of tournament result

### Engineering Rigor

Most submissions: "Here's my agent"

**Your submission**:
```
"We systematically tested 3 approaches:

Agent 1 (LLaMA 3.1 8B): Conservative, high accuracy
- Question quality: 7.2/10
- Answer accuracy: 88%
- Tournament win rate: 60%

Agent 2 (Qwen 2.5 14B): Aggressive, creative
- Question quality: 8.8/10
- Answer accuracy: 82%
- Tournament win rate: 70%  ← Winner!

Agent 3 (Mistral 7B): Domain specialist (science)
- Question quality: 7.8/10 (in science: 9.2/10)
- Answer accuracy: 90% (in science: 95%)
- Tournament win rate: 55%

Winner: Agent 2 (Qwen 2.5 14B)
Rationale: Best balance of creative questions and accurate answers.
          Generated hardest questions while maintaining 82% accuracy.
          Won 70% of mock tournament rounds.

Comprehensive results and methodology attached."
```

**This is what impresses judges.** Even if you don't win tournament, this submission demonstrates:
- Scientific methodology
- Engineering rigor
- Systematic comparison
- Data-driven decision making
- Comprehensive documentation

---

## Conclusion

### Summary of Strategy

**Three agents, three approaches, one winner:**

1. **Agent 1 (Foundation)**: Safe, reliable, guaranteed solid performance
2. **Agent 2 (Challenger)**: Risky, creative, potential to dominate
3. **Agent 3 (Hybrid)**: Specialized, efficient, niche advantage

**Rationale**:
- Maximize winning probability through diversification
- Demonstrate comprehensive experimental methodology
- Hedge against uncertainty in tournament format
- Generate valuable comparative insights
- Create impressive submission regardless of outcome

### Why This Will Work

1. **Technically sound**: All 3 approaches are proven
2. **Resource efficient**: Parallel execution in 4 days
3. **Budget appropriate**: $180-250 well within $600 budget
4. **Risk managed**: Multiple backups and fallback plans
5. **Strategically superior**: Better than single-agent approach

### Success Metrics

**Technical success**: All 3 agents train and perform better than baseline
**Competition success**: Winner advances at least one round in tournament
**Submission success**: Comprehensive documentation impresses judges
**Learning success**: Gain valuable insights into model comparison

---

## Next Steps

### Immediate (Now)
1. ✅ Create 3 MI300X droplets
2. ⬜ Note IP addresses
3. ⬜ Setup SSH config
4. ⬜ Install dependencies on all 3

### Today/Tomorrow
5. ⬜ Prepare 3 different datasets
6. ⬜ Configure each agent with its model
7. ⬜ Verify all setups

### Sunday-Tuesday
8. ⬜ Train all 3 agents in parallel
9. ⬜ Monitor progress daily
10. ⬜ Run comparative tests

### Wednesday
11. ⬜ Evaluate all 3 systematically
12. ⬜ Select winner
13. ⬜ Prepare comprehensive submission
14. ⬜ Submit to competition

---

**This three-agent strategy maximizes your chances of success while demonstrating exceptional engineering methodology. Even if you don't win the tournament, you'll have conducted a thorough, scientific experiment that showcases technical excellence.**

🚀 **Ready to deploy!**
