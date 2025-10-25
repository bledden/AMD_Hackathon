# Competition Strategy Guide

Strategic guide for maximizing your Q&A agent's tournament performance.

---

## Tournament Format Understanding

### How It Works
1. **Bracket-style tournament** - Single elimination or round-robin
2. **Each round**:
   - Your agent generates a question for opponent
   - Opponent's agent tries to answer your question
   - Opponent generates a question for you
   - Your agent tries to answer opponent's question
3. **Scoring**: Based on:
   - Question difficulty (harder = more points if opponent fails)
   - Answer accuracy (correct = points, incorrect = lose points)
   - Speed (may be timed)

### Winning Strategy
**Generate challenging questions + Answer accurately = Win**

---

## Strategy Types

### 1. Aggressive Strategy
**Goal**: Generate very difficult questions to stump opponents

**Parameters**:
```python
question_temp = 0.8        # Higher creativity
answer_temp = 0.2          # Very precise answers
difficulty = "very challenging"
```

**Pros**:
- High potential points from opponent failures
- Can knock out stronger opponents
- Impressive question quality

**Cons**:
- Risk of generating unfair/unanswerable questions
- May lose points for bad questions
- High variance

**Best for**:
- Later rounds against strong opponents
- Topics you know extremely well
- When ahead in points

### 2. Balanced Strategy (RECOMMENDED)
**Goal**: Solid questions + reliable answers

**Parameters**:
```python
question_temp = 0.7        # Moderate creativity
answer_temp = 0.3          # Accurate answers
difficulty = "challenging"
```

**Pros**:
- Consistent performance
- Good questions that are fair
- Reliable answer accuracy
- Works across all rounds

**Cons**:
- May not maximize points
- Can be beaten by aggressive strategy if lucky

**Best for**:
- Default strategy
- Early/middle rounds
- Unknown opponent strength

### 3. Defensive Strategy
**Goal**: Prioritize answer accuracy over question difficulty

**Parameters**:
```python
question_temp = 0.6        # Conservative questions
answer_temp = 0.1          # Maximum accuracy
difficulty = "moderately challenging"
```

**Pros**:
- Very high answer accuracy
- Minimize point loss
- Safe approach

**Cons**:
- Opponent may answer your questions
- Lower potential points
- May lose to aggressive strategy

**Best for**:
- When opponent is weak
- When you need guaranteed points
- Final rounds with narrow leads

---

## Question Generation Strategy

### What Makes a Good Tournament Question?

#### 1. Difficulty Sweet Spot
```python
# Too easy - opponent will answer
"What is the capital of France?"  # ✗

# Too hard - may be marked unfair
"What is the exact wavelength of the 3rd harmonic
in a quantum well with barrier height 2.3 eV?"  # ✗

# Just right - challenging but answerable
"What is the Heisenberg Uncertainty Principle and
why is it significant in quantum mechanics?"  # ✓
```

#### 2. Question Types That Work

**Conceptual Understanding** (Best):
```
- "Explain the relationship between X and Y"
- "Why does X happen when Y occurs?"
- "How does X work at a fundamental level?"
```

**Application** (Good):
```
- "What would happen if X..."
- "How would you solve X using Y?"
```

**Factual with Reasoning** (Moderate):
```
- "What is X and why is it important?"
- "Who discovered X and what impact did it have?"
```

**Pure Factual** (Risky - too easy):
```
- "What is X?"
- "When did X happen?"
```

#### 3. Question Engineering Prompts

**For your fine-tuning dataset**:
```python
# Include diverse instructions:
instructions = [
    "Generate a challenging question about {topic} that tests deep understanding",
    "Create a question on {topic} that requires both knowledge and reasoning",
    "Write a question about {topic} that goes beyond simple recall",
    "Generate a question that tests {skill} in the context of {topic}",
]
```

**For inference**:
```python
# Be specific in your prompts
prompt = """Generate a challenging question about {topic}
that tests deep understanding rather than simple recall.
The question should be clear, specific, and answerable
with knowledge of the topic."""
```

---

## Answer Strategy

### What Makes a Good Tournament Answer?

#### 1. Accuracy First
```
Wrong answer = Lose points
Correct but incomplete = Partial points
Correct and complete = Full points
```

#### 2. Conciseness vs Completeness

**Too Brief**:
```
Q: What is CRISPR?
A: A gene editing tool.  # ✗ Incomplete
```

**Too Verbose**:
```
Q: What is CRISPR?
A: CRISPR, which stands for Clustered Regularly Interspaced
Short Palindromic Repeats, is a revolutionary gene editing
technology that was discovered in bacteria as part of their
immune system... [500 words]  # ✗ Too long
```

**Just Right**:
```
Q: What is CRISPR?
A: CRISPR is a gene editing technology that uses a guide RNA
to direct the Cas9 enzyme to specific DNA sequences, where it
makes precise cuts. This allows scientists to remove, add, or
modify genetic material with unprecedented accuracy, enabling
applications in medicine, agriculture, and research.  # ✓
```

#### 3. Answer Temperature Tuning

```python
# Very low temp for factual questions
if is_factual(question):
    temperature = 0.1  # Deterministic, accurate

# Moderate temp for conceptual questions
elif is_conceptual(question):
    temperature = 0.3  # Accurate but natural

# Don't use high temp for answers
# High temp = creativity, not accuracy
```

---

## Theme-Specific Strategies

### Science Theme

**Strong Topics**:
- Quantum mechanics (complex, lots of concepts)
- Molecular biology (detailed mechanisms)
- Thermodynamics (counterintuitive principles)

**Good Questions**:
```
- "Explain the wave-particle duality in quantum mechanics"
- "How does CRISPR-Cas9 gene editing work?"
- "What is entropy and why does it always increase?"
```

**Avoid**:
- Simple definitions
- Memorized facts
- Too specialized (unfair)

### History Theme

**Strong Topics**:
- Cause and effect relationships
- Multiple perspectives on events
- Long-term consequences

**Good Questions**:
```
- "What were the primary causes of World War I?"
- "How did the Industrial Revolution change society?"
- "What was the significance of the Treaty of Versailles?"
```

### Technology Theme

**Strong Topics**:
- How technologies work
- Trade-offs and limitations
- Real-world applications

**Good Questions**:
```
- "How does blockchain achieve decentralization?"
- "What are the key differences between machine learning
   and deep learning?"
- "Explain how public key cryptography works"
```

---

## Adaptation During Tournament

### Round 1: Probe Opponent
- Use **balanced strategy**
- Generate moderate difficulty question
- Assess opponent's answer quality
- Note their question style

### Round 2: Adjust
**If opponent is weak**:
- Switch to **defensive strategy**
- Focus on answer accuracy
- Generate medium-hard questions

**If opponent is strong**:
- Consider **aggressive strategy**
- Generate harder questions
- Maintain answer accuracy

### Round 3+: Execute
- Use strategy that's working
- Fine-tune temperature if needed
- Stay consistent

---

## Fine-Tuning for Competition

### Dataset Composition

**Question Generation Examples** (50%):
```python
# Variety of difficulties
{
    "instruction": "Generate a challenging question about quantum mechanics",
    "output": "What is the physical interpretation of wave function collapse?"
}

# Variety of skills
{
    "instruction": "Generate a question that tests understanding of photosynthesis",
    "output": "How do the light-dependent and light-independent reactions
               of photosynthesis work together to produce glucose?"
}
```

**Question Answering Examples** (50%):
```python
# Concise, accurate answers
{
    "instruction": "Answer: What is quantum entanglement?",
    "output": "Quantum entanglement is a phenomenon where two or more
               particles become correlated such that the quantum state of
               one particle cannot be described independently of the others,
               even when separated by large distances. Measuring one particle
               instantaneously affects the other."
}
```

### Training Tips

1. **Balance your dataset**:
   - 50% question generation
   - 50% question answering
   - Equal representation of difficulty levels

2. **Quality over quantity**:
   - 500 high-quality examples > 2000 mediocre examples
   - Manually review generated questions
   - Test answers for accuracy

3. **Theme-specific training**:
   - Focus heavily on your competition theme
   - Include edge cases
   - Cover breadth of subtopics

---

## Real-Time Tactics

### During Your Turn

**Generating Question** (30 seconds):
1. Choose topic within theme
2. Select difficulty based on opponent
3. Generate question
4. Quick sanity check:
   - Is it clear?
   - Is it fair?
   - Would I know the answer?
5. Submit

**Answering Question** (60 seconds):
1. Read question carefully
2. Identify question type (factual, conceptual, application)
3. Set appropriate temperature
4. Generate answer
5. Quick review:
   - Is it accurate?
   - Is it complete?
   - Is it concise?
6. Submit

### Time Management

```
Total time per round: ~2 minutes

Question generation: 20-30 seconds
  ├─ Topic selection: 5s
  ├─ Generation: 10-15s
  └─ Review: 5-10s

Question answering: 30-60 seconds
  ├─ Comprehension: 10s
  ├─ Generation: 15-30s
  └─ Review: 5-20s
```

---

## Common Pitfalls to Avoid

### Question Generation

❌ **Don't**:
- Generate yes/no questions (too simple)
- Make questions too specialized (unfair)
- Include multiple questions (unclear)
- Generate overly long questions (confusing)

✓ **Do**:
- Ask conceptual questions
- Make questions specific but fair
- Keep questions focused
- Aim for 20-100 words

### Answer Generation

❌ **Don't**:
- Ramble or include irrelevant info
- Give partial answers
- Use overly technical jargon
- Hedge excessively ("maybe", "possibly")

✓ **Do**:
- Answer directly and completely
- Use clear, precise language
- Structure answers well
- Be confident

---

## Pre-Competition Preparation

### 1 Week Before
- [ ] Model trained and validated
- [ ] Strategy selected
- [ ] Temperature parameters tuned
- [ ] Batch of sample questions generated
- [ ] Sample answers reviewed for accuracy

### 1 Day Before
- [ ] Final model checkpoint saved
- [ ] Inference pipeline tested
- [ ] GPU instance ready
- [ ] Backup plans in place
- [ ] Good night's sleep!

### Competition Day
- [ ] Instance running
- [ ] Model loaded
- [ ] Quick test run
- [ ] Confident and ready!

---

## Post-Round Analysis

After each round, note:
- **My question quality**: Too easy? Too hard?
- **My answer accuracy**: Correct? Complete?
- **Opponent patterns**: What types of questions?
- **Adjustments needed**: Strategy change?

Use this to adapt for next rounds.

---

## Winning Mindset

### Keys to Success
1. **Preparation**: Train well, test thoroughly
2. **Consistency**: Balanced strategy works
3. **Adaptability**: Read opponents, adjust
4. **Composure**: Stay calm, trust your model
5. **Learning**: Each round teaches something

### Remember
- **Perfect is the enemy of good** - Don't overthink
- **Trust your training** - Model knows more than you think
- **Have fun** - It's a learning experience!

---

## Quick Reference: Strategy Selection

```
Opponent Unknown → Balanced Strategy
Opponent Weak    → Defensive Strategy
Opponent Strong  → Aggressive Strategy
Final Round      → Defensive Strategy (if ahead)
                 → Aggressive Strategy (if behind)
```

---

**Go win that tournament! 🏆**
