# Final Execution Plan: Hybrid Ensemble-Enhanced Phi-4

**Strategy**: Ensemble data generation + Phi-4 14B dual-task training + role-switching architecture

**Agreement**: Claude and GPT are fully aligned on this approach.

---

## Executive Summary

### What We're Building

**Single Phi-4 14B model trained to handle both Q-Agent and A-Agent tasks via role-switching**

**Data Source**: Ensemble-generated synthetic Q&A from 3 teacher models + curated real puzzles

**Architecture**: Role tokens `<Q>` and `<A>` signal which task to perform

**Expected Performance**:
- A-Agent: 85-90% accuracy
- Q-Agent: 35-45% stump rate
- Speed: Comfortable under 10s/6s limits
- **Win Probability: 85%**

---

## Phase 1: Ensemble Data Generation

### Goal: Create 300-500 high-quality Q&A pairs

### Teacher Models:
1. **Phi-4 14B** - Best reasoning, logical consistency
2. **Qwen3 8B** (or larger if available) - Creative generation, agent capabilities
3. **Mistral NeMo 12B** - Domain expertise, JSON compliance

### Process:

#### Step 1.1: Generate Questions from Each Teacher (8-10 GPU hours)

**For each teacher model:**
- Load model with Unsloth/HuggingFace
- Prompt to generate 100-150 logic puzzles:
  - 50-75 Seating Arrangement puzzles
  - 50-75 Blood Relations puzzles
- Use temperature 0.8 for creativity
- Ensure output includes:
  - Question text
  - 4 multiple-choice options (A/B/C/D)
  - Correct answer
  - Explanation/reasoning

**Prompt Template** (for each teacher):
```
You are an expert at creating challenging logic puzzles. Generate a difficult multiple-choice puzzle about [TOPIC: seating arrangements / blood relations].

Format your response as JSON:
{
  "question": "The puzzle description...",
  "choices": {
    "A": "First option",
    "B": "Second option",
    "C": "Third option",
    "D": "Fourth option"
  },
  "correct_answer": "B",
  "explanation": "Why B is correct..."
}

Make the puzzle challenging but solvable. Ensure all 4 choices are plausible.
```

**Output**: 3 sets of ~100-150 questions each = 300-450 raw questions

---

#### Step 1.2: Cross-Validation Committee (2-3 GPU hours)

**For each generated question:**

1. **Extract the question and choices** (remove correct answer)
2. **Feed to all 3 teacher models** as answerers
3. **Collect answers** from each teacher

**Answer Prompt Template**:
```
Answer this multiple-choice question. Respond with ONLY the letter (A, B, C, or D) of the correct answer.

Question: [question text]
A) [choice A]
B) [choice B]
C) [choice C]
D) [choice D]

Your answer (letter only):
```

4. **Committee Voting**:
   - If 3/3 agree: ✅ **Keep** (high confidence)
   - If 2/3 agree: ✅ **Keep** (moderate confidence, use majority answer)
   - If 0/3 or 1/3 agree: ❌ **Flag for manual review**

5. **Verification**:
   - Compare committee answer with original teacher's answer
   - If mismatch: flag for review
   - If match: high confidence

**Output**: ~250-350 validated questions (75-80% pass rate expected)

---

#### Step 1.3: Manual Review & Curation (2-4 human hours)

**User (you) will review:**
- All flagged disagreements
- Random sample of 20-30 "passed" questions for sanity check

**Criteria**:
- Question is clear and unambiguous
- Exactly one correct answer
- All 4 choices are plausible
- Follows hackathon rules (no "numeric counting" style)
- Difficulty level appropriate (not trivial, not impossible)

**Decisions**:
- ✅ Keep as-is
- ✏️ Edit for clarity
- ❌ Discard

**Output**: ~200-300 finalized high-quality Q&A pairs

---

#### Step 1.4: Augment with Real Data (1-2 human hours)

**Sources**:
- Online logic puzzle repositories
- Existing Seating Arrangement / Blood Relations datasets
- IndiaBix, GeeksforGeeks, Interview prep sites

**Target**: Add 50-100 real-world puzzles with known correct answers

**Format**: Convert to same JSON structure as synthetic data

**Output**: Final dataset of 250-400 Q&A pairs (synthetic + real)

---

## Phase 2: Dual-Task Dataset Formatting

### Goal: Prepare data for Phi-4 with role-switching

### Format Structure:

We need TWO types of training examples:

#### Q-Agent Examples (Question Generation)

**Input** (User prompt):
```
<Q> Generate a challenging seating arrangement puzzle.
```

**Output** (Assistant response):
```json
{
  "Q-agent": {
    "question": "Five people A, B, C, D, E are sitting in a row...",
    "choices": {
      "A": "A is at position 3",
      "B": "B is at position 2",
      "C": "C is at position 4",
      "D": "D is at position 1"
    },
    "answer": "B",
    "explanation": "From the given conditions..."
  }
}
```

#### A-Agent Examples (Question Answering)

**Input** (User prompt):
```
<A> Question: Five people A, B, C, D, E are sitting in a row. A is not at either end. B is immediately to the right of A. C is at position 5. Where is D?

Choices:
A) Position 1
B) Position 2
C) Position 3
D) Position 4
```

**Output** (Assistant response):
```json
{
  "A-agent": {
    "answer": "A",
    "reasoning": "Since A is not at ends and B is right of A, and C is at 5, D must be at position 1."
  }
}
```

### Dataset Balancing:

**Ratio**: 2:1 A-Agent to Q-Agent examples

**Reasoning**: A-Agent accuracy is more critical (50% of score), so we want the model to prioritize answering correctly over creative question generation.

**From 300 Q&A pairs**:
- Create 300 A-Agent training examples
- Create 150 Q-Agent training examples
- **Total**: 450 training examples

### Conversation Format (Unsloth):

```python
{
  "conversations": [
    {
      "from": "system",
      "value": "You are a logic puzzle expert. When prompted with <Q>, generate challenging puzzles. When prompted with <A>, answer puzzles correctly."
    },
    {
      "from": "user",
      "value": "<A> Question: ... Choices: ..."
    },
    {
      "from": "assistant",
      "value": "{\"A-agent\": {\"answer\": \"B\", \"reasoning\": \"...\"}}"
    }
  ]
}
```

**Output**: `training_data.json` with 450 conversation examples

---

## Phase 3: Phi-4 Dual-Task Fine-Tuning

### Model Configuration:

**Base Model**: `microsoft/phi-4` (14B parameters)
**Quantization**: 4-bit (BNB) for memory efficiency
**Training Method**: LoRA (Low-Rank Adaptation)

### LoRA Configuration:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-4-bnb-4bit",
    max_seq_length=4096,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

# Add special tokens
tokenizer.add_tokens(["<Q>", "<A>"])
model.resize_token_embeddings(len(tokenizer))

# Configure LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # LoRA rank (higher = more capacity)
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)
```

### Training Configuration:

```python
from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs/phi4_dual_task",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch = 8
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=10,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=True,  # Use BF16 on MI300X
    optim="adamw_8bit",
    gradient_checkpointing=True,
    max_grad_norm=1.0,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_seq_length=4096,
    dataset_text_field="text",  # or use formatting_func
    args=training_args,
)

trainer.train()
```

### Training Details:

**Duration**: ~30-40 GPU hours (3 epochs on 450 examples)
**Cost**: ~$60-80
**Monitoring**:
- Training loss (should decrease steadily)
- Validation on 10% held-out set
- Generate sample Q and A every epoch to check quality

### Post-Training:

```python
# Save merged model (LoRA + base)
model.save_pretrained_merged(
    "phi4_dual_task_merged",
    tokenizer,
    save_method="merged_16bit"
)
```

**Output**: Fine-tuned Phi-4 model with dual-task capabilities

---

## Phase 4: Agent Implementation

### File Structure (AIAC/agents/):

```
agents/
├── question_agent.py      # Q-Agent entry point
├── question_model.py      # Q-Agent model loading
├── answer_agent.py        # A-Agent entry point
├── answer_model.py        # A-Agent model loading
└── shared_model.py        # Common model loader (used by both)
```

### Shared Model Loader (shared_model.py):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class DualTaskModel:
    def __init__(self, model_path="./phi4_dual_task_merged"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model.eval()  # Set to evaluation mode

    def generate(self, prompt, max_tokens=512, temperature=0.7, stop_tokens=None):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part (after prompt)
        response = response[len(prompt):].strip()

        return response
```

---

### Q-Agent Implementation (question_agent.py):

```python
import json
import random
from shared_model import DualTaskModel

class QuestionAgent:
    def __init__(self):
        self.model = DualTaskModel()
        self.topics = ["seating arrangement", "blood relations"]

    def generate_question(self):
        # Alternate between topics or pick randomly
        topic = random.choice(self.topics)

        # Prompt with <Q> token
        prompt = f"<Q> Generate a challenging {topic} puzzle.\n\n"

        # Generate with higher temperature for creativity
        response = self.model.generate(
            prompt,
            max_tokens=512,
            temperature=0.8,
            stop_tokens=["}"]  # Stop after JSON closes
        )

        # Parse JSON
        try:
            # Extract JSON from response
            json_str = self.extract_json(response)
            output = json.loads(json_str)

            # Validate format
            assert "Q-agent" in output
            assert "question" in output["Q-agent"]
            assert "choices" in output["Q-agent"]
            assert "answer" in output["Q-agent"]

            return output

        except Exception as e:
            # If parsing fails, try to fix or regenerate
            print(f"Error parsing Q-Agent output: {e}", file=sys.stderr)
            # Could implement retry logic here
            return None

    def extract_json(self, text):
        # Extract JSON from text (find first { to last })
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != 0:
            return text[start:end]
        return text

if __name__ == "__main__":
    agent = QuestionAgent()
    question = agent.generate_question()

    if question:
        # Output ONLY the JSON to stdout
        print(json.dumps(question, indent=2))
```

---

### A-Agent Implementation (answer_agent.py):

```python
import json
import sys
from shared_model import DualTaskModel

class AnswerAgent:
    def __init__(self):
        self.model = DualTaskModel()

    def answer_question(self, question_json):
        # Parse input question
        q_data = question_json.get("Q-agent", question_json)
        question_text = q_data["question"]
        choices = q_data["choices"]

        # Format choices for prompt
        choices_text = "\n".join([f"{k}) {v}" for k, v in choices.items()])

        # Prompt with <A> token
        prompt = f"<A> Question: {question_text}\n\nChoices:\n{choices_text}\n\n"

        # Generate with low temperature for consistency
        response = self.model.generate(
            prompt,
            max_tokens=256,
            temperature=0.2,  # Low temp for deterministic answering
            stop_tokens=["}"]
        )

        # Parse JSON
        try:
            json_str = self.extract_json(response)
            output = json.loads(json_str)

            # Validate format
            assert "A-agent" in output
            assert "answer" in output["A-agent"]

            # Ensure answer is one of the valid choices
            answer = output["A-agent"]["answer"]
            assert answer in choices.keys()

            return output

        except Exception as e:
            print(f"Error parsing A-Agent output: {e}", file=sys.stderr)
            return None

    def extract_json(self, text):
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != 0:
            return text[start:end]
        return text

if __name__ == "__main__":
    # Read question from stdin or file
    question_input = sys.stdin.read()
    question_json = json.loads(question_input)

    agent = AnswerAgent()
    answer = agent.answer_question(question_json)

    if answer:
        # Output ONLY the JSON to stdout
        print(json.dumps(answer, indent=2))
```

---

## Phase 5: Testing & Validation

### Test Suite:

#### 5.1 Format Compliance Tests

```python
def test_format_compliance():
    """Test that outputs match exact JSON format"""

    # Test Q-Agent
    q_agent = QuestionAgent()
    for i in range(20):
        output = q_agent.generate_question()
        assert output is not None, f"Q-Agent failed on iteration {i}"
        assert "Q-agent" in output
        assert all(k in output["Q-agent"] for k in ["question", "choices", "answer"])
        assert len(output["Q-agent"]["choices"]) == 4
        assert output["Q-agent"]["answer"] in ["A", "B", "C", "D"]

    # Test A-Agent
    a_agent = AnswerAgent()
    for q_output in test_questions:
        output = a_agent.answer_question(q_output)
        assert output is not None
        assert "A-agent" in output
        assert "answer" in output["A-agent"]
        assert output["A-agent"]["answer"] in ["A", "B", "C", "D"]
```

#### 5.2 Speed Benchmarks

```python
import time

def test_speed():
    """Ensure generation is under time limits"""

    # Q-Agent speed test (must be <10s)
    q_agent = QuestionAgent()
    times = []
    for i in range(10):
        start = time.time()
        q_agent.generate_question()
        elapsed = time.time() - start
        times.append(elapsed)

    avg_q_time = sum(times) / len(times)
    max_q_time = max(times)

    print(f"Q-Agent avg: {avg_q_time:.2f}s, max: {max_q_time:.2f}s")
    assert max_q_time < 10.0, f"Q-Agent too slow: {max_q_time}s"

    # A-Agent speed test (must be <6s)
    a_agent = AnswerAgent()
    times = []
    for q in test_questions:
        start = time.time()
        a_agent.answer_question(q)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_a_time = sum(times) / len(times)
    max_a_time = max(times)

    print(f"A-Agent avg: {avg_a_time:.2f}s, max: {max_a_time:.2f}s")
    assert max_a_time < 6.0, f"A-Agent too slow: {max_a_time}s"
```

#### 5.3 End-to-End Testing

```python
def test_end_to_end():
    """Test Q-Agent generates answerable questions"""

    q_agent = QuestionAgent()
    a_agent = AnswerAgent()

    correct = 0
    total = 20

    for i in range(total):
        # Generate question
        q_output = q_agent.generate_question()

        # Answer question
        a_output = a_agent.answer_question(q_output)

        # Check if answer matches correct answer
        correct_ans = q_output["Q-agent"]["answer"]
        given_ans = a_output["A-agent"]["answer"]

        if given_ans == correct_ans:
            correct += 1

    accuracy = correct / total
    print(f"Self-consistency accuracy: {accuracy:.1%}")

    # Should be >80% for well-trained model
    assert accuracy > 0.8, f"Low self-consistency: {accuracy:.1%}"
```

---

## Resource Estimation

### GPU Time Breakdown:

| Phase | Task | GPU Hours | Cost |
|-------|------|-----------|------|
| 1.1 | Generate from 3 teachers | 8-10 | $16-20 |
| 1.2 | Cross-validation | 2-3 | $4-6 |
| 1.3-1.4 | Manual curation | 0 | $0 |
| 2 | Dataset formatting | 0 | $0 |
| 3 | Phi-4 fine-tuning (3 epochs) | 30-40 | $60-80 |
| 4 | Agent implementation | 0 | $0 |
| 5 | Testing & validation | 4-6 | $8-12 |
| **TOTAL** | | **44-59 hrs** | **$88-118** |

**Safety Buffer**: Add 20% for retries/debugging = ~$100-140 total

**Budget**: ✅ Well under $300 limit

---

## Timeline (Assuming Start Now)

### Sunday Night (4-6 hours):
- [ ] Create MI300X droplet with ROCm 6.4.0
- [ ] Setup Docker container
- [ ] Install dependencies (Unsloth, PyTorch, Transformers)
- [ ] Verify GPU works
- [ ] Download all 3 teacher models
- [ ] Test generation from one teacher

### Monday (Full Day):
- [ ] Run data generation from all 3 teachers (8-10 hrs GPU)
- [ ] Implement cross-validation script
- [ ] Run committee voting (2-3 hrs GPU)
- [ ] Begin manual review of flagged questions

### Monday Evening/Tuesday Morning:
- [ ] Finish manual curation
- [ ] Gather real puzzle data (50-100 examples)
- [ ] Format dataset for dual-task training
- [ ] Start Phi-4 fine-tuning (runs overnight)

### Tuesday (Full Day):
- [ ] Fine-tuning continues (30-40 hrs total, runs through Tuesday night)
- [ ] Meanwhile: Implement agent scripts (question_agent.py, answer_agent.py)
- [ ] Write test suite

### Tuesday Evening/Wednesday Morning:
- [ ] Fine-tuning completes
- [ ] Load trained model into agents
- [ ] Run full test suite (format, speed, end-to-end)
- [ ] Debug any issues

### Wednesday (Until 7 PM):
- [ ] Final validation runs
- [ ] Package submission files
- [ ] Write brief documentation
- [ ] Submit by 5-6 PM (buffer before 7 PM deadline)

---

## Success Criteria

### Minimum Viable Product:
- ✅ Q-Agent generates valid JSON questions
- ✅ A-Agent generates valid JSON answers
- ✅ Both meet speed limits (<10s / <6s)
- ✅ Format 100% compliant
- ✅ A-Agent achieves >75% accuracy on test set

### Target Performance:
- ✅ A-Agent achieves 85-90% accuracy
- ✅ Q-Agent produces challenging questions (30-40% opponent failure rate estimated)
- ✅ Zero format errors in 100 test runs
- ✅ Average speed well under limits (Q: <5s, A: <3s)

### Stretch Goals:
- 🎯 A-Agent achieves >90% accuracy
- 🎯 Q-Agent produces creative, varied questions
- 🎯 Self-consistency test shows >90% (our A-Agent answers our Q-Agent correctly)

---

## Risk Mitigation

### Risk 1: Role-switching doesn't work well
**Mitigation**: Test early (after 1 epoch). If model confuses tasks, increase role token emphasis or retrain with stronger task separation.

### Risk 2: Ensemble data has quality issues
**Mitigation**: Manual review ALL disagreements. Better to have 200 perfect examples than 400 noisy ones.

### Risk 3: Fine-tuning doesn't converge
**Mitigation**: Monitor loss closely. If not decreasing, adjust learning rate or increase data quality filtering.

### Risk 4: JSON formatting errors
**Mitigation**: Post-processing fallbacks. If parse fails, implement regex extraction or format correction.

### Risk 5: Speed exceeds limits
**Mitigation**: Reduce max_tokens, use greedy decoding (temp=0) for A-Agent, optimize prompt length.

---

## Next Steps

**Immediate Actions:**

1. **User**: Create MI300X droplet with ROCm 6.4.0 (takes 5 min)
2. **Claude**: Prepare data generation scripts
3. **GPT**: Prepare training pipeline code

**Ready to proceed?** Let's start with droplet creation!

---

## Contact Points

- **Claude**: Data generation, validation, documentation
- **GPT**: Training pipeline, agent implementation, testing
- **User**: Manual review, droplet management, final decisions

**Let's build this! 🚀**
