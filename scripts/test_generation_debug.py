"""Debug script to test what models are generating"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Test with Phi-4 first
model_name = "microsoft/phi-4"
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
print("Model loaded")

# Test simple prompt
prompt = """Generate a multiple-choice question about seating arrangements.

Respond ONLY with valid JSON:
{
  "question": "...",
  "choices": {
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  },
  "correct_answer": "A",
  "explanation": "..."
}"""

# Apply chat template
messages = [{"role": "user", "content": prompt}]

if hasattr(tokenizer, 'apply_chat_template'):
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"Using chat template")
else:
    formatted = prompt
    print(f"No chat template")

inputs = tokenizer(formatted, return_tensors="pt").to("cuda")

print("Generating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.8,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print("\n===== RAW RESPONSE =====")
print(response)
print("\n===== END RESPONSE =====")

# Try to find JSON
import json
start = response.find("{")
end = response.rfind("}") + 1

if start != -1 and end != 0:
    json_str = response[start:end]
    print(f"\n===== EXTRACTED JSON STRING =====")
    print(json_str)

    # Try to parse
    try:
        data = json.loads(json_str)
        print(f"\n✅ Successfully parsed JSON!")
        print(json.dumps(data, indent=2))
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON parse error: {e}")
        print("First 100 chars of JSON string:", json_str[:100])
else:
    print(f"\n❌ No JSON found in response")