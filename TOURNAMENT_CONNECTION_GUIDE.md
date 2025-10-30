# AMD Hackathon - Tournament Agent Connection Guide

## Quick Status
- **Answer Agent**: Qwen2.5-7B-Instruct (92% accuracy)
- **Question Agent**: Pre-generated question pool selector
- **Server**: 129.212.186.194
- **Deployment**: AIAC/agents directory

---

## Connection Methods

### Method 1: Tournament Server (HTTP API)
The tournament server is running on the remote server with HTTP endpoints.

**Check if running:**
```bash
ssh amd-hackathon "docker exec rocm curl http://localhost:5000/health"
```

**Expected response:**
```json
{"accuracy":"92%","model":"Qwen2.5-7B-Instruct","status":"ready"}
```

**Answer a question:**
```bash
curl -X POST http://129.212.186.194:5000/answer_question \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the capital of France?",
    "choices": {
      "A": "London",
      "B": "Paris", 
      "C": "Berlin",
      "D": "Madrid"
    }
  }'
```

**If port 5000 is not exposed externally, you need to:**
```bash
# SSH tunnel to access from your machine
ssh -L 5000:localhost:5000 amd-hackathon

# Then access at: http://localhost:5000/
```

---

### Method 2: AIAC Agents (Module-based)
The agents are deployed in `/workspace/AIAC/agents/` in the Docker container.

**Test Answer Agent:**
```bash
ssh amd-hackathon "docker exec rocm bash -c 'echo \"{\\\"question\\\": \\\"What is 2+2?\\\", \\\"choices\\\": {\\\"A\\\": \\\"3\\\", \\\"B\\\": \\\"4\\\", \\\"C\\\": \\\"5\\\", \\\"D\\\": \\\"6\\\"}}\" | python3 -c \"import sys; sys.path.insert(0, \\\"/workspace/AIAC\\\"); from agents.answer_model import answer_question; import json; data = json.loads(input()); print(answer_question(data))\"'"
```

**Or run as module:**
```bash
# Copy test question to server
ssh amd-hackathon "docker exec rocm bash -c 'mkdir -p /workspace/AIAC/outputs && echo \"[{\\\"id\\\": 1, \\\"question\\\": \\\"What is the capital of France?\\\", \\\"choices\\\": {\\\"A\\\": \\\"London\\\", \\\"B\\\": \\\"Paris\\\", \\\"C\\\": \\\"Berlin\\\", \\\"D\\\": \\\"Madrid\\\"}}]\" > /workspace/AIAC/outputs/questions.json'"

# Run answer agent
ssh amd-hackathon "docker exec rocm bash -c 'cd /workspace && python3 -m AIAC.agents.answer_model'"

# Check results
ssh amd-hackathon "docker exec rocm cat /workspace/AIAC/outputs/answers.json"
```

---

### Method 3: Direct Python Import
If you're inside the Docker container:

```python
import sys
sys.path.insert(0, '/workspace')

from AIAC.agents.answer_model import answer_question

question_data = {
    "question": "What is the capital of France?",
    "choices": {
        "A": "London",
        "B": "Paris",
        "C": "Berlin",
        "D": "Madrid"
    }
}

answer = answer_question(question_data)
print(f"Answer: {answer}")
```

---

## Checking Deployment Status

**1. Verify files are deployed:**
```bash
ssh amd-hackathon "docker exec rocm ls -la /workspace/AIAC/agents/"
```

**Should show:**
- `answer_model.py` (Qwen2.5-7B with timeout)
- `question_model.py` (Question pool selector)
- `__init__.py`

**2. Check if model is downloaded:**
```bash
ssh amd-hackathon "docker exec rocm ls -la /home/rocm-user/AMD_Hackathon/models/qwen2.5_7b_instruct/"
```

**3. Check tournament server logs:**
```bash
ssh amd-hackathon "docker exec rocm tail -50 /home/rocm-user/AMD_Hackathon/logs/tournament_server.log"
```

---

## Troubleshooting

**If tournament server isn't running:**
```bash
# Start it
ssh amd-hackathon "docker exec -d rocm bash -c 'cd /home/rocm-user/AMD_Hackathon && python3 tournament_server.py > logs/tournament_server.log 2>&1'"

# Check it started
ssh amd-hackathon "docker exec rocm curl http://localhost:5000/health"
```

**If Qwen model isn't found:**
```bash
# Check if model exists
ssh amd-hackathon "docker exec rocm ls /home/rocm-user/AMD_Hackathon/models/qwen2.5_7b_instruct/config.json"

# If not, update path in answer_model.py to point to correct location
```

**If AIAC agents aren't deployed:**
```bash
# Copy from local to server
scp -r AIAC/agents amd-hackathon:/tmp/
ssh amd-hackathon "docker exec rocm mkdir -p /workspace/AIAC"
ssh amd-hackathon "docker cp /tmp/agents rocm:/workspace/AIAC/"
```

---

## Performance Specs

**Answer Agent (Qwen2.5-7B):**
- Validation Accuracy: 92% (46/50 questions)
- Average Response Time: 0.228s
- Max Response Time: <6s (guaranteed with timeout)
- Timeout Fallback: Returns "B" if generation exceeds 5.5s

**Question Agent:**
- Response Time: <1s (reads from pre-generated pool)
- Pool Size: Depends on `/workspace/question_pool.json`

---

## Server Details

- **IP**: 129.212.186.194
- **SSH**: `ssh amd-hackathon` (requires passphrase)
- **Docker Container**: `rocm`
- **GPU**: AMD MI300X (192GB VRAM)
- **Models Directory**: `/home/rocm-user/AMD_Hackathon/models/`
- **AIAC Directory**: `/workspace/AIAC/`

