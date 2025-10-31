# AMD Hackathon - 3 Agent Deployment Checklist

**Date**: October 27, 2025 (Sunday)
**Deadline**: October 29, 2025, 7 PM PT (Wednesday)
**Timeline**: 4-day sprint (Sat-Wed)

---

## ✅ Pre-Deployment (COMPLETED)

- ✅ GitHub repo created: https://github.com/bledden/AMD_Hackathon
- ✅ All documentation pushed to GitHub
- ✅ SSH key generated and added to DigitalOcean
- ✅ Agent 1 droplet destroyed (wrong ROCm version)
- ✅ Confirmed ROCm 6.4.0 requirement with organizers

---

## 🚀 Phase 1: Create 3 MI300X Droplets (NOW)

### Droplet Specifications
- **Image**: ROCm 6.4.0 (NOT PyTorch 2.6.0 + ROCm 7.0.0)
- **GPU**: Single MI300X (192GB VRAM, 20 vCPU, 240GB RAM)
- **Cost**: $1.99/hr per droplet
- **Region**: Any available
- **SSH Key**: Your generated key (already added)

### Droplet Names
```
agent-1-foundation-llama3
agent-2-challenger-qwen
agent-3-hybrid-mistral
```

### Create Each Droplet
1. Go to: https://amd.digitalocean.com/
2. Click "Create" → "Droplets"
3. **Choose image**:
   - Click "Marketplace"
   - Search for "ROCm 6.4.0"
   - Select the ROCm 6.4.0 image (NOT the PyTorch + ROCm 7.0.0)
4. **Choose plan**:
   - GPU Droplets
   - Single MI300X ($1.99/hr)
5. **Choose datacenter**: Any available
6. **Authentication**: Select your SSH key
7. **Hostname**: Use agent name (e.g., `agent-1-foundation-llama3`)
8. Click "Create Droplet"
9. **Wait 2-3 minutes** for droplet to start
10. **Copy IP address** and save it

**Repeat 3 times** for all 3 agents.

---

## 📝 Phase 2: Configure SSH (After IPs received)

### Update ~/.ssh/config

Add the following (replace IPs with actual values):

```bash
# AMD Hackathon Agents
Host agent-1
    HostName [AGENT-1-IP]
    User root
    ServerAliveInterval 60

Host agent-2
    HostName [AGENT-2-IP]
    User root
    ServerAliveInterval 60

Host agent-3
    HostName [AGENT-3-IP]
    User root
    ServerAliveInterval 60
```

### Test SSH Access
```bash
ssh agent-1
# Should connect without password
exit

ssh agent-2
exit

ssh agent-3
exit
```

---

## 🔧 Phase 3: Setup Each Agent (Parallel)

**Do this for EACH agent** (open 3 terminal windows):

### Agent 1 Setup
```bash
# Terminal 1
ssh agent-1

# Verify ROCm
rocm-smi
# Should show MI300X GPU

# Clone repo
git clone https://github.com/bledden/AMD_Hackathon.git
cd AMD_Hackathon

# Install dependencies
bash setup/install_dependencies.sh
# This takes ~10-15 minutes

# Verify installation
python3 setup/verify_gpu.py
# Should show all checks passed

# Start Docker container
docker pull edaamd/aiac:latest
docker run -d --name aiac-agent1 \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video --cap-add=SYS_PTRACE \
  -p 8888:8888 \
  -v $(pwd):/workspace \
  edaamd/aiac:latest

# Get Jupyter token
docker logs aiac-agent1 | grep token
# Save this token!
```

### Agent 2 Setup
```bash
# Terminal 2
ssh agent-2

# Same steps as Agent 1
rocm-smi
git clone https://github.com/bledden/AMD_Hackathon.git
cd AMD_Hackathon
bash setup/install_dependencies.sh
python3 setup/verify_gpu.py

docker pull edaamd/aiac:latest
docker run -d --name aiac-agent2 \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video --cap-add=SYS_PTRACE \
  -p 8888:8888 \
  -v $(pwd):/workspace \
  edaamd/aiac:latest

docker logs aiac-agent2 | grep token
```

### Agent 3 Setup
```bash
# Terminal 3
ssh agent-3

# Same steps as Agent 1 and 2
rocm-smi
git clone https://github.com/bledden/AMD_Hackathon.git
cd AMD_Hackathon
bash setup/install_dependencies.sh
python3 setup/verify_gpu.py

docker pull edaamd/aiac:latest
docker run -d --name aiac-agent3 \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video --cap-add=SYS_PTRACE \
  -p 8888:8888 \
  -v $(pwd):/workspace \
  edaamd/aiac:latest

docker logs aiac-agent3 | grep token
```

---

## 🌐 Phase 4: Access JupyterLab (Optional)

**If you want to use JupyterLab interface:**

### Forward ports from local machine
```bash
# Terminal 1 (Agent 1)
ssh -L 8881:localhost:8888 agent-1

# Terminal 2 (Agent 2)
ssh -L 8882:localhost:8888 agent-2

# Terminal 3 (Agent 3)
ssh -L 8883:localhost:8888 agent-3
```

### Access in browser
- Agent 1: http://localhost:8881 (use token from Agent 1)
- Agent 2: http://localhost:8882 (use token from Agent 2)
- Agent 3: http://localhost:8883 (use token from Agent 3)

---

## 📊 Phase 5: Prepare Datasets (Parallel)

### Agent 1: Foundation (Curated Quality)
```bash
ssh agent-1
cd ~/AMD_Hackathon

# Create curated dataset focusing on accuracy
python3 data/prepare_dataset.py \
  --strategy manual \
  --n-samples 500 \
  --output data/processed/agent1_train.json

# Validate format
python3 -c "import json; data=json.load(open('data/processed/agent1_train.json')); print(f'Samples: {len(data)}'); print('Sample:', json.dumps(data[0], indent=2))"
```

### Agent 2: Challenger (Synthetic Creative)
```bash
ssh agent-2
cd ~/AMD_Hackathon

# Create large synthetic dataset
python3 data/prepare_dataset.py \
  --strategy hybrid \
  --n-samples 1000 \
  --n-synthetic 500 \
  --output data/processed/agent2_train.json

# Validate format
python3 -c "import json; data=json.load(open('data/processed/agent2_train.json')); print(f'Samples: {len(data)}'); print('Sample:', json.dumps(data[0], indent=2))"
```

### Agent 3: Hybrid (Domain Specialist)
```bash
ssh agent-3
cd ~/AMD_Hackathon

# Create domain-focused dataset (science/tech)
python3 data/prepare_dataset.py \
  --strategy existing \
  --theme technology \
  --n-samples 750 \
  --output data/processed/agent3_train.json

# Validate format
python3 -c "import json; data=json.load(open('data/processed/agent3_train.json')); print(f'Samples: {len(data)}'); print('Sample:', json.dumps(data[0], indent=2))"
```

---

## 🏋️ Phase 6: Configure Training (Each Agent)

### Agent 1: LLaMA 3.1 8B Configuration
```bash
ssh agent-1
cd ~/AMD_Hackathon

# Edit training config
cat > training/configs/agent1_config.py << 'EOF'
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name = "unsloth/llama-3.1-8b-instruct-bnb-4bit"
    max_seq_length = 2048
    load_in_4bit = True

@dataclass
class LoRAConfig:
    r = 16
    lora_alpha = 16
    lora_dropout = 0.05
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

@dataclass
class TrainingConfig:
    output_dir = "training/outputs/agent1_llama3"
    num_train_epochs = 3
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 2
    learning_rate = 2e-4
    warmup_steps = 10
    logging_steps = 10
    save_strategy = "epoch"
    fp16 = True
    optim = "adamw_8bit"
EOF
```

### Agent 2: Qwen 2.5 7B Configuration
```bash
ssh agent-2
cd ~/AMD_Hackathon

# Edit training config
cat > training/configs/agent2_config.py << 'EOF'
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name = "unsloth/qwen2.5-7b-instruct-bnb-4bit"
    max_seq_length = 2048
    load_in_4bit = True

@dataclass
class LoRAConfig:
    r = 24
    lora_alpha = 24
    lora_dropout = 0.05
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

@dataclass
class TrainingConfig:
    output_dir = "training/outputs/agent2_qwen"
    num_train_epochs = 3
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 2
    learning_rate = 2e-4
    warmup_steps = 10
    logging_steps = 10
    save_strategy = "epoch"
    fp16 = True
    optim = "adamw_8bit"
EOF
```

### Agent 3: Mistral 7B Configuration
```bash
ssh agent-3
cd ~/AMD_Hackathon

# Edit training config
cat > training/configs/agent3_config.py << 'EOF'
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name = "unsloth/mistral-7b-v0.3-bnb-4bit"
    max_seq_length = 2048
    load_in_4bit = True

@dataclass
class LoRAConfig:
    r = 16
    lora_alpha = 16
    lora_dropout = 0.05
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

@dataclass
class TrainingConfig:
    output_dir = "training/outputs/agent3_mistral"
    num_train_epochs = 3
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 2
    learning_rate = 2e-4
    warmup_steps = 10
    logging_steps = 10
    save_strategy = "epoch"
    fp16 = True
    optim = "adamw_8bit"
EOF
```

---

## 🚀 Phase 7: Start Training (Parallel)

### Agent 1 Training
```bash
ssh agent-1
cd ~/AMD_Hackathon

# Start training (runs in background)
nohup python3 training/scripts/train.py \
  --config agent1_config \
  --dataset data/processed/agent1_train.json \
  > training/logs/agent1_training.log 2>&1 &

# Save process ID
echo $! > training/logs/agent1_pid.txt

# Monitor progress
tail -f training/logs/agent1_training.log
```

### Agent 2 Training
```bash
ssh agent-2
cd ~/AMD_Hackathon

nohup python3 training/scripts/train.py \
  --config agent2_config \
  --dataset data/processed/agent2_train.json \
  > training/logs/agent2_training.log 2>&1 &

echo $! > training/logs/agent2_pid.txt
tail -f training/logs/agent2_training.log
```

### Agent 3 Training
```bash
ssh agent-3
cd ~/AMD_Hackathon

nohup python3 training/scripts/train.py \
  --config agent3_config \
  --dataset data/processed/agent3_train.json \
  > training/logs/agent3_training.log 2>&1 &

echo $! > training/logs/agent3_pid.txt
tail -f training/logs/agent3_training.log
```

---

## 📊 Phase 8: Monitor All Agents

### Quick Status Check (Run from local machine)
```bash
# Check all agents at once
for i in 1 2 3; do
  echo "=== Agent $i ==="
  ssh agent-$i "rocm-smi | grep GPU; ps aux | grep train | grep -v grep | head -1"
  echo ""
done
```

### Check Training Logs
```bash
# Agent 1
ssh agent-1 "tail -30 ~/AMD_Hackathon/training/logs/agent1_training.log"

# Agent 2
ssh agent-2 "tail -30 ~/AMD_Hackathon/training/logs/agent2_training.log"

# Agent 3
ssh agent-3 "tail -30 ~/AMD_Hackathon/training/logs/agent3_training.log"
```

### Calculate Costs
```bash
# Check uptime and calculate cost
for i in 1 2 3; do
  echo "=== Agent $i ==="
  ssh agent-$i "uptime -p"
  # Multiply hours by $1.99
done
```

---

## 🎯 Phase 9: Evaluation & Testing (Tuesday Evening)

### Test Each Agent
```bash
# Agent 1
ssh agent-1
cd ~/AMD_Hackathon
python3 inference/generate_qa.py \
  --model training/outputs/agent1_llama3 \
  --mode tournament \
  --n-questions 10

# Agent 2
ssh agent-2
cd ~/AMD_Hackathon
python3 inference/generate_qa.py \
  --model training/outputs/agent2_qwen \
  --mode tournament \
  --n-questions 10

# Agent 3
ssh agent-3
cd ~/AMD_Hackathon
python3 inference/generate_qa.py \
  --model training/outputs/agent3_mistral \
  --mode tournament \
  --n-questions 10
```

### Run Mock Tournament
```bash
# Create tournament script
cat > test_tournament.sh << 'EOF'
#!/bin/bash

echo "=== Mock Tournament: Agent 1 vs Agent 2 ==="
python3 inference/tournament_agent.py --agent1 agent1 --agent2 agent2 --rounds 5

echo "=== Mock Tournament: Agent 1 vs Agent 3 ==="
python3 inference/tournament_agent.py --agent1 agent1 --agent2 agent3 --rounds 5

echo "=== Mock Tournament: Agent 2 vs Agent 3 ==="
python3 inference/tournament_agent.py --agent1 agent2 --agent2 agent3 --rounds 5
EOF

chmod +x test_tournament.sh
./test_tournament.sh
```

---

## 🏆 Phase 10: Select Winner & Submit (Wednesday)

### Compare All 3 Agents
1. **Question Quality** (40% weight)
   - Difficulty level
   - Clarity
   - Variety

2. **Answer Accuracy** (40% weight)
   - Correct answers
   - Reasoning quality
   - Confidence

3. **Tournament Performance** (20% weight)
   - Win rate
   - Strategy effectiveness
   - Time efficiency

### Create Submission
```bash
# Package winner (e.g., Agent 1)
ssh agent-1
cd ~/AMD_Hackathon

# Export model
python3 -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained('training/outputs/agent1_llama3')
model.save_pretrained('submission/agent1_final')
tokenizer.save_pretrained('submission/agent1_final')
"

# Download to local
scp -r agent-1:~/AMD_Hackathon/submission/agent1_final ~/Desktop/

# Submit via hackathon platform
```

---

## 🛑 Phase 11: Cleanup (After Wednesday)

### Stop All Droplets
```bash
# Via DigitalOcean web interface:
# Droplets → agent-1 → Power → Power Off
# Droplets → agent-2 → Power → Power Off
# Droplets → agent-3 → Power → Power Off

# Or via API:
doctl compute droplet delete agent-1-foundation-llama3 -f
doctl compute droplet delete agent-2-challenger-qwen -f
doctl compute droplet delete agent-3-hybrid-mistral -f
```

### Calculate Final Costs
```bash
# Total hours × $1.99 per droplet
# Expected: $180-250 total
```

---

## 📋 Daily Checklist

### Saturday (Day 1)
- [ ] Create 3 MI300X droplets with ROCm 6.4.0
- [ ] Setup SSH config
- [ ] Install dependencies on all 3
- [ ] Start Docker containers
- [ ] Prepare datasets
- [ ] Configure training scripts
- [ ] **Cost**: ~$24-36

### Sunday (Day 2)
- [ ] Start training on all 3 agents (morning)
- [ ] Monitor progress (every 4-6 hours)
- [ ] Check for errors
- [ ] Verify GPU utilization
- [ ] **Cost**: ~$60-72

### Monday (Day 3)
- [ ] Continue monitoring training
- [ ] Check intermediate checkpoints
- [ ] Adjust if needed (restart failing agents)
- [ ] **Cost**: ~$48-60

### Tuesday (Day 4)
- [ ] Training should complete by evening
- [ ] Run evaluation scripts
- [ ] Test all 3 agents
- [ ] Mock tournament battles
- [ ] Select winner
- [ ] **Cost**: ~$48-72

### Wednesday (Day 5)
- [ ] Final testing (morning)
- [ ] Package submission
- [ ] Submit by 7 PM PT
- [ ] Stop all droplets
- [ ] **Cost**: ~$0-12

---

## ⚠️ Emergency Procedures

### If Training Fails
```bash
# Check logs
ssh agent-X "tail -100 ~/AMD_Hackathon/training/logs/agentX_training.log"

# Check GPU
ssh agent-X "rocm-smi"

# Restart from checkpoint
ssh agent-X "cd ~/AMD_Hackathon && python3 training/scripts/resume_training.py --checkpoint training/outputs/agentX/checkpoint-XXX"
```

### If Running Out of Time
**Priority**: Ensure Agent 1 completes (Foundation model is safest bet)
- If Tuesday evening and Agent 3 won't finish → stop it
- Submit Agent 1 or Agent 2 instead

### If Over Budget
- Stop least promising agent early
- Focus resources on best performer

---

## 🎯 Success Metrics

### Minimum Success (MUST HAVE)
- [ ] At least 1 agent fully trained
- [ ] Agent passes validation tests
- [ ] Submission ready by Wednesday 7 PM PT
- [ ] Cost < $250

### Ideal Success (NICE TO HAVE)
- [ ] All 3 agents complete training
- [ ] Comprehensive comparison data
- [ ] Multiple submission options
- [ ] Cost < $200

---

## 📞 Resources

- **Hackathon Discord**: [Link from organizers]
- **Unsloth Docs**: https://github.com/unslothai/unsloth
- **ROCm Docs**: https://rocm.docs.amd.com/
- **Our Documentation**: /Users/bledden/Documents/AMD_Hackathon/

---

**Ready to start? Begin with Phase 1: Create 3 MI300X Droplets!** 🚀
