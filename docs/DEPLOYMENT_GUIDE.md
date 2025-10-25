# Deployment Guide - AMD MI300X Q&A Agent

Complete guide for deploying and running your Q&A agent on AMD MI300X GPU.

---

## Pre-Deployment Checklist

### Local Setup (Before MI300X)
- [ ] Git repository initialized and pushed to GitHub
- [ ] All scripts tested locally (without GPU)
- [ ] Dataset prepared and validated
- [ ] Training configuration reviewed
- [ ] Scripts are executable (`chmod +x`)

### DigitalOcean Account
- [ ] Access to amd.digitalocean.com
- [ ] Credits available ($300 developer credits)
- [ ] SSH key generated and added
- [ ] Budget alert set ($50 threshold)

---

## Step 1: Deploy MI300X Instance

### 1.1 Log into DigitalOcean
```bash
# Navigate to: https://amd.digitalocean.com
# Log in with your credentials
```

### 1.2 Create Droplet
1. Click **"Create"** → **"Droplets"**
2. **Image**: PyTorch 2.6.0, ROCm 7.0.0
3. **Plan**: Single MI300X
   - 1 GPU - 192 GB VRAM
   - 20 vCPU - 240 GB RAM
   - Cost: **$1.99/hr**
4. **Datacenter**: Choose closest region
5. **Authentication**: Add your SSH key
6. **Hostname**: amd-hackathon-qa-agent
7. Click **"Create Droplet"**

### 1.3 Note Instance Details
```bash
# Save these details:
IP Address: ___.___.___.___
SSH Key: ~/.ssh/id_rsa
Username: root (or ubuntu)
```

---

## Step 2: Connect and Setup Environment

### 2.1 SSH into Instance
```bash
# Connect to your instance
ssh root@YOUR_IP_ADDRESS

# Or if using different user:
ssh ubuntu@YOUR_IP_ADDRESS
```

### 2.2 Verify GPU
```bash
# Check AMD GPU
rocm-smi

# Should show MI300X with 192GB memory
```

### 2.3 Clone Repository
```bash
# Clone your project
git clone https://github.com/bledden/AMD_Hackathon.git
cd AMD_Hackathon
```

### 2.4 Run Setup Script
```bash
# Install all dependencies
bash setup/install_dependencies.sh

# This will:
# - Install Unsloth with ROCm support
# - Install training libraries (transformers, datasets, trl, etc.)
# - Install utilities
# - Verify GPU access
# Takes ~5-10 minutes
```

### 2.5 Verify Installation
```bash
# Run verification script
python3 setup/verify_gpu.py

# All checks should pass ✓
```

---

## Step 3: Prepare Dataset

### 3.1 Choose Your Theme
Edit [data/dataset_config.py](../data/dataset_config.py:29):
```python
# Options: "science", "history", "space", "technology"
SELECTED_THEME = "science"  # Change this
```

### 3.2 Generate Dataset
```bash
# Option A: Manual examples (quick start)
python3 data/prepare_dataset.py --strategy manual

# Option B: With synthetic examples
python3 data/prepare_dataset.py --strategy hybrid --n-synthetic 200

# Option C: Use existing dataset
python3 data/prepare_dataset.py --strategy existing --existing-dataset squad
```

### 3.3 Verify Dataset
```bash
# Check generated files
ls -lh data/processed/
# Should see: train.json, val.json, metadata.json

# Preview data
head -n 20 data/processed/train.json
```

---

## Step 4: Training

### 4.1 Start Training (First Run)
```bash
# Use default configuration (100 steps)
python3 training/scripts/train.py --config default

# Or fast config for testing (20 steps)
python3 training/scripts/train.py --config fast

# Or high quality (200 steps)
python3 training/scripts/train.py --config high_quality
```

### 4.2 Monitor Training
```bash
# In a separate SSH session:
watch -n 1 rocm-smi

# Monitor GPU usage, memory, temperature
```

### 4.3 Training Output
Training will save to `training/outputs/`:
```
training/outputs/
├── checkpoint-50/      # Intermediate checkpoint
├── checkpoint-100/     # Final checkpoint
└── final_model/        # Best model
    ├── merged_16bit/   # For inference
    ├── adapter_model.safetensors
    └── adapter_config.json
```

### 4.4 Expected Training Time
- **Fast config (20 steps)**: ~5-10 minutes
- **Default config (100 steps)**: ~20-30 minutes
- **High quality (200 steps)**: ~40-60 minutes

**Cost**: ~$1-2 per training run

---

## Step 5: Testing and Inference

### 5.1 Test Question Generation
```bash
# Generate 3 questions on science
python3 inference/generate_qa.py --mode generate --topic science --n-questions 3
```

### 5.2 Test Answer Generation
```bash
# Answer a question
python3 inference/generate_qa.py --mode answer \
  --question "What is quantum entanglement?"
```

### 5.3 Interactive Mode
```bash
# Interactive Q&A testing
python3 inference/generate_qa.py --mode interactive

# Commands:
#   /q [topic]     - Generate question
#   /a [question]  - Answer question
#   /quit          - Exit
```

### 5.4 Tournament Simulation
```bash
# Simulate tournament rounds
python3 inference/tournament_agent.py \
  --theme science \
  --strategy balanced \
  --rounds 3
```

---

## Step 6: Evaluation

### 6.1 Evaluate Tournament Results
```bash
python3 evaluation/evaluate.py --mode tournament \
  --input evaluation/tournament_results.json
```

### 6.2 Evaluate Single Q&A Pair
```bash
python3 evaluation/evaluate.py --mode pair \
  --question "What is CRISPR?" \
  --answer "CRISPR is a gene editing technology..."
```

### 6.3 Interpret Results
- **Question Quality**: 70%+ is good, 80%+ is excellent
- **Answer Quality**: 75%+ is good, 85%+ is excellent
- Look for:
  - Question clarity and specificity
  - Answer accuracy and completeness
  - Appropriate difficulty level

---

## Step 7: Iteration (If Needed)

### 7.1 Improve Dataset
```python
# Add more examples to data/prepare_dataset.py
# Focus on:
# - Weak areas from evaluation
# - More diverse question types
# - Better quality answers
```

### 7.2 Re-train
```bash
# Train again with improved dataset
python3 training/scripts/train.py --config default
```

### 7.3 Compare Checkpoints
```bash
# Test different checkpoints
python3 inference/generate_qa.py \
  --model-path training/outputs/checkpoint-50

python3 inference/generate_qa.py \
  --model-path training/outputs/checkpoint-100
```

---

## Step 8: Final Preparation for Competition

### 8.1 Select Best Model
```bash
# Based on evaluation, choose best checkpoint
# Usually final_model or last checkpoint
```

### 8.2 Test Tournament Scenarios
```bash
# Test with sample opponent questions
python3 inference/tournament_agent.py \
  --rounds 5 \
  --opponent-questions \
    "What is the speed of light?" \
    "Explain black holes" \
    "How does photosynthesis work?"
```

### 8.3 Tune Strategy
```bash
# Try different strategies
python3 inference/tournament_agent.py --strategy aggressive
python3 inference/tournament_agent.py --strategy balanced
python3 inference/tournament_agent.py --strategy defensive

# Choose best performing strategy
```

### 8.4 Create Submission Package
```bash
# Save final model and configs
mkdir submission
cp -r training/outputs/final_model submission/
cp data/dataset_config.py submission/
cp inference/tournament_agent.py submission/

# Create README for submission
cat > submission/README.md << 'EOF'
# Q&A Agent Submission - AMD Hackathon

**Theme**: Science
**Strategy**: Balanced
**Base Model**: LLaMA 3 8B
**Fine-tuning**: LoRA (r=16, 100 steps)
**Dataset**: 1000 Q&A pairs

## Usage
```bash
python tournament_agent.py --model-path final_model
```
EOF
```

---

## Cost Management

### Monitor Costs
```bash
# Track instance runtime
# Start time: record when you launch
START=$(date +%s)

# Check elapsed time
ELAPSED=$(( $(date +%s) - START ))
HOURS=$(echo "scale=2; $ELAPSED / 3600" | bc)
COST=$(echo "scale=2; $HOURS * 1.99" | bc)
echo "Runtime: $HOURS hours"
echo "Estimated cost: \$$COST"
```

### Stop Instance When Not Using
```bash
# From DigitalOcean dashboard:
# Droplets → Your Instance → Power → Power Off

# Or destroy instance:
# Droplets → Your Instance → Destroy

# WARNING: Save your model first!
```

### Save Checkpoints to Object Storage (Optional)
```bash
# Install rclone
sudo apt-get install rclone

# Configure DigitalOcean Spaces
rclone config

# Upload checkpoints
rclone copy training/outputs/ spaces:my-bucket/hackathon/
```

---

## Troubleshooting

### GPU Not Detected
```bash
# Check ROCm installation
rocm-smi --version

# Verify PyTorch sees GPU
python3 -c "import torch; print(torch.cuda.is_available())"

# Restart if needed
sudo reboot
```

### Out of Memory Error
```python
# Edit training/configs/default_config.py
# Reduce batch size:
per_device_train_batch_size = 1
gradient_accumulation_steps = 8

# Or use smaller model:
model_name = "unsloth/tinyllama-bnb-4bit"
```

### Slow Training
```bash
# Check GPU utilization
rocm-smi

# Should see ~80-100% GPU usage
# If low, increase batch size or reduce gradient accumulation
```

### Model Generation Issues
```python
# Adjust temperature in inference/generate_qa.py
# Higher temp (0.7-0.9) = more creative
# Lower temp (0.1-0.3) = more focused
```

### SSH Connection Lost
```bash
# Use tmux for persistent sessions
sudo apt-get install tmux

# Start tmux session
tmux new -s training

# Run training in tmux
python3 training/scripts/train.py

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

---

## Quick Reference Commands

```bash
# GPU status
rocm-smi

# Train model
python3 training/scripts/train.py --config default

# Generate questions
python3 inference/generate_qa.py --mode generate --topic science

# Answer questions
python3 inference/generate_qa.py --mode answer --question "..."

# Tournament simulation
python3 inference/tournament_agent.py --rounds 3

# Evaluate results
python3 evaluation/evaluate.py --mode tournament

# Monitor GPU
watch -n 1 rocm-smi

# Check costs
echo "Hours: $(($(date +%s) - START_TIME) / 3600)"
```

---

## Competition Day Checklist

### Before Competition
- [ ] Model trained and tested
- [ ] Inference pipeline working
- [ ] Strategy selected and tuned
- [ ] Backup checkpoint saved
- [ ] Instance running and accessible

### During Competition
- [ ] Submit agent on time
- [ ] Monitor performance
- [ ] Take notes on questions/answers
- [ ] Identify improvement areas

### After Competition
- [ ] Download results
- [ ] Save final model
- [ ] Stop/destroy instance
- [ ] Document learnings
- [ ] Celebrate! 🎉

---

## Support Resources

- **Unsloth Discord**: https://discord.gg/unsloth
- **AMD ROCm Issues**: https://github.com/ROCm/ROCm/issues
- **DigitalOcean Support**: support.digitalocean.com
- **Project Repo**: https://github.com/bledden/AMD_Hackathon

---

**Good luck with your Q&A agent! 🚀**
