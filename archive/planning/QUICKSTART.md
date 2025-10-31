# Quick Start Guide - AMD Hackathon Q&A Agent

Get your Q&A agent running in 30 minutes or less!

---

## Overview

This project provides everything you need to build, train, and deploy a Q&A agent for the AMD Hackathon competition using:
- **AMD MI300X GPU** (192GB VRAM)
- **Unsloth** optimization library (2x faster, 80% less memory)
- **LLaMA 3 8B** or Mistral 7B base model
- **LoRA fine-tuning** for efficiency

**Competition**: October 29, 2025
**Budget**: $30-50 (15-25 hours GPU time @ $1.99/hr)

---

## 10-Minute Local Setup

Before deploying to MI300X, set up your local environment:

### 1. Clone Repository
```bash
git clone https://github.com/bledden/AMD_Hackathon.git
cd AMD_Hackathon
```

### 2. Review Project Structure
```
AMD_Hackathon/
├── README.md              # Project overview
├── PROJECT_PLAN.md        # Detailed 5-day plan
├── QUICKSTART.md          # This file
├── setup/                 # MI300X setup scripts
│   ├── install_dependencies.sh
│   └── verify_gpu.py
├── data/                  # Dataset preparation
│   ├── dataset_config.py
│   └── prepare_dataset.py
├── training/              # Fine-tuning scripts
│   ├── configs/default_config.py
│   └── scripts/train.py
├── inference/             # Q&A generation
│   ├── generate_qa.py
│   └── tournament_agent.py
├── evaluation/            # Quality evaluation
│   └── evaluate.py
└── docs/                  # Detailed guides
    ├── DEPLOYMENT_GUIDE.md
    └── COMPETITION_STRATEGY.md
```

### 3. Choose Your Theme
Edit [data/dataset_config.py](data/dataset_config.py#L29):
```python
# Options: "science", "history", "space", "technology"
SELECTED_THEME = "science"  # Change this!
```

### 4. Review Configuration
Check [training/configs/default_config.py](training/configs/default_config.py) - defaults are good!

**Ready for MI300X!**

---

## 20-Minute MI300X Deployment

### Step 1: Deploy Instance (5 min)
1. Go to https://amd.digitalocean.com
2. Create → Droplets
3. Select:
   - **Image**: PyTorch 2.6.0 + ROCm 7.0.0
   - **Plan**: Single MI300X ($1.99/hr)
4. Add your SSH key
5. Launch!

### Step 2: Setup Environment (10 min)
```bash
# SSH into instance
ssh root@YOUR_IP_ADDRESS

# Clone repo
git clone https://github.com/bledden/AMD_Hackathon.git
cd AMD_Hackathon

# Run automated setup
bash setup/install_dependencies.sh

# Verify (should see all ✓)
python3 setup/verify_gpu.py
```

### Step 3: Prepare Dataset (5 min)
```bash
# Generate training data (manual examples for quick start)
python3 data/prepare_dataset.py --strategy manual

# Check output
ls -lh data/processed/
# Should see: train.json, val.json, metadata.json
```

**Environment Ready! 🚀**

---

## 30-Minute First Training Run

### Start Training
```bash
# Fast config for testing (20 steps, ~5-10 min)
python3 training/scripts/train.py --config fast

# OR default config for real training (100 steps, ~20-30 min)
python3 training/scripts/train.py --config default
```

### Monitor GPU
```bash
# In a separate SSH session
watch -n 1 rocm-smi
```

### Expected Output
```
==========================================
AMD Hackathon - Q&A Agent Fine-tuning
==========================================
Config: default
Model: unsloth/llama-3-8b-bnb-4bit
==========================================

Loading dataset from data/processed/train.json...
Loaded 200 examples
✓ Datasets formatted

==========================================
Setting up model with Unsloth...
==========================================
Loading model: unsloth/llama-3-8b-bnb-4bit
✓ Model loaded
Adding LoRA adapters (r=16)...
✓ LoRA adapters added
==========================================

Setting up trainer...

==========================================
Training Configuration
==========================================
Train examples: 180
Val examples: 20
Batch size: 2
Gradient accumulation: 4
Effective batch size: 8
Learning rate: 0.0002
Max steps: 100
Output dir: training/outputs
==========================================

Starting training...
==========================================
[Training progress bars...]
==========================================
✓ Training completed in X seconds
==========================================

Saving model to training/outputs/final_model...
✓ Model saved
✓ 16-bit model saved

==========================================
Training Complete! 🎉
==========================================
```

**Cost**: ~$1 (fast) or ~$1-2 (default)

---

## Test Your Agent

### Generate Questions
```bash
python3 inference/generate_qa.py \
  --mode generate \
  --topic "quantum mechanics" \
  --n-questions 3
```

**Example Output**:
```
1. What is the Heisenberg Uncertainty Principle and why is it significant?
2. How does quantum entanglement differ from classical correlation?
3. Explain the concept of wave-particle duality in quantum mechanics.
```

### Answer Questions
```bash
python3 inference/generate_qa.py \
  --mode answer \
  --question "What is CRISPR?"
```

**Example Output**:
```
Answer: CRISPR is a gene editing technology that uses a guide RNA
to direct the Cas9 enzyme to specific DNA sequences, allowing
scientists to precisely modify genetic material with applications
in medicine, agriculture, and research.
```

### Tournament Simulation
```bash
python3 inference/tournament_agent.py \
  --theme science \
  --strategy balanced \
  --rounds 3
```

---

## Evaluate Quality

```bash
python3 evaluation/evaluate.py \
  --mode tournament \
  --input evaluation/tournament_results.json
```

**Look for**:
- Question Quality: 70%+ is good, 80%+ is excellent
- Answer Quality: 75%+ is good, 85%+ is excellent

---

## Competition Ready Checklist

- [ ] Repository cloned
- [ ] Theme selected
- [ ] MI300X instance deployed
- [ ] Environment setup complete
- [ ] GPU verified
- [ ] Dataset prepared (500-2000 examples)
- [ ] Model trained (100+ steps)
- [ ] Questions tested (quality 70%+)
- [ ] Answers tested (quality 75%+)
- [ ] Tournament simulation run
- [ ] Strategy selected (balanced recommended)
- [ ] Final model saved

---

## Key Files Reference

### Must Read
1. **[PROJECT_PLAN.md](PROJECT_PLAN.md)** - Detailed 5-day implementation plan
2. **[docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)** - Complete deployment steps
3. **[docs/COMPETITION_STRATEGY.md](docs/COMPETITION_STRATEGY.md)** - Tournament tactics

### Configuration
- [data/dataset_config.py](data/dataset_config.py) - Theme selection
- [training/configs/default_config.py](training/configs/default_config.py) - Training parameters

### Scripts
- [setup/install_dependencies.sh](setup/install_dependencies.sh) - MI300X setup
- [data/prepare_dataset.py](data/prepare_dataset.py) - Dataset preparation
- [training/scripts/train.py](training/scripts/train.py) - Training
- [inference/tournament_agent.py](inference/tournament_agent.py) - Competition agent
- [evaluation/evaluate.py](evaluation/evaluate.py) - Quality evaluation

---

## Common Commands

```bash
# Setup
bash setup/install_dependencies.sh
python3 setup/verify_gpu.py

# Data
python3 data/prepare_dataset.py --strategy manual

# Training
python3 training/scripts/train.py --config default

# Testing
python3 inference/generate_qa.py --mode interactive
python3 inference/tournament_agent.py --rounds 3

# Evaluation
python3 evaluation/evaluate.py --mode tournament

# Monitoring
rocm-smi                    # GPU status
watch -n 1 rocm-smi         # Live GPU monitoring
```

---

## Troubleshooting

### GPU Not Detected
```bash
rocm-smi --version
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory
Edit [training/configs/default_config.py](training/configs/default_config.py):
```python
per_device_train_batch_size = 1  # Reduce from 2
gradient_accumulation_steps = 8  # Increase from 4
```

### Slow Training
- Check GPU utilization: `rocm-smi`
- Should be 80-100% during training
- If low, issue with setup

### Poor Question Quality
- Add more examples to dataset
- Increase training steps
- Tune temperature in inference

---

## Next Steps

1. **Review Strategy**: Read [docs/COMPETITION_STRATEGY.md](docs/COMPETITION_STRATEGY.md)
2. **Improve Dataset**: Add more high-quality examples
3. **Longer Training**: Use `--config high_quality` (200 steps)
4. **Optimize**: Test different strategies (aggressive, defensive)
5. **Practice**: Run tournament simulations

---

## Cost Tracking

```bash
# Track your spend
START_TIME=$(date +%s)
# ... work on MI300X ...
ELAPSED=$(( $(date +%s) - START_TIME ))
HOURS=$(echo "scale=2; $ELAPSED / 3600" | bc)
COST=$(echo "scale=2; $HOURS * 1.99" | bc)
echo "Runtime: $HOURS hours = \$$COST"
```

**Remember**: Stop instance when not using!

---

## Support

- **Unsloth Discord**: https://discord.gg/unsloth
- **AMD ROCm**: https://github.com/ROCm/ROCm/issues
- **Project Issues**: https://github.com/bledden/AMD_Hackathon/issues

---

## Timeline

### Today (Friday)
- [ ] Deploy MI300X
- [ ] Setup environment
- [ ] Prepare dataset
- [ ] First training run
- [ ] Test inference

### Weekend
- [ ] Improve dataset
- [ ] Longer training
- [ ] Optimize strategy
- [ ] Tournament testing

### Monday-Tuesday
- [ ] Final tuning
- [ ] Strategy selection
- [ ] Submission prep

### Wednesday
- [ ] Competition! 🏆

---

**You're ready to build an awesome Q&A agent! Good luck! 🚀**
