# AMD Hackathon Q&A Agent - 3-Agent Parallel Strategy

Competition project for AMD Hackathon using Unsloth and AMD Instinct MI300X GPUs to build **3 parallel Q&A agents** that compete in tournament-style battles.

## Overview

This project fine-tunes **3 different language models in parallel** using the Unsloth library optimized for AMD MI300X GPUs to create multiple AI agents that can:
- Generate creative and challenging questions on specific themes
- Answer questions accurately and concisely
- Compete in bracket-style Q&A tournaments

**Strategy**: Deploy 3 agents simultaneously, test them all, and submit the best performer while showcasing the complete experimental process.

## Three-Agent Approach

### Agent 1: "Foundation" - Conservative Reliable
- **Model**: Microsoft Phi-4 14B Instruct (2025)
- **Strategy**: Proven SFT approach with curated dataset
- **Strength**: Superior reasoning (matches GPT-4o-mini)
- **Performance**: Fits in <15GB VRAM, 2x faster with Unsloth
- **Cost**: ~$120-160 (60-80 hours)

### Agent 2: "Challenger" - Aggressive Creative
- **Model**: Qwen3 8B Instruct (2025)
- **Strategy**: Synthetic data + creative question generation
- **Strength**: Hybrid reasoning, 8x longer context
- **Performance**: Dynamic 2.0 quantization, best MMLU scores
- **Cost**: ~$120-160 (60-80 hours)

### Agent 3: "Hybrid" - Domain Specialist
- **Model**: Mistral NeMo 12B Instruct (2024)
- **Strategy**: Deep domain expertise (science/tech/history)
- **Strength**: Superior instruction following, 128K context
- **Performance**: Fits in 12GB VRAM, 2x faster with Unsloth
- **Cost**: ~$80-120 (40-60 hours, faster training)

**Total Budget**: $180-250 (3 MI300X droplets running in parallel)

## Tech Stack

- **GPU**: 3× AMD Instinct MI300X (DigitalOcean)
- **Platform**: ROCm 6.4.0 (REQUIRED - confirmed by organizers)
- **Optimization**: Unsloth (2x faster training, 70% less memory)
- **Fine-tuning**: Supervised Fine-Tuning (SFT) with LoRA/QLoRA
- **Models**: 2025-era models with full Unsloth support
- **Available Credits**: $300 now + $300 after Wednesday = $600 total

## Timeline

- **Start**: Saturday, October 26, 2025
- **Saturday**: Deploy all 3 agents, setup environments
- **Sunday**: First fine-tunes for all 3 agents
- **Monday**: Iteration and optimization
- **Tuesday**: Final tuning and comparative testing
- **Wednesday**: Evaluation, select winner, submit to competition
- **Duration**: 4-day sprint

## Project Structure

```
AMD_Hackathon/
├── README.md              # This file
├── PROJECT_PLAN.md        # Detailed 5-day plan (original single-agent)
├── QUICKSTART.md          # Quick start guide
├── setup/                 # Setup scripts and environment configs
│   ├── install_dependencies.sh
│   └── verify_gpu.py
├── data/                  # Dataset preparation and generation
│   ├── dataset_config.py
│   └── prepare_dataset.py
├── training/              # Fine-tuning scripts and configs
│   ├── configs/default_config.py
│   └── scripts/train.py
├── inference/             # Model inference and Q&A generation
│   ├── generate_qa.py
│   └── tournament_agent.py
├── evaluation/            # Testing and evaluation scripts
│   └── evaluate.py
└── docs/                  # Additional documentation
    ├── DEPLOYMENT_GUIDE.md
    └── COMPETITION_STRATEGY.md
```

## Quick Start (Per Agent)

### 1. Deploy MI300X Instances
```bash
# On DigitalOcean AMD platform (amd.digitalocean.com)
# Deploy 3 separate droplets:
# - agent-1-foundation-llama3
# - agent-2-challenger-qwen
# - agent-3-hybrid-mistral
# Each: PyTorch 2.6.0 + ROCm 7.0.0, Single MI300X ($1.99/hr)
```

### 2. Setup Each Agent
```bash
# SSH into each instance separately
ssh root@agent-1-IP
git clone https://github.com/bledden/AMD_Hackathon.git
cd AMD_Hackathon
bash setup/install_dependencies.sh
python3 setup/verify_gpu.py
```

### 3. Configure Each Agent
```python
# Agent 1: Edit training/configs/agent1_config.py
model_name = "unsloth/Phi-4-bnb-4bit"
r = 16  # Conservative LoRA

# Agent 2: Edit training/configs/agent2_config.py
model_name = "unsloth/Qwen3-8B-Instruct-bnb-4bit"
r = 24  # Aggressive LoRA

# Agent 3: Edit training/configs/agent3_config.py
model_name = "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit"
r = 16  # Balanced LoRA
# Choose domain specialization in data/dataset_config.py
```

### 4. Prepare Datasets
```bash
# Agent 1: Curated quality dataset
python3 data/prepare_dataset.py --strategy manual --theme science

# Agent 2: Large synthetic dataset
python3 data/prepare_dataset.py --strategy hybrid --n-synthetic 2000

# Agent 3: Domain-specialized
python3 data/prepare_dataset.py --strategy existing --theme technology
```

### 5. Train in Parallel
```bash
# On each agent (runs simultaneously on 3 droplets):
python3 training/scripts/train.py --config default
```

### 6. Monitor All Agents
```bash
# From local machine with SSH config
for agent in agent-1 agent-2 agent-3; do
  ssh $agent "rocm-smi | grep GPU"
done
```

## Why 3 Agents?

1. **Maximize Winning Chances**: Test multiple strategies simultaneously
2. **Comprehensive Submission**: Show full experimental process
3. **Hedge Against Failure**: If one approach fails, others may succeed
4. **Time-Efficient**: Run Sat-Tue in parallel vs sequentially
5. **Learning Value**: Direct comparison of approaches

## Development Plan

### Saturday (Day 1) - Parallel Setup
- Deploy all 3 MI300X droplets (3 hours)
- Install Unsloth + dependencies on each (1.5 hours)
- Verify GPU works on all 3 (30 min)
- Prepare datasets in parallel (4 hours)
- **Cost**: ~$24-36 (3 × 4-6hrs × $1.99)

### Sunday (Day 2) - First Fine-tunes
- Run first fine-tune on all 3 agents (8-10 hours)
- Initial testing and evaluation (2 hours)
- **Cost**: ~$60-72

### Monday (Day 3) - Iteration & Optimization
- Improve datasets based on results (2 hours)
- Second fine-tune on all 3 agents (8-10 hours)
- **Cost**: ~$48-60

### Tuesday (Day 4) - Final Tuning & Testing
- Final refinements (4-6 hours)
- Mock tournaments between agents (4-6 hours)
- Document each approach (2 hours)
- **Cost**: ~$48-72

### Wednesday (Day 5) - Evaluation & Submission
- Compare all 3 agents systematically (3 hours, no GPU)
- Select winner based on performance metrics
- Prepare comprehensive submission
- Submit winner to competition
- **Cost**: $0 (local evaluation)

**Total Cost**: $180-240 typical, up to $437 maximum

## Competition Strategy

### Question Generation
- **Agent 1**: Moderate difficulty, well-structured
- **Agent 2**: Challenging questions, creative phrasing
- **Agent 3**: Expert-level domain questions

### Answer Strategy
- **Agent 1**: High accuracy priority
- **Agent 2**: Balance creativity and correctness
- **Agent 3**: Deep domain knowledge

### Selection Criteria (Wednesday)
- Question quality: 40% weight
- Answer accuracy: 40% weight
- Tournament strategy: 20% weight

## Monitoring Multiple Agents

### SSH Config Setup
```bash
# ~/.ssh/config
Host agent-1
    HostName AGENT-1-IP
    User root

Host agent-2
    HostName AGENT-2-IP
    User root

Host agent-3
    HostName AGENT-3-IP
    User root
```

### Check All Agents
```bash
# Monitor all 3 simultaneously
for i in 1 2 3; do
  echo "=== Agent $i ==="
  ssh agent-$i "rocm-smi; tail -10 ~/AMD_Hackathon/training/outputs/training.log"
done
```

### Cost Tracking
```bash
# Calculate total spend
AGENT1_HOURS=60  # Update with actual hours
AGENT2_HOURS=65
AGENT3_HOURS=50
TOTAL_COST=$(echo "scale=2; ($AGENT1_HOURS + $AGENT2_HOURS + $AGENT3_HOURS) * 1.99" | bc)
echo "Total cost: \$$TOTAL_COST"
```

## Resources

- **Parallel Strategy Guide**: [AMD_HACKATHON_PARALLEL_STRATEGY.md](/Users/bledden/Documents/AMD_HACKATHON_PARALLEL_STRATEGY.md)
- **Original Single-Agent Plan**: [PROJECT_PLAN.md](PROJECT_PLAN.md)
- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **Deployment Guide**: [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)
- **Competition Strategy**: [docs/COMPETITION_STRATEGY.md](docs/COMPETITION_STRATEGY.md)
- **Unsloth**: https://github.com/unslothai/unsloth
- **AMD Blog Post**: https://www.amd.com/en/developer/resources/technical-articles/2025/10x-model-fine-tuning-using-synthetic-data-with-unsloth.html
- **ROCm Docs**: https://rocm.docs.amd.com/

## Budget Summary

- **Available**: $300 (now) + $300 (after Wed) = $600 total
- **AMD Hackathon (3 agents)**: $180-250
- **Remaining for dendritic research**: $350-420
- **Buffer**: Within budget, plenty of headroom

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, refer to:
- Unsloth Discord: https://discord.gg/unsloth
- AMD ROCm Issues: https://github.com/ROCm/ROCm/issues

---

**Updated Strategy**: This README reflects the **3-agent parallel approach** with updated budget ($180-250) and timeline (Sat-Wed). The original single-agent plan is preserved in [PROJECT_PLAN.md](PROJECT_PLAN.md) for reference.
