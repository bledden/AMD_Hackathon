# AMD Hackathon Q&A Agent

Competition project for AMD Hackathon using Unsloth and AMD Instinct MI300X GPUs to build a Q&A agent that competes in tournament-style battles.

## Overview

This project fine-tunes language models using the Unsloth library optimized for AMD MI300X GPUs to create an AI agent that can:
- Generate creative and challenging questions on specific themes
- Answer questions accurately and concisely
- Compete in bracket-style Q&A tournaments

## Tech Stack

- **GPU**: AMD Instinct MI300X (DigitalOcean)
- **Platform**: PyTorch 2.6.0 + ROCm 7.0.0
- **Optimization**: Unsloth (2x faster training, 80% less memory)
- **Base Model**: TBD (LLaMA 3 8B, Mistral 7B, or Qwen 2.5 7B)
- **Fine-tuning**: Supervised Fine-Tuning (SFT) with LoRA/QLoRA

## Timeline

- **Start**: October 24, 2025 (Friday)
- **Competition**: October 29, 2025 (Wednesday)
- **Duration**: 5-day sprint
- **Budget**: $30-50 of DigitalOcean AMD credits

## Project Structure

```
AMD_Hackathon/
├── README.md           # This file
├── setup/             # Setup scripts and environment configs
├── data/              # Dataset preparation and generation
├── training/          # Fine-tuning scripts and configs
├── inference/         # Model inference and Q&A generation
├── evaluation/        # Testing and evaluation scripts
└── docs/              # Additional documentation
```

## Quick Start

### 1. Deploy MI300X Instance
```bash
# On DigitalOcean AMD platform (amd.digitalocean.com)
# Select: Single MI300X with PyTorch 2.6.0 + ROCm 7.0.0
# Cost: $1.99/hr
```

### 2. Install Dependencies
```bash
# SSH into MI300X instance
pip install "unsloth[rocm] @ git+https://github.com/unslothai/unsloth.git"
pip install transformers datasets trl accelerate peft bitsandbytes
```

### 3. Verify GPU
```bash
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
rocm-smi
```

## Development Plan

### Phase 1: Setup & Baseline (Day 1)
- Deploy MI300X instance
- Install Unsloth and dependencies
- Load base model and test inference
- Establish baseline performance

### Phase 2: Dataset & Fine-tuning (Day 2)
- Create/gather Q&A dataset for chosen theme
- Format data for instruction tuning
- Run initial fine-tune with LoRA
- Evaluate and iterate

### Phase 3: Optimization (Day 3)
- Improve dataset quality and diversity
- Refine hyperparameters
- Run second fine-tune
- Systematic evaluation

### Phase 4: Polish (Day 4-5)
- Optimize for tournament format
- Tune question difficulty and answer accuracy
- Final model checkpoint
- Documentation and submission prep

## Competition Strategy

### Question Generation
- Generate diverse, challenging questions
- Balance difficulty to maximize opponent errors
- Use temperature ~0.7-0.9 for creativity

### Answer Strategy
- Prioritize accuracy over eloquence
- Keep answers concise
- Use temperature ~0.1-0.3 for consistency

## Resources

- **Unsloth**: https://github.com/unslothai/unsloth
- **AMD Blog Post**: https://www.amd.com/en/developer/resources/technical-articles/2025/10x-model-fine-tuning-using-synthetic-data-with-unsloth.html
- **ROCm Docs**: https://rocm.docs.amd.com/

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, refer to:
- Unsloth Discord: https://discord.gg/unsloth
- AMD ROCm Issues: https://github.com/ROCm/ROCm/issues
