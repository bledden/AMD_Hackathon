# Repository Structure

This document describes the organization of the AMD Hackathon repository.

## Root Directory Files

```
├── README.md                           # Main project documentation with complete journey
├── COMPLETE_JOURNEY_DOCUMENTATION.md   # 20-page technical deep-dive
├── TOURNAMENT_CONNECTION_GUIDE.md      # Deployment and testing guide
├── answer_agent_qwen7b_final.py       # Final tournament solution (backup/reference)
├── tournament_server_qwen7b.py        # HTTP server implementation (backup/reference)
└── .gitignore                         # Git ignore patterns
```

## Main Directories

### AIAC/ - Tournament Submission
```
AIAC/
└── agents/
    ├── __init__.py
    ├── answer_model.py      # FINAL: Qwen2.5-7B + timeout (92% accuracy)
    └── question_model.py    # FINAL: Question pool selector
```
**Purpose**: Official tournament submission structure. These are the production agents.

### scripts/ - Training & Testing Scripts
```
scripts/
├── generate_distillation_data.py      # Attempt 1: Reasoning chains generation
├── create_simple_training_data.py     # Attempt 2: Simple Q→A format
├── analyze_failures.py                # Attempt 3: Domain weakness analysis
├── train_ultra_minimal.py             # Attempt 4: Anti-mode-collapse training
├── test_qwen7b_quick.py              # Attempt 5: Winner validation
├── test_baseline_speed.py             # Speed compliance testing
├── test_baseline_tokens4.py           # Token optimization testing
├── test_baseline_final.py             # Final baseline validation
├── debug_adapter_tokens.py            # Mode collapse detection
├── download_deepseek_r1_32b.py       # Model download utility
└── [other experimental scripts]
```
**Purpose**: All training attempts, testing scripts, and utilities from the journey.

### docs/ - Documentation
```
docs/
├── COMPLETE_JOURNEY_DOCUMENTATION.md   # Complete technical analysis
├── TOURNAMENT_CONNECTION_GUIDE.md      # Deployment instructions
├── COMPETITION_STRATEGY.md             # Strategy documentation
├── DEPLOYMENT_GUIDE.md                 # Deployment guide
├── RESEARCH_FINDINGS.md                # Research notes
└── [other docs]
```
**Purpose**: Technical documentation, guides, and research notes.

### data/ - Datasets
```
data/
└── curriculum/
    └── val_5k.json          # Validation dataset (5000 questions)
```
**Purpose**: Training and validation datasets.

### archive/ - Historical Files
```
archive/
├── planning/                # Original strategy documents
│   ├── CURRENT_STRATEGY_DISTILLATION.md
│   ├── FINAL_EXECUTION_PLAN.md
│   ├── HACKATHON_REQUIREMENTS_AND_STRATEGY.md
│   └── [15+ planning documents]
│
├── legacy_agents/           # Non-final agent implementations
│   ├── answer_agent.py                  # DeepSeek-R1 version
│   ├── answer_agent_deepseek_timeout.py # Timeout attempt
│   ├── answer_agent_qwen7b_backup.py   # Qwen backup
│   ├── question_agent.py                # Old question agent
│   └── tournament_server.py             # Old server
│
├── AIAC.tar.gz             # Archive snapshots
├── AIAC_updated.tar.gz
└── README_backup.md        # Old README version
```
**Purpose**: Historical planning documents, legacy implementations, and backups.

### tests/ - Test Scripts
```
tests/
├── test_question_agent.py        # Question agent tests
└── test_tournament_endpoints.py  # Server endpoint tests
```
**Purpose**: Unit and integration tests.

### results/ - Test Results (empty, for future use)
```
results/
└── [Test outputs, benchmark results, etc.]
```
**Purpose**: Store test results, validation outputs, benchmark data.

## Directory Usage Guide

### For Users/Reviewers:
1. **Start here**: `README.md` - Complete story and quick overview
2. **Deep dive**: `COMPLETE_JOURNEY_DOCUMENTATION.md` - 20-page technical analysis
3. **Deployment**: `TOURNAMENT_CONNECTION_GUIDE.md` - How to deploy/test
4. **Production code**: `AIAC/agents/` - Tournament submission files

### For Developers:
1. **Scripts**: `scripts/` - All training/testing attempts
2. **Documentation**: `docs/` - Technical guides and research
3. **Legacy code**: `archive/legacy_agents/` - Previous implementations
4. **Planning**: `archive/planning/` - Strategy evolution

### For Researchers:
1. **Journey**: `COMPLETE_JOURNEY_DOCUMENTATION.md` - All attempts documented
2. **Scripts**: `scripts/` - Reproducible experiments
3. **Planning**: `archive/planning/` - Decision-making process
4. **Results**: Check validation accuracies in journey doc

## Key Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `AIAC/agents/answer_model.py` | Tournament answer agent | PRODUCTION |
| `AIAC/agents/question_model.py` | Tournament question agent | PRODUCTION |
| `README.md` | Complete project story | CURRENT |
| `COMPLETE_JOURNEY_DOCUMENTATION.md` | Technical deep-dive | REFERENCE |
| `scripts/test_qwen7b_quick.py` | Winner validation script | REFERENCE |
| `archive/legacy_agents/*` | Old implementations | ARCHIVED |
| `archive/planning/*` | Strategy documents | ARCHIVED |

## Clean Repository Commands

```bash
# View only production files
ls AIAC/agents/

# View all training attempts
ls scripts/

# View historical planning
ls archive/planning/

# View legacy implementations
ls archive/legacy_agents/
```

## Notes

- **AIAC/** is the tournament submission directory (main deliverable)
- **archive/** contains all historical documents (not needed for understanding final solution)
- **scripts/** contains the journey's experiments (good for learning/reproduction)
- Root-level `.py` files are backups of the final solution for easy reference
