#!/bin/bash
# Organize AMD Hackathon repository

# Move planning/strategy documents to archive/planning
mv CURRENT_STRATEGY_DISTILLATION.md archive/planning/
mv DEPLOYMENT_CHECKLIST.md archive/planning/
mv DETAILED_OPTIONS_ANALYSIS.md archive/planning/
mv DISTILLATION_STATUS.md archive/planning/
mv FINAL_EXECUTION_PLAN.md archive/planning/
mv FINAL_STRATEGY_DECISION.md archive/planning/
mv FINAL_STRATEGY_SPECS.md archive/planning/
mv GPT_PLAN_TECHNICAL_ANALYSIS.md archive/planning/
mv HACKATHON_BRIEFING_FOR_EXTERNAL_REVIEW.md archive/planning/
mv HACKATHON_REQUIREMENTS_AND_STRATEGY.md archive/planning/
mv MODEL_COMPARISON_VALIDATED.md archive/planning/
mv PROJECT_PLAN.md archive/planning/
mv QUICKSTART.md archive/planning/
mv SINGLE_DROPLET_DECISION_ANALYSIS.md archive/planning/
mv STATUS_UPDATE.md archive/planning/
mv THREE_AGENT_STRATEGY_EXPLAINED.md archive/planning/

# Move legacy/backup agents to archive/legacy_agents
mv answer_agent.py archive/legacy_agents/
mv answer_agent_deepseek_timeout.py archive/legacy_agents/
mv answer_agent_qwen7b_backup.py archive/legacy_agents/
mv question_agent.py archive/legacy_agents/
mv tournament_server.py archive/legacy_agents/

# Move test scripts to tests/
mv test_question_agent.py tests/
mv test_tournament_endpoints.py tests/

# Move archives to archive/
mv AIAC.tar.gz archive/ 2>/dev/null || true
mv AIAC_updated.tar.gz archive/ 2>/dev/null || true
mv README_backup.md archive/ 2>/dev/null || true

# Move main documentation to docs/ (keep at root level via symlinks)
cp COMPLETE_JOURNEY_DOCUMENTATION.md docs/
cp TOURNAMENT_CONNECTION_GUIDE.md docs/

# Move current implementation to root (already there)
# Keep: answer_agent_qwen7b_final.py, tournament_server_qwen7b.py, AIAC/

echo "Repository organized!"
