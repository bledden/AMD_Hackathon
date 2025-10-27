#!/bin/bash
# Simple script to check Qwen2.5-72B training status

echo "=========================================="
echo "🎯 QWEN2.5-72B TRAINING STATUS CHECK"
echo "=========================================="
echo ""

# Check if training process is running
echo "📊 Process Status:"
ssh amd-hackathon "docker exec rocm bash -c 'ps aux | grep train_qwen2.5_unsloth.py | grep -v grep' 2>/dev/null" | head -1
if [ $? -eq 0 ]; then
    echo "✅ Training is RUNNING"
else
    echo "❌ Training not found (may have completed or crashed)"
fi
echo ""

# Check GPU memory usage
echo "💾 GPU Memory Usage:"
ssh amd-hackathon "docker exec rocm bash -c 'rocm-smi --showmeminfo vram'" 2>/dev/null | grep -A 2 "GPU\|Memory"
echo ""

# Get latest training progress
echo "📈 Latest Training Progress:"
ssh amd-hackathon "docker exec rocm bash -c 'tail -30 /home/rocm-user/AMD_Hackathon/training_qwen2.5.log 2>/dev/null | grep -E \"(Chunk|step|loss|eval|checkpoint|complete|error|GPU)\" | tail -10'"
echo ""

# Check current chunk and step
echo "📍 Current Position:"
ssh amd-hackathon "docker exec rocm bash -c 'grep \"Chunk\" /home/rocm-user/AMD_Hackathon/training_qwen2.5.log 2>/dev/null | tail -1'"
echo ""

# Show latest loss values
echo "📉 Recent Loss Values:"
ssh amd-hackathon "docker exec rocm bash -c 'grep \"loss\" /home/rocm-user/AMD_Hackathon/training_qwen2.5.log 2>/dev/null | tail -5'"
echo ""

# Estimate completion
echo "⏰ Time Estimate:"
STARTED=$(ssh amd-hackathon "docker exec rocm bash -c 'stat -c %Y /home/rocm-user/AMD_Hackathon/training_qwen2.5.log' 2>/dev/null")
NOW=$(date +%s)
if [ ! -z "$STARTED" ]; then
    ELAPSED=$((NOW - STARTED))
    HOURS=$((ELAPSED / 3600))
    MINUTES=$(((ELAPSED % 3600) / 60))
    echo "   Training has been running for: ${HOURS}h ${MINUTES}m"
    echo "   Expected total time: 6-8 hours"
    echo "   Estimated completion: $((8 - HOURS))h remaining (worst case)"
fi
echo ""
echo "=========================================="
echo "For detailed logs, run:"
echo "ssh amd-hackathon \"docker exec rocm tail -100 /home/rocm-user/AMD_Hackathon/training_qwen2.5.log\""
echo "=========================================="