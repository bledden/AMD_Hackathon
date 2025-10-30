#!/bin/bash
# Monitor distillation progress

echo "=========================================="
echo "🔬 DISTILLATION PROGRESS MONITOR"
echo "=========================================="
echo ""

# Check if process is running
echo "📊 Process Status:"
ssh amd-hackathon "docker exec rocm bash -c 'ps aux | grep generate_distillation_data | grep -v grep' 2>/dev/null" | head -1
if [ $? -eq 0 ]; then
    echo "✅ Distillation is RUNNING"
else
    echo "❌ Distillation not running (may have completed or crashed)"
fi
echo ""

# Check progress from log
echo "📈 Latest Progress:"
ssh amd-hackathon "docker exec rocm bash -c 'tail -50 /home/rocm-user/AMD_Hackathon/logs/distillation_generation.log 2>/dev/null | grep -E \"(Processing|questions|Failed|checkpoint|Rate|Estimated)\" | tail -15'"
echo ""

# Check output file size
echo "💾 Output File:"
ssh amd-hackathon "docker exec rocm bash -c 'ls -lh /home/rocm-user/AMD_Hackathon/data/distillation/*3000q* 2>/dev/null || echo \"File not yet created\"'"
echo ""

# Check checkpoints
echo "📍 Checkpoints:"
ssh amd-hackathon "docker exec rocm bash -c 'ls -lh /home/rocm-user/AMD_Hackathon/data/distillation/*checkpoint* 2>/dev/null | tail -5 || echo \"No checkpoints yet\"'"
echo ""

# GPU memory
echo "💾 GPU Memory:"
ssh amd-hackathon "docker exec rocm bash -c 'rocm-smi --showmeminfo vram 2>/dev/null | grep -E \"GPU\|Memory\" | head -5'"
echo ""

echo "=========================================="
echo "To follow logs in real-time:"
echo "ssh amd-hackathon \"docker exec rocm tail -f /home/rocm-user/AMD_Hackathon/logs/distillation_generation.log\""
echo "=========================================="
