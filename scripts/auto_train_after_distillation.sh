#!/bin/bash
# Automatically start training when distillation completes

echo "=========================================="
echo "🤖 AUTO-TRAINING MONITOR"
echo "=========================================="
echo ""
echo "Waiting for distillation to complete..."
echo ""

# Wait for distillation to finish
while true; do
    # Check if distillation process is still running
    RUNNING=$(ssh amd-hackathon "docker exec rocm bash -c 'ps aux | grep generate_distillation_data.py | grep -v grep' 2>/dev/null" | wc -l)

    if [ "$RUNNING" -eq "0" ]; then
        echo "✓ Distillation process completed!"
        break
    fi

    # Show progress
    PROGRESS=$(ssh amd-hackathon "docker exec rocm bash -c 'tail -5 /home/rocm-user/AMD_Hackathon/logs/distillation_generation.log 2>/dev/null | grep Processing | tail -1'")
    echo "$(date '+%H:%M:%S') - $PROGRESS"

    sleep 300  # Check every 5 minutes
done

echo ""
echo "Checking if distillation was successful..."

# Check if output file exists
FILE_EXISTS=$(ssh amd-hackathon "docker exec rocm bash -c 'ls /home/rocm-user/AMD_Hackathon/data/distillation/distillation_3000q.json 2>/dev/null' 2>/dev/null" | wc -l)

if [ "$FILE_EXISTS" -eq "0" ]; then
    echo "❌ ERROR: Distillation output file not found!"
    echo "Check logs: ssh amd-hackathon \"docker exec rocm tail -100 /home/rocm-user/AMD_Hackathon/logs/distillation_generation.log\""
    exit 1
fi

echo "✓ Distillation data file found!"
echo ""

# Show file size
ssh amd-hackathon "docker exec rocm bash -c 'ls -lh /home/rocm-user/AMD_Hackathon/data/distillation/distillation_3000q.json'"
echo ""

echo "=========================================="
echo "🚀 STARTING ADAPTER TRAINING"
echo "=========================================="
echo ""

# Start training
ssh amd-hackathon "docker exec rocm bash -c 'cd /home/rocm-user/AMD_Hackathon && nohup python3 scripts/train_distilled_adapter.py --data /home/rocm-user/AMD_Hackathon/data/distillation/distillation_3000q.json --output /home/rocm-user/AMD_Hackathon/models/distilled_adapter_3k --epochs 2 > logs/training_distilled.log 2>&1 &'"

sleep 5

# Show initial training output
echo "Training started! Initial output:"
ssh amd-hackathon "docker exec rocm bash -c 'tail -50 /home/rocm-user/AMD_Hackathon/logs/training_distilled.log'"

echo ""
echo "=========================================="
echo "✓ Training job launched!"
echo "Monitor with:"
echo "ssh amd-hackathon \"docker exec rocm tail -f /home/rocm-user/AMD_Hackathon/logs/training_distilled.log\""
echo "=========================================="
