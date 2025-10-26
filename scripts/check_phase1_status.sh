#!/bin/bash
# Quick status check for Phase 1 data generation

echo "===== PHASE 1 STATUS ====="
echo ""

# Check if process is running
RUNNING=$(ssh amd-hackathon "docker exec rocm pgrep -f generate_synthetic_data" 2>/dev/null)

if [ -n "$RUNNING" ]; then
    echo "✅ Process running (PID: $RUNNING)"
    echo ""

    # GPU usage
    echo "GPU Usage:"
    ssh amd-hackathon "docker exec rocm rocm-smi" | grep -E "GPU%|VRAM%" | tail -1
    echo ""

    # Recent progress
    echo "Recent Progress:"
    ssh amd-hackathon "docker exec rocm tail -20 /home/rocm-user/AMD_Hackathon/logs/phase1_generation.log" | grep -E "Generating|questions|Validation|Total|Saved"
    echo ""

    # Check for output files
    RAW_EXISTS=$(ssh amd-hackathon "docker exec rocm test -f /home/rocm-user/AMD_Hackathon/data/synthetic/raw_questions.json && echo yes || echo no")
    VALIDATED_EXISTS=$(ssh amd-hackathon "docker exec rocm test -f /home/rocm-user/AMD_Hackathon/data/synthetic/validated_questions.json && echo yes || echo no")

    if [ "$RAW_EXISTS" = "yes" ]; then
        echo "✅ Raw questions file exists"
    fi

    if [ "$VALIDATED_EXISTS" = "yes" ]; then
        echo "✅ Validated questions file exists - PHASE 1 COMPLETE!"
    fi
else
    echo "❌ Process not running"
    echo ""

    # Check if completed
    VALIDATED_EXISTS=$(ssh amd-hackathon "docker exec rocm test -f /home/rocm-user/AMD_Hackathon/data/synthetic/validated_questions.json && echo yes || echo no")

    if [ "$VALIDATED_EXISTS" = "yes" ]; then
        echo "🎉 Phase 1 completed successfully!"
    else
        echo "⚠️ Process stopped - checking last log entries..."
        echo ""
        ssh amd-hackathon "docker exec rocm tail -30 /home/rocm-user/AMD_Hackathon/logs/phase1_generation.log"
    fi
fi

echo ""
echo "======================="
