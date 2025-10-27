#!/bin/bash
# Disk and Training Monitor - Prevents disk full crashes
# Run this in background while training runs

ALERT_THRESHOLD=85  # Alert if disk usage exceeds 85%
CHECK_INTERVAL=300  # Check every 5 minutes

while true; do
    # Get disk usage percentage
    DISK_USAGE=$(ssh amd-hackathon "df -h | grep '/dev/vda1' | awk '{print \$5}' | sed 's/%//'")

    # Get current timestamp
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Check if training is still running
    TRAINING_RUNNING=$(ssh amd-hackathon "docker exec rocm ps aux | grep 'train_qwen2.5_unsloth.py' | grep -v grep | wc -l")

    # Get latest training progress
    LATEST_PROGRESS=$(ssh amd-hackathon "docker exec rocm tail -5 /home/rocm-user/AMD_Hackathon/training_fresh.log | grep -E 'Chunk|%' | tail -1")

    echo "[$TIMESTAMP] Disk: ${DISK_USAGE}% | Training: $([[ $TRAINING_RUNNING -gt 0 ]] && echo 'RUNNING' || echo 'STOPPED') | Progress: $LATEST_PROGRESS"

    # Alert if disk usage is high
    if [ "$DISK_USAGE" -gt "$ALERT_THRESHOLD" ]; then
        echo "⚠️  WARNING: Disk usage at ${DISK_USAGE}%! Checking for large files..."

        # Find large HuggingFace cache files
        ssh amd-hackathon "docker exec rocm du -sh /root/.cache/huggingface 2>/dev/null || echo 'Cache check failed'"

        # If disk is critically full (>95%), auto-clean cache
        if [ "$DISK_USAGE" -gt 95 ]; then
            echo "🚨 CRITICAL: Disk at ${DISK_USAGE}%! Auto-cleaning HuggingFace cache..."
            ssh amd-hackathon "rm -rf /var/lib/docker/overlay2/*/diff/root/.cache/huggingface/hub/*"
            echo "✅ Cache cleared"
        fi
    fi

    # Check if training has stopped unexpectedly
    if [ "$TRAINING_RUNNING" -eq 0 ]; then
        echo "❌ WARNING: Training process stopped! Check logs."
        # Exit monitoring loop
        break
    fi

    sleep $CHECK_INTERVAL
done

echo "Monitoring stopped at $(date)"
