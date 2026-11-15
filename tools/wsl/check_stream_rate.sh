#!/bin/bash
# скорость стрима: последние эпизоды + время кадра
tail -6 /root/live_stream.out | grep -E "эпизод|policy|env готов" || true
echo "---"
ls /root/hlc-logs/s2_walk_run_a/run_0/TRAIN_DONE 2>/dev/null && echo RUN_A_DONE
ls /root/hlc-logs/s3_run 2>/dev/null || echo "s3_run ещё нет"
