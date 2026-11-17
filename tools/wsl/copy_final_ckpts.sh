#!/bin/bash
# Копия финальных чекпоинтов всех стадий в backups/final_ckpts (для ветки-бэкапа).
BK="/mnt/g/LEARN WALK/backups/final_ckpts"
PAIRS="s1_balance/run_1/model_3499.pt s2_walk/run_0/model_5499.pt s2_walk_run_a/run_0/model_5999.pt s3_run/run_0/model_5999.pt s3_sprint/run_0/model_5999.pt s4_jump/run_0/model_5999.pt s6_fall/run_0/model_11998.pt s6_fall2/run_0/model_6999.pt s8_dance/run_0/model_7999.pt s9_sit/run_2/model_3999.pt"
for p in $PAIRS; do
  stage=$(dirname "$(dirname "$p")")
  mkdir -p "$BK/$stage"
  cp "/root/hlc-logs/$p" "$BK/$stage/"
  cp "/root/hlc-logs/$(dirname "$p")/cmd.json" "$BK/$stage/" 2>/dev/null
done
cp "/mnt/g/LEARN WALK/data/motions/npz_g1/unified_segments.json" "$BK/" 2>/dev/null
du -sh "$BK"
ls "$BK"
echo "COPY_EXIT=$?"