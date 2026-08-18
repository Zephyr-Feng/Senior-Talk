#!/bin/bash
# 孤独 LoRA 开卡批次主控：训练 → heldout 评估 → EATD 双分片 → 汇总 → 关机
# 由 nohup 启动，进度写入 lonely_batch_progress.log
set -u
BASE=/root/autodl-tmp/senior_project
PROG=$BASE/lonely_batch_progress.log
say(){ echo "[$(date +%m-%d\ %H:%M)] $1" | tee -a $PROG; }

cd $BASE
source /root/miniconda3/etc/profile.d/conda.sh
conda activate psy-b

say "1/4 训练启动 (train_lora_lonely.py, 540 条, 预计 ~5h)"
python scripts/train_lora_lonely.py > training_output/train_lonely.log 2>&1
if ! grep -q "训练完成" training_output/train_lonely.log; then
  say "✗ 训练失败（见 training_output/train_lonely.log），批次终止，不关机"
  exit 1
fi
say "2/4 训练完成，开始 heldout 评估（60 条）"
python scripts/eval_lonely_full.py heldout > training_output/eval_heldout.log 2>&1
say "3/4 heldout 评估完成，开始 EATD 全量双分片"
python scripts/eval_lonely_full.py eatd 0 2 > training_output/eval_eatd_shard0.log 2>&1 &
P1=$!
python scripts/eval_lonely_full.py eatd 1 2 > training_output/eval_eatd_shard1.log 2>&1 &
P2=$!
wait $P1 $P2
say "4/4 EATD 评估完成，汇总结果"
python scripts/summarize_lonely.py \
  training_output/eval_lonely/eval_lonely_eatd_shard0.jsonl \
  training_output/eval_lonely/eval_lonely_eatd_shard1.jsonl \
  --out training_output/eval_lonely/summary_eatd.json \
  > training_output/eval_lonely/final_report.txt 2>&1
grep -E "Acc|TP=" training_output/eval_heldout.log >> training_output/eval_lonely/final_report.txt
say "批次完成，验证汇总文件后关机"
if [ -f training_output/eval_lonely/summary_eatd.json ]; then
  ls -la training_output/eval_lonely/summary_eatd.json
  shutdown now
else
  say "✗ summary_eatd.json 缺失，不关机（需人工检查）"
fi
