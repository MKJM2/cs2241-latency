#!/usr/bin/env sh
python lbf.py \
  --data_path data/processed_urls.csv \
  --target_fpr 0.01 \
  --lbf_neg_split 0.7 \
  --neg_val_split 0.5 \
  --seed 42
