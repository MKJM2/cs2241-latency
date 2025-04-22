#!/usr/bin/env sh
python fast_plbf.py \
  --data_path data/processed_urls.csv \
  --N 1000 \
  --k 10 \
  --F 0.01 \
  --plbf_test_split 0.2 \
  --seed 42 \
  --use_fast_dp
