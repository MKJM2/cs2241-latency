#!/usr/bin/env sh
python bloom_filter.py \
  --data_path data/processed_urls.csv \
  --error_rate 0.01 \
  --test_split 0.2 \
  --seed 42
