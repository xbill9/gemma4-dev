#!/bin/bash
# Two full passes, alternating the models.
cd "$(dirname "$0")"
for round in 1 2; do
  for m in 2b 4b; do
    python3 profile_depth.py $m profile-$m.jsonl 2>>profile-$m.err
  done
done
