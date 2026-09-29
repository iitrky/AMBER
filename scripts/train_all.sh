#!/usr/bin/env bash
# Retrain AMBER and its ablation variants from scratch (overwrites models/).
# Phase A: behaviour cloning from Greedy-SINR; Phase B: self-critical policy gradient.
# Both phases checkpoint every epoch/iteration, so an interrupted run resumes where it stopped.
# Runtime: about 15-20 min (full) and 10-15 min per ablation on one CPU core.
set -e
cd "$(dirname "$0")/../src"
for TAG in full nograph nostage2; do
  rm -f ../models/bc_${TAG}.pkl ../models/rl_${TAG}.pkl
  python bc.py $TAG 25          # Phase A (builds ../data/bc_data.pkl on first run)
  python train.py $TAG 775      # Phase B (argument = total iterations)
done
