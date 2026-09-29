#!/usr/bin/env bash
# Reproduce every table and figure of the paper using the trained models in models/.
# Runtime: roughly 30-45 minutes on one CPU core.
set -e
cd "$(dirname "$0")/../src"
python exp.py e1                                   # Table II, Fig. 4
for K in 100 200 300 500 800; do python exp.py e2 $K; done   # Table III, Fig. 5
for I in -68 -65 -62 -59 -56; do python exp.py e3 $I; done   # Fig. 6
python exp.py e4                                   # Table IV, Fig. 7
python paired.py                                   # paired significance test (Sec. VI-B)
python figs.py                                     # regenerate all figures into figures/
echo "Done. Results in results/, figures in figures/."
