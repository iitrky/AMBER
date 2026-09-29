"""Quick start: run one 30-slot inventory round with AMBER and the Greedy-SINR benchmark.
Usage (from the repository root):  python examples/quickstart.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'src'))

from env import Scenario, run_episode
from baselines import greedy_sinr_policy
from exp import amber

K, T, SEED = 200, 30, 7
for name, policy in [('Greedy-SINR', greedy_sinr_policy), ('AMBER', amber())]:
    sc = Scenario(K, seed=5_000_000 + SEED)          # unseen test layout
    cum, viol = run_episode(sc, policy, T=T, seed=SEED)
    print(f'{name:12s} tags read after 10/20/30 slots: {cum[9]:3d} / {cum[19]:3d} / {cum[29]:3d}'
          f'   interference violations: {viol}')
