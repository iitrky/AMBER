"""Core correctness tests. Run from the repository root:  python -m pytest tests  (or: python tests/test_core.py)"""
import os, sys
import numpy as np
SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, SRC)
os.chdir(SRC)

from env import Scenario
import amber as A
from baselines import greedy_sinr_policy, greedy_snr_policy, random_policy


def test_projection_always_feasible():
    """Proposition 1: the projection layer returns a feasible CE set for any candidate."""
    rng = np.random.default_rng(0)
    for s in range(30):
        sc = Scenario(100, seed=s)
        for _ in range(20):
            a_hat = (rng.random(sc.cfg.M) < rng.random()).astype(float)
            a = sc.project(a_hat, prio=rng.random(sc.cfg.M) + 1e-3)
            assert sc.feasible(a)
            assert np.all(a <= a_hat)          # projection only switches emitters off


def test_reads_bounded_by_capacity():
    """At most one decodable tag per (reader, subchannel): reads per slot <= R*C."""
    sc = Scenario(300, seed=1)
    unread = np.ones(sc.K, bool)
    for pol in (greedy_snr_policy, greedy_sinr_policy):
        a, s = pol(sc, unread)
        assert sc.evaluate(a, s, unread).sum() <= sc.cfg.R * sc.cfg.C


def test_read_tags_are_energised_and_scheduled():
    sc = Scenario(200, seed=2)
    unread = np.ones(sc.K, bool)
    a, (sub, r) = random_policy(sc, unread, np.random.default_rng(0))
    succ = sc.evaluate(a, (sub, r), unread)
    assert np.all(sc.incident(a)[succ] >= sc.sens)
    assert np.all(sub[succ] >= 0)


def test_trained_amber_acts_feasibly():
    import pickle
    p = pickle.load(open('../models/rl_full.pkl', 'rb'))['p']
    A.FLAGS.update(graph=True, stage2=True)
    sc = Scenario(150, seed=3)
    unread = np.ones(sc.K, bool)
    a, s, info = A.act(p, sc, unread, np.random.default_rng(0), greedy=False)
    assert sc.feasible(a)
    lp, ent = A.logprob(p, info['gi'], info['X2'], info['a_raw'], info['lists'])
    assert np.isfinite(lp) and np.isfinite(ent)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('PASS', name)
