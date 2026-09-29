import numpy as np, pickle
import amber as A
from baselines import greedy_ce
def sinr_admit(sc, a, unread, order_score=None):
    rsel, snr = sc.best_reader(a)
    cand = np.where(unread & (sc.incident(a) >= sc.sens))[0]
    sc_ = snr if order_score is None else order_score
    cand = cand[np.argsort(-sc_[cand])]
    sub = -np.ones(sc.K, int); used = np.zeros((sc.cfg.R, sc.cfg.C), bool); cur = 0
    for k in cand:
        r = rsel[k]; best_c, best_val = -1, cur
        if used[r].all(): continue
        for c in range(sc.cfg.C):
            if used[r, c]: continue
            sub[k] = c; val = sc.evaluate(a, (sub, rsel), unread).sum()
            if val > best_val: best_c, best_val = c, val
            sub[k] = -1
        if best_c >= 0: sub[k] = best_c; used[r, best_c] = True; cur = best_val
    return sub, rsel
def make_hybrid(p, tag_order=True, flags=dict(graph=True, stage2=True), project=True):
    def f(sc, unread, rng=None):
        A.FLAGS.update(flags)
        gi = A.graph_inputs(sc, unread); lc = A.ce_logits(p, gi); pc = 1/(1+np.exp(-lc))
        a = (pc > 0.5).astype(float)
        if project: a = sc.project(a, prio=pc+1e-3)
        os_ = None
        if tag_order:
            X2, r, s, en = A.stage2_feats(sc, a, unread, gi); os_ = A.tag_logits(p, gi, X2)
        return a, sinr_admit(sc, a, unread, os_)
    return f
