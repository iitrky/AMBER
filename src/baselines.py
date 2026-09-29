import numpy as np


def random_policy(sc, unread, rng):
    a = (rng.random(sc.cfg.M) < 0.5).astype(float)
    a = sc.project(a, prio=rng.random(sc.cfg.M) + 1e-3)
    rsel, snr = sc.best_reader(a)
    sel = unread & (sc.incident(a) >= sc.sens)
    sub = sc.rank_assign(sel, rng.random(sc.K), rsel)
    return a, (sub, rsel)


def allon_policy(sc, unread, rng=None):
    a = sc.project(np.ones(sc.cfg.M), prio=(sc.inc[:, unread] >= sc.sens).sum(1) + 1e-3)
    rsel, snr = sc.best_reader(a)
    sel = unread & (sc.incident(a) >= sc.sens)
    sub = sc.rank_assign(sel, snr, rsel)
    return a, (sub, rsel)


def greedy_ce(sc, unread):
    """Greedy coverage: add CE that energizes most unread tags, keeping UE constraint."""
    M = sc.cfg.M
    a = np.zeros(M)
    base = np.zeros(sc.K)
    while True:
        best, bg = -1, 0
        cov0 = ((base >= sc.sens) & unread).sum()
        for m in range(M):
            if a[m]:
                continue
            a2 = a.copy(); a2[m] = 1
            if not sc.feasible(a2):
                continue
            g = ((base + sc.inc[m] >= sc.sens) & unread).sum() - cov0
            if g > bg:
                best, bg = m, g
        if best < 0:
            return a
        a[best] = 1
        base = base + sc.inc[best]


def greedy_snr_policy(sc, unread, rng=None):
    a = greedy_ce(sc, unread)
    rsel, snr = sc.best_reader(a)
    sel = unread & (sc.incident(a) >= sc.sens)
    sub = sc.rank_assign(sel, snr, rsel)
    return a, (sub, rsel)


def greedy_sinr_policy(sc, unread, rng=None):
    """Heavy benchmark: greedy CE + sequential SINR-checked tag admission per (reader, subchannel)."""
    a = greedy_ce(sc, unread)
    rsel, snr = sc.best_reader(a)
    cand = np.where(unread & (sc.incident(a) >= sc.sens))[0]
    cand = cand[np.argsort(-snr[cand])]
    sub = -np.ones(sc.K, int)
    used = np.zeros((sc.cfg.R, sc.cfg.C), bool)
    cur = 0
    for k in cand:
        r = rsel[k]
        best_c, best_val = -1, cur
        for c in range(sc.cfg.C):
            if used[r, c]:
                continue
            sub[k] = c
            val = sc.evaluate(a, (sub, rsel), unread).sum()
            if val > best_val:
                best_c, best_val = c, val
            sub[k] = -1
        if best_c >= 0:
            sub[k] = best_c
            used[r, best_c] = True
            cur = best_val
    return a, (sub, rsel)


def upper_bound(sc, unread):
    """Relaxation bound: tags energizable under greedy coverage, capped by R*C decoding slots."""
    a = greedy_ce(sc, unread)
    n = (unread & (sc.incident(a) >= sc.sens)).sum()
    return min(n, sc.cfg.R * sc.cfg.C)
