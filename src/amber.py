"""AMBER: two-stage heterogeneous graph policy for Ambient IoT carrier scheduling and tag grouping.
Stage 1: carrier-emitter (CE) activation (Bernoulli) + feasibility projection layer.
Stage 2: tag admission (Bernoulli) conditioned on realised CE set, then per-reader subchannel ranking.
Trained with self-critical policy gradient (greedy baseline).
"""
import autograd.numpy as anp
import numpy as np
from autograd import grad
from env import DB

H = 32
KNN = 8


def relu(x):
    return anp.maximum(x, 0.0)


def init_params(rng, graph=True, stage2=True):
    def W(i, o):
        return rng.normal(0, np.sqrt(2.0 / i), (i, o))
    p = {
        "c_in": W(7, H), "t_in": W(5, H),
        "t2_in": W(5 + 4, H),
        "vc": W(H, 1) * 0.1, "vt": W(H, 1) * 0.1, "bc": np.zeros(1), "bt": np.zeros(1),
    }
    for l in range(2):
        p[f"ca{l}"], p[f"cb{l}"] = W(H, H), W(H, H)
        p[f"ta{l}"], p[f"tb{l}"], p[f"te{l}"] = W(H, H), W(H, H), W(H, H)
    return p


FLAGS = {"graph": True, "stage2": True}


# ------------------------- features -------------------------
def graph_inputs(sc, unread):
    cfg, K, M = sc.cfg, sc.K, sc.cfg.M
    indiv = np.array([np.all(sc.ue_int[m] <= sc.I_max) for m in range(M)])
    inc_u = sc.inc * unread[None]
    cover = (sc.inc >= sc.sens) & unread[None]
    strongest = np.zeros(M)
    if unread.any():
        np.add.at(strongest, np.argmax(sc.inc[:, unread], 0), 1)
    n_u = max(unread.sum(), 1)
    Xc = np.stack([
        sc.ce[:, 0] / cfg.area, sc.ce[:, 1] / cfg.area,
        np.clip(DB(sc.ue_int.max(1) / sc.I_max) / 20, -2, 2),
        (DB((sc.leak).max(1)) - DB(sc.n0)) / 40,
        cover.sum(1) / n_u, strongest / n_u, indiv.astype(float)], 1)
    aall = indiv.astype(float)
    inc_all = aall @ sc.inc
    S = inc_all[:, None] * sc.btr
    snr = (S / ((aall @ sc.leak)[None] + sc.n0)).max(1)
    Xt = np.stack([unread.astype(float), sc.tag[:, 0] / cfg.area, sc.tag[:, 1] / cfg.area,
                   np.clip((DB(inc_all) - cfg.P_sens_dbm) / 20, -3, 3),
                   np.clip(DB(snr) / 20, -3, 3)], 1)
    # CE-tag edges: share of incident power (unread tags only)
    E = inc_u / (inc_u.sum(0, keepdims=True) + 1e-30)          # M x K
    Ec = E / (E.sum(1, keepdims=True) + 1e-9)                    # row-normalised for CE aggregation
    # tag-tag kNN (row-normalised, only unread neighbours matter via features)
    d = np.linalg.norm(sc.tag[:, None] - sc.tag[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, 1)[:, :KNN]
    return dict(Xc=Xc, Xt=Xt, E=E, Ec=Ec, nn=nn, indiv=indiv)


def stage2_feats(sc, a, unread, gi):
    cfg = sc.cfg
    inc = sc.incident(a)
    S = inc[:, None] * sc.btr
    leak = a @ sc.leak
    snr = S / (leak[None] + sc.n0)
    r = np.argmax(snr, 1)
    s = snr[np.arange(sc.K), r]
    en = ((inc >= sc.sens) & unread).astype(float)
    # local contention: energised unread tags sharing the same best reader
    cnt = np.bincount(r, weights=en, minlength=cfg.R)[r]
    X2 = np.concatenate([gi["Xt"], np.stack([
        en, np.clip((DB(inc) - cfg.P_sens_dbm) / 20, -3, 3),
        np.clip(DB(s) / 20, -3, 3), cnt / (cfg.C * 4)], 1)], 1)
    return X2, r, s, en


# ------------------------- network -------------------------
def encode(p, Xc, Xt, E, Ec, nn, tin="t_in"):
    hc = relu(anp.dot(Xc, p["c_in"]))
    ht = relu(anp.dot(Xt, p[tin]))
    if not FLAGS["graph"]:
        return hc, ht
    for l in range(2):
        mc = anp.dot(Ec, ht)                     # tags -> CE
        mt = anp.dot(E.T, hc)                    # CE -> tags
        mn = anp.mean(ht[nn], axis=1)            # tag <-> tag (interference neighbourhood)
        hc = relu(anp.dot(hc, p[f"ca{l}"]) + anp.dot(mc, p[f"cb{l}"])) + hc
        ht = relu(anp.dot(ht, p[f"ta{l}"]) + anp.dot(mt, p[f"tb{l}"]) + anp.dot(mn, p[f"te{l}"])) + ht
    return hc, ht


def ce_logits(p, gi):
    hc, _ = encode(p, gi["Xc"], gi["Xt"], gi["E"], gi["Ec"], gi["nn"])
    return (anp.dot(hc, p["vc"]) + p["bc"])[:, 0]


def tag_logits(p, gi, X2):
    if FLAGS["stage2"]:
        _, ht = encode(p, gi["Xc"], X2, gi["E"], gi["Ec"], gi["nn"], tin="t2_in")
    else:
        X2m = anp.concatenate([gi["Xt"], np.zeros((len(X2), 4))], 1)
        _, ht = encode(p, gi["Xc"], X2m, gi["E"], gi["Ec"], gi["nn"], tin="t2_in")
    return (anp.dot(ht, p["vt"]) + p["bt"])[:, 0]


def logsig(x):
    return -anp.logaddexp(0.0, -x)


# ------------------------- acting -------------------------
def rank_lists(sc, lt, r, en, rng=None):
    """Plackett-Luce top-C ordering of energised tags per reader; position = subchannel index."""
    lists = []
    sub = -np.ones(sc.K, int)
    for rr in range(sc.cfg.R):
        cand = np.where((en > 0) & (r == rr))[0]
        if len(cand) == 0:
            continue
        key = lt[cand] + (rng.gumbel(size=len(cand)) if rng is not None else 0.0)
        pick = cand[np.argsort(-key)][: sc.cfg.C]
        sub[pick] = np.arange(len(pick))
        lists.append((cand, pick))
    return sub, lists


def act(p, sc, unread, rng=None, greedy=True, project=True):
    gi = graph_inputs(sc, unread)
    lc = ce_logits(p, gi)
    pc = 1 / (1 + np.exp(-lc))
    a_raw = (pc > 0.5).astype(float) if greedy else (rng.random(len(pc)) < pc).astype(float)
    a = sc.project(a_raw, prio=pc + 1e-3) if project else a_raw
    X2, r, s, en = stage2_feats(sc, a, unread, gi)
    lt = 5.0 * np.tanh(tag_logits(p, gi, X2) / 5.0)
    sub, lists = rank_lists(sc, lt, r, en, None if greedy else rng)
    return a, (sub, r), dict(gi=gi, X2=X2, a_raw=a_raw, lists=lists, en=en)


def make_policy(p, project=True):
    return lambda sc, unread, rng=None: act(p, sc, unread, greedy=True, project=project)[:2]


def pl_logprob(lt, lists):
    lp = 0.0
    for cand, pick in lists:
        remaining = np.ones(len(cand), bool)
        pos = {k: i for i, k in enumerate(cand)}
        lc = lt[cand]
        for k in pick:
            lp = lp + lt[k] - anp.log(anp.sum(anp.exp(lc) * remaining) + 1e-12)
            remaining = remaining.copy(); remaining[pos[k]] = False
    return lp


def logprob(p, gi, X2, a_raw, lists):
    lc = ce_logits(p, gi)
    lpc = anp.sum(a_raw * logsig(lc) + (1 - a_raw) * logsig(-lc))
    lt = tag_logits(p, gi, X2)
    lt = 5.0 * anp.tanh(lt / 5.0)          # bounded scores for numerical stability
    lpt = pl_logprob(lt, lists)
    pc = 1 / (1 + anp.exp(-lc))
    ent = -anp.sum(pc * logsig(lc) + (1 - pc) * logsig(-lc))
    return lpc + lpt, ent


class Adam:
    def __init__(self, p, lr=1e-3):
        self.lr, self.t = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}

    def step(self, p, g):
        self.t += 1
        for k in self.m:
            gk = np.clip(g[k], -5, 5)
            self.m[k] = 0.9 * self.m[k] + 0.1 * gk
            self.v[k] = 0.999 * self.v[k] + 0.001 * gk ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            p[k] = p[k] + self.lr * mh / (np.sqrt(vh) + 1e-8)   # ascent
