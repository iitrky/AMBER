"""Ambient IoT inventory environment (3GPP Rel-19/20 style, topology with separate carrier emitters).
Indoor factory, sub-1 GHz, bistatic backscatter.
"""
import numpy as np

DB = lambda x: 10 * np.log10(np.maximum(x, 1e-30))
LIN = lambda x: 10 ** (np.asarray(x) / 10)


class Config:
    area = 60.0          # m, square hall side
    fc = 0.9             # GHz
    M = 16               # carrier emitters (4x4 grid)
    R = 4                # readers
    L = 8                # legacy UEs
    C = 4                # backscatter subchannels per reader
    P_ce_dbm = 30.0      # CE transmit power
    beta_db = -6.0       # backscatter modulation / reflection loss
    eta = 0.3            # (folded into sensitivity) rectifier efficiency, informational
    P_sens_dbm = -20.0   # tag activation sensitivity (incident RF power)
    bw_hz = 15e3         # backscatter subchannel bandwidth
    nf_db = 7.0
    leak_db = -80.0      # residual CW leakage after reader suppression
    aclr_db = -45.0      # CE leakage into legacy UE band
    I_max_dbm = -62.0    # tolerable aggregate interference at legacy UEs
    gamma_db = 3.0       # decoding SINR threshold
    shadow_db = 5.7


def pathloss_db(d, fc, rng):
    d = np.maximum(d, 1.0)
    plos = np.exp(-d / 20.0)
    los = rng.random(d.shape) < plos
    pl_los = 31.84 + 21.5 * np.log10(d) + 19.0 * np.log10(fc)
    pl_nlos = 33.0 + 25.5 * np.log10(d) + 20.0 * np.log10(fc)
    pl = np.where(los, pl_los, np.maximum(pl_los, pl_nlos))
    sh = rng.normal(0, 1, d.shape) * np.where(los, 4.0, 5.7)
    return pl + sh


class Scenario:
    """One random drop: fixed geometry & large-scale gains."""

    def __init__(self, K, cfg=Config, seed=0, ce_grid=True):
        rng = np.random.default_rng(seed)
        self.cfg, self.K = cfg, K
        A = cfg.area
        g = (np.arange(4) + 0.5) * A / 4
        self.ce = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)[: cfg.M]
        self.ce += rng.normal(0, 1.5, self.ce.shape)
        r = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]) * A
        self.rd = r[: cfg.R] + rng.normal(0, 2.0, (cfg.R, 2))
        # tags clustered around shelves/work cells
        nc = 10
        centers = rng.uniform(0.05 * A, 0.95 * A, (nc, 2))
        lab = rng.integers(0, nc, K)
        self.tag = np.clip(centers[lab] + rng.normal(0, 4.0, (K, 2)), 0, A)
        self.ue = rng.uniform(0, A, (cfg.L, 2))

        def gain(a, b):
            d = np.linalg.norm(a[:, None] - b[None], axis=-1)
            return -pathloss_db(d, cfg.fc, rng)  # dB

        self.G_ct = gain(self.ce, self.tag)   # M x K
        self.G_tr = gain(self.tag, self.rd)   # K x R
        self.G_cr = gain(self.ce, self.rd)    # M x R
        self.G_cu = gain(self.ce, self.ue)    # M x L
        self.N0 = -174 + 10 * np.log10(cfg.bw_hz) + cfg.nf_db  # dBm
        # precompute linear quantities
        self.Pce = LIN(cfg.P_ce_dbm)  # mW
        self.inc = self.Pce * LIN(self.G_ct)              # M x K incident mW per CE
        self.ue_int = self.Pce * LIN(self.G_cu + cfg.aclr_db)   # M x L
        self.leak = self.Pce * LIN(self.G_cr + cfg.leak_db)     # M x R
        self.I_max = LIN(cfg.I_max_dbm)
        self.n0 = LIN(self.N0)
        self.sens = LIN(cfg.P_sens_dbm)
        self.gth = LIN(cfg.gamma_db)
        self.btr = LIN(cfg.beta_db + self.G_tr)           # K x R

    def move_ues(self, mob, speed=1.0):
        """legacy UE random-walk mobility; shadowing redrawn (slot-level dynamics of the constraint)"""
        self.ue = np.clip(self.ue + mob.normal(0, speed, self.ue.shape), 0, self.cfg.area)
        d = np.linalg.norm(self.ce[:, None] - self.ue[None], axis=-1)
        self.G_cu = -pathloss_db(d, self.cfg.fc, mob)
        self.ue_int = self.Pce * LIN(self.G_cu + self.cfg.aclr_db)

    # ---------- physics ----------
    def feasible(self, a):
        return np.all(a @ self.ue_int <= self.I_max + 1e-15)

    def project(self, a, prio=None):
        """Feasibility layer: drop active CEs with largest (interference / priority) until constraint holds."""
        a = a.copy().astype(float)
        if prio is None:
            prio = np.ones(len(a))
        while not self.feasible(a):
            I = a @ self.ue_int
            viol = np.argmax(I / self.I_max)
            share = a * self.ue_int[:, viol] / np.maximum(prio, 1e-6)
            a[np.argmax(share)] = 0.0
        return a

    def incident(self, a):
        return a @ self.inc  # K, mW

    def rx_power(self, a):
        """backscatter power at each reader: sum over CEs (bistatic, incoherent) -> K x R"""
        return self.incident(a)[:, None] * self.btr

    def evaluate(self, a, sched, unread):
        """a: M binary CE activation; sched: K int subchannel (-1 idle) with assigned reader rsel (K).
        Returns boolean success vector."""
        sub, rsel = sched
        inc = self.incident(a)
        on = (sub >= 0) & (inc >= self.sens) & unread
        S = self.rx_power(a)  # K x R
        leak = a @ self.leak  # R
        succ = np.zeros(self.K, bool)
        for c in range(self.cfg.C):
            idx = np.where(on & (sub == c))[0]
            if len(idx) == 0:
                continue
            Sc = S[idx]                     # n x R
            tot = Sc.sum(0)                 # R
            r = rsel[idx]
            sig = Sc[np.arange(len(idx)), r]
            sinr = sig / (tot[r] - sig + leak[r] + self.n0)
            succ[idx] = sinr >= self.gth
        return succ

    # ---------- helpers shared by policies ----------
    def best_reader(self, a):
        S = self.rx_power(a)
        leak = a @ self.leak
        snr = S / (leak[None] + self.n0)
        r = np.argmax(snr, 1)
        return r, snr[np.arange(self.K), r]

    def rank_assign(self, sel, score, rsel):
        """Assign selected tags to subchannels: per reader, top-C by score get subchannels 0..C-1."""
        sub = -np.ones(self.K, int)
        for r in range(self.cfg.R):
            idx = np.where(sel & (rsel == r))[0]
            if len(idx) == 0:
                continue
            idx = idx[np.argsort(-score[idx])][: self.cfg.C]
            sub[idx] = np.arange(len(idx))
        return sub


def run_episode(sc, policy, T=30, rng=None, ue_speed=1.0, seed=0):
    """Inventory round of T slots with mobile legacy UEs. Returns cumulative reads per slot, violations."""
    rng = rng or np.random.default_rng(seed)
    mob = np.random.default_rng(seed + 12345)
    unread = np.ones(sc.K, bool)
    ue0, cu0 = sc.ue.copy(), sc.ue_int.copy()
    cum, viol = [], 0
    for t in range(T):
        a, s = policy(sc, unread, rng)
        viol += int(not sc.feasible(a))
        succ = sc.evaluate(a, s, unread)
        unread &= ~succ
        cum.append(sc.K - unread.sum())
        sc.move_ues(mob, ue_speed)
    sc.ue, sc.ue_int = ue0, cu0
    return np.array(cum), viol
