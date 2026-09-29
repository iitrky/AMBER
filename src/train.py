"""Stage B: self-critical policy-gradient fine-tuning (resumable)."""
import numpy as np, time, pickle, sys, os
from autograd import grad
from env import Scenario
import amber as A
from baselines import greedy_snr_policy

FL = dict(full=dict(graph=True, stage2=True), nograph=dict(graph=False, stage2=True), nostage2=dict(graph=True, stage2=False))

def run(tag, iters, B=6, T=12, lr=5e-4, ent_w=0.005):
    A.FLAGS.update(FL[tag]); ck = f'../models/rl_{tag}.pkl'
    if os.path.exists(ck):
        st = pickle.load(open(ck, 'rb'))
    else:
        bcst = pickle.load(open(f'../models/bc_{tag}.pkl', 'rb'))
        st = dict(p=bcst['p'], opt=A.Adam(bcst['p'], lr), it=0, log=[])
    p, opt = st['p'], st['opt']
    g_obj = grad(lambda th, gi, X2, ar, li, adv: adv * A.logprob(th, gi, X2, ar, li)[0] + ent_w * A.logprob(th, gi, X2, ar, li)[1])
    rng = np.random.default_rng(st['it'] + 17); t0 = time.time()
    for it in range(st['it'], iters):
        envs = []
        for b in range(B):
            s = int(rng.integers(1_000, 1_000_000))
            sc = Scenario(int(rng.choice([100, 200, 300])), seed=s); mob = np.random.default_rng(s + 7)
            unread = np.ones(sc.K, bool)
            for _ in range(int(rng.integers(0, 14))):   # diverse inventory stages
                a, s_ = greedy_snr_policy(sc, unread); unread &= ~sc.evaluate(a, s_, unread); sc.move_ues(mob)
            envs.append((sc, unread, mob))
        G = {k: np.zeros_like(v) for k, v in p.items()}; R1 = R0 = 0
        for t in range(T):
            for sc, unread, mob in envs:
                a, s_, info = A.act(p, sc, unread, rng, greedy=False)
                succ = sc.evaluate(a, s_, unread)
                ag, sg, _ = A.act(p, sc, unread, greedy=True)
                rb = sc.evaluate(ag, sg, unread).sum()
                adv = (succ.sum() - rb) / (sc.cfg.R * sc.cfg.C)
                R1 += succ.sum(); R0 += rb
                if adv != 0:
                    g = g_obj(p, info['gi'], info['X2'], info['a_raw'], info['lists'], adv)
                    for k in G: G[k] += g[k] / (B * T)
                unread &= ~succ; sc.move_ues(mob)
        opt.step(p, G); st['log'].append((R1 / (B * T), R0 / (B * T)))
        if it % 5 == 0:
            print(tag, it, f'sample {R1/(B*T):.2f} greedy-self {R0/(B*T):.2f} {time.time()-t0:.0f}s', flush=True)
        st['it'] = it + 1
        pickle.dump(st, open(ck, 'wb'))

if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]))
