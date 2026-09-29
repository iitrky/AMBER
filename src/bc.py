"""Stage A: teacher-guided pre-training (behaviour cloning from Greedy-SINR). Resumable."""
import numpy as np, pickle, time, os, sys
from autograd import grad
import autograd.numpy as anp
from env import Scenario
import amber as A
from baselines import greedy_sinr_policy

def make_data(n_ep=50, T=15, seed=0, fn='../data/bc_data.pkl'):
    rng = np.random.default_rng(seed); D = []
    for e in range(n_ep):
        s = int(rng.integers(1_000, 1_000_000)); sc = Scenario(int(rng.choice([100,200,300])), seed=s)
        mob = np.random.default_rng(s+7); unread = np.ones(sc.K, bool)
        for t in range(T):
            a, (sub, r) = greedy_sinr_policy(sc, unread)
            gi = A.graph_inputs(sc, unread)
            X2, r2, s2, en = A.stage2_feats(sc, a, unread, gi)
            lists = []
            for rr in range(sc.cfg.R):
                cand = np.where((en > 0) & (r2 == rr))[0]
                pick = cand[sub[cand] >= 0]; pick = pick[np.argsort(sub[pick])]
                if len(cand) and len(pick): lists.append((cand, pick))
            D.append(dict(gi=gi, X2=X2, a=a, lists=lists, en=en))
            unread &= ~sc.evaluate(a, (sub, r), unread); sc.move_ues(mob)
    pickle.dump(D, open(fn, 'wb')); return D

def bc_loss(p, d):
    lc = A.ce_logits(p, d['gi']); lt = A.tag_logits(p, d['gi'], d['X2'])
    l1 = -anp.mean(d['a']*A.logsig(lc) + (1-d['a'])*A.logsig(-lc))
    lt = 5.0*anp.tanh(lt/5.0)
    l2 = -A.pl_logprob(lt, d['lists']) / (sum(len(pk) for _, pk in d['lists']) + 1)
    return l1 + l2

if __name__ == '__main__':
    tag = sys.argv[1]; epochs = int(sys.argv[2])
    A.FLAGS.update(dict(full=dict(graph=True, stage2=True), nograph=dict(graph=False, stage2=True),
                        nostage2=dict(graph=True, stage2=False))[tag])
    D = pickle.load(open('../data/bc_data.pkl','rb')) if os.path.exists('../data/bc_data.pkl') else make_data()
    ck = f'../models/bc_{tag}.pkl'
    if os.path.exists(ck):
        st = pickle.load(open(ck,'rb')); p, opt, ep0, hist = st['p'], st['opt'], st['ep'], st['hist']
    else:
        p = A.init_params(np.random.default_rng(0)); opt = A.Adam(p, 3e-3); ep0 = 0; hist = []
    g = grad(lambda th, d: -bc_loss(th, d))   # Adam does ascent
    rng = np.random.default_rng(ep0); t0 = time.time()
    for ep in range(ep0, ep0+epochs):
        idx = rng.permutation(len(D)); L = []
        for i in range(0, len(idx), 16):
            G = None
            for j in idx[i:i+16]:
                gj = g(p, D[j])
                G = gj if G is None else {k: G[k]+gj[k] for k in G}
            opt.step(p, {k: v/16 for k, v in G.items()})
        L = np.mean([bc_loss(p, D[j]) for j in idx[:100]]); hist.append(L)
        print(tag, 'epoch', ep, 'loss', round(float(L),4), f'{time.time()-t0:.0f}s', flush=True)
        pickle.dump(dict(p=p, opt=opt, ep=ep+1, hist=hist), open(ck,'wb'))
