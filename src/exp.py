import numpy as np, pickle, json, sys, time
from env import Scenario, run_episode, LIN
import amber as A
from baselines import *
from hybrid import make_hybrid

FL = dict(full=dict(graph=True, stage2=True), nograph=dict(graph=False, stage2=True), nostage2=dict(graph=True, stage2=False))
P = {}
def pol(tag, src='rl', project=True):
    key = (tag, src)
    if key not in P: P[key] = pickle.load(open(f'../models/{src}_{tag}.pkl', 'rb'))['p']
    p = P[key]
    def f(sc, unread, rng=None):
        A.FLAGS.update(FL[tag]); return A.act(p, sc, unread, greedy=True, project=project)[:2]
    return f

def amber(tag='full', src='rl', project=True):
    return make_hybrid(pickle.load(open(f'../models/{src}_{tag}.pkl', 'rb'))['p'], True, FL[tag], project)

def methods():
    return {'Random': random_policy, 'All-on': allon_policy, 'Greedy-SNR': greedy_snr_policy,
            'Greedy-SINR': greedy_sinr_policy, 'AMBER (BC only)': amber('full', 'bc'), 'AMBER': amber()}

def episode(method, K, seed, T, I_max=None):
    sc = Scenario(K, seed=2_000_000 + seed)
    if I_max is not None: sc.I_max = LIN(I_max)
    times = []
    def timed(sc_, u, rng=None):
        t = time.perf_counter(); out = method(sc_, u, rng); times.append(time.perf_counter() - t); return out
    c, v = run_episode(sc, timed, T=T, rng=np.random.default_rng(seed), seed=seed)
    return c.tolist(), v, float(np.mean(times))

def save(name, obj): json.dump(obj, open(f'../results/{name}.json', 'w'))

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'e1':   # cumulative reads, K=200
        out = {}
        for n, m in methods().items():
            rs = [episode(m, 200, s, 30) for s in range(20)]
            out[n] = dict(curve=np.mean([r[0] for r in rs], 0).tolist(), std=np.std([r[0] for r in rs], 0).tolist(),
                          viol=int(sum(r[1] for r in rs)), t=float(np.mean([r[2] for r in rs])))
            print(n, np.round(out[n]['curve'][9]), np.round(out[n]['curve'][-1], 1), flush=True)
        save('e1', out)
    elif mode == 'e2':   # scalability, one K per call
        K = int(sys.argv[2]); T = int(np.ceil(1.5 * K / 16)); ns = 8 if K <= 300 else 4
        out = {}
        M_ = methods(); M_['AMBER-Lite'] = amber('nograph')
        for n in ['Greedy-SNR', 'Greedy-SINR', 'AMBER', 'AMBER-Lite']:
            m = M_[n]; rs = [episode(m, K, s, T) for s in range(ns)]
            out[n] = dict(final=float(np.mean([r[0][-1] for r in rs]) / K),
                          auc=float(np.mean([np.mean(r[0]) for r in rs]) / K), t=float(np.mean([r[2] for r in rs])))
            print(K, n, out[n], flush=True)
        save(f'e2_{K}', out)
    elif mode == 'e3':   # interference threshold sweep
        I = float(sys.argv[2]); out = {}
        for n in ['All-on', 'Greedy-SNR', 'Greedy-SINR', 'AMBER']:
            rs = [episode(methods()[n], 200, s, 30, I_max=I) for s in range(10)]
            out[n] = dict(r10=float(np.mean([r[0][9] for r in rs])), r30=float(np.mean([r[0][-1] for r in rs])))
            print(I, n, out[n], flush=True)
        save(f'e3_{int(I)}', out)
    elif mode == 'e4':   # ablations
        V = {'AMBER (full)': amber(), 'w/o graph message passing': amber('nograph'),
             'w/o stage-2 conditioning': amber('nostage2'), 'w/o RL fine-tuning (BC only)': amber('full', 'bc'),
             'w/o SINR-certified decoder': pol('full'), 'w/o feasibility layer': amber(project=False)}
        out = {}
        for n, m in V.items():
            rs = [episode(m, 200, s, 30) for s in range(12)]
            out[n] = dict(r10=float(np.mean([r[0][9] for r in rs])), r20=float(np.mean([r[0][19] for r in rs])),
                          r30=float(np.mean([r[0][-1] for r in rs])), viol=float(sum(r[1] for r in rs)) / (12 * 30))
            print(n, out[n], flush=True)
        save('e4', out)
