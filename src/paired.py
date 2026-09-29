import numpy as np, json
from scipy import stats
from exp import episode, amber
from baselines import greedy_sinr_policy
m=amber(); A=[];G=[]
for s in range(30):
    A.append(episode(m,200,100+s,30)); G.append(episode(greedy_sinr_policy,200,100+s,30))
a30=np.array([r[0][-1] for r in A]); g30=np.array([r[0][-1] for r in G])
a20=np.array([r[0][19] for r in A]); g20=np.array([r[0][19] for r in G])
d=a30-g30; t=stats.ttest_rel(a30,g30); w=stats.wilcoxon(a30,g30)
out=dict(mean_diff30=float(d.mean()),ci=float(1.96*d.std(ddof=1)/np.sqrt(30)),p_t=float(t.pvalue),p_w=float(w.pvalue),
  wins=int((d>0).sum()),ties=int((d==0).sum()),mean_diff20=float((a20-g20).mean()),p20=float(stats.ttest_rel(a20,g20).pvalue),
  a30=float(a30.mean()),g30=float(g30.mean()))
print(out); json.dump(out,open('../results/paired.json','w'))
