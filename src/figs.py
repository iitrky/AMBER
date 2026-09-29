import numpy as np, json, pickle, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from env import Scenario
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.grid':True,'grid.alpha':0.3,'savefig.dpi':220,'savefig.bbox':'tight'})
R=lambda n: json.load(open(f'../results/{n}.json'))
sty={'Random':('0.6',':','x'),'All-on':('tab:purple','-.','v'),'Greedy-SNR':('tab:orange','--','s'),
     'Greedy-SINR':('tab:green','--','^'),'AMBER (BC only)':('tab:gray','-','d'),'AMBER':('tab:red','-','o'),'AMBER-Lite':('tab:blue','-','D')}
# Fig1 layout
sc=Scenario(200,seed=2_000_003); f,ax=plt.subplots(figsize=(4.6,4.2))
ax.scatter(*sc.tag.T,s=8,c='0.55',label='A-IoT tags')
ax.scatter(*sc.ce.T,s=70,marker='^',c='tab:red',label='Carrier emitters')
ax.scatter(*sc.rd.T,s=90,marker='s',c='tab:blue',label='Readers')
ax.scatter(*sc.ue.T,s=60,marker='*',c='tab:green',label='Legacy UEs')
ax.set_xlim(0,60);ax.set_ylim(0,60);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.legend(fontsize=7,loc='upper right',framealpha=0.9)
plt.savefig('../figures/fig1_layout.png');plt.close()
# Fig2 architecture
f,ax=plt.subplots(figsize=(7.2,2.6));ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,3)
B=[(0.1,1.1,'Heterogeneous\ngraph state\n(CE, tag, UE, reader)'),(2.0,1.1,'Stage 1\nGraph encoder\nCE activation\n(Bernoulli)'),
   (3.9,1.1,'Feasibility\nprojection layer\n(legacy UE\ninterference)'),(5.8,1.1,'Stage 2\nConditioned tag\npriority scores\n(Plackett-Luce)'),
   (7.7,1.1,'SINR-certified\nsubchannel\nadmission')]
for x,y,t in B:
    ax.add_patch(FancyBboxPatch((x,y),1.7,1.3,boxstyle='round,pad=0.05',fc='#eef3fb',ec='tab:blue'));ax.text(x+0.85,y+0.65,t,ha='center',va='center',fontsize=7.5)
for i in range(4): ax.annotate('',xy=(B[i+1][0],1.75),xytext=(B[i][0]+1.72,1.75),arrowprops=dict(arrowstyle='->'))
ax.annotate('',xy=(5.8,0.95),xytext=(3.4,0.95),arrowprops=dict(arrowstyle='->',ls='--',color='0.4'));ax.text(4.6,0.65,'realised CE set a',ha='center',fontsize=7)
ax.text(9.75,1.75,'→ reads',fontsize=7.5,va='center')
ax.text(5,2.75,'Training: behaviour cloning from Greedy-SINR, then self-critical policy gradient',ha='center',fontsize=8,style='italic')
plt.savefig('../figures/fig2_architecture.png');plt.close()
# Fig3 convergence
bc=pickle.load(open('../models/bc_full.pkl','rb'))['hist'];rl=np.array(pickle.load(open('../models/rl_full.pkl','rb'))['log'])
f,(a1,a2)=plt.subplots(1,2,figsize=(7.2,2.7))
a1.plot(np.arange(1,len(bc)+1),bc,'o-',ms=3);a1.set_xlabel('Pre-training epoch');a1.set_ylabel('Imitation loss');a1.set_title('(a) Stage A',fontsize=9)
ma=lambda x,w=25: np.convolve(x,np.ones(w)/w,'valid')
a2.plot(ma(rl[:,0]-rl[:,1]),c='tab:red');a2.axhline(0,c='k',lw=0.8);a2.set_xlabel('Policy-gradient iteration');a2.set_ylabel('Sampled − greedy reward\n(reads/slot, 25-it MA)');a2.set_title('(b) Stage B',fontsize=9)
plt.tight_layout();plt.savefig('../figures/fig3_convergence.png');plt.close()
# Fig4 cumulative
e1=R('e1');f,ax=plt.subplots(figsize=(4.8,3.4));t=np.arange(1,31)
for n,v in e1.items():
    c,l,m=sty[n];ax.plot(t,v['curve'],color=c,ls=l,marker=m,markevery=3,ms=4,label=n)
ax.set_xlabel('Time slot');ax.set_ylabel('Cumulative tags read (K = 200)');ax.legend(fontsize=7,loc='upper left')
ins=ax.inset_axes([0.56,0.1,0.4,0.32])
for n in ['Greedy-SNR','Greedy-SINR','AMBER (BC only)','AMBER']:
    c,l,m=sty[n];ins.plot(t[19:],e1[n]['curve'][19:],color=c,ls=l,marker=m,ms=3)
ins.tick_params(labelsize=6)
plt.savefig('../figures/fig4_cumulative.png');plt.close()
# Fig5 scalability
Ks=[100,200,300,500,800];E={K:R(f'e2_{K}') for K in Ks}
f,(a1,a2)=plt.subplots(1,2,figsize=(7.2,2.9))
for n in ['Greedy-SNR','Greedy-SINR','AMBER-Lite','AMBER']:
    c,l,m=sty[n];a1.plot(Ks,[E[K][n]['auc'] for K in Ks],color=c,ls=l,marker=m,label=n)
    a2.plot(Ks,[1e3*E[K][n]['t'] for K in Ks],color=c,ls=l,marker=m,label=n)
a1.set_xlabel('Number of tags K');a1.set_ylabel('Normalised inventory AUC');a1.set_title('(a) Inventory speed',fontsize=9);a1.legend(fontsize=7)
a2.set_xlabel('Number of tags K');a2.set_ylabel('Decision time per slot (ms)');a2.set_yscale('log');a2.set_title('(b) Run time (1 CPU core)',fontsize=9)
plt.tight_layout();plt.savefig('../figures/fig5_scalability.png');plt.close()
# Fig6 Imax sweep
Is=[-68,-65,-62,-59,-56];E3={I:R(f'e3_{I}') for I in Is};f,ax=plt.subplots(figsize=(4.8,3.2))
for n in ['All-on','Greedy-SNR','Greedy-SINR','AMBER']:
    c,l,m=sty[n];ax.plot(Is,[E3[I][n]['r30'] for I in Is],color=c,ls=l,marker=m,label=n)
ax.axvline(-62,c='0.5',lw=0.8,ls=':');ax.text(-61.8,152,'training\nvalue',fontsize=7)
ax.set_xlabel('Legacy UE interference limit $I_{max}$ (dBm)');ax.set_ylabel('Tags read after 30 slots');ax.legend(fontsize=7)
plt.savefig('../figures/fig6_imax.png');plt.close()
# Fig7 ablation
e4=R('e4');names=list(e4);f,ax=plt.subplots(figsize=(5.2,3.0))
y=np.arange(len(names));ax.barh(y,[e4[n]['r30'] for n in names],color=['tab:red']+['0.6']*(len(names)-2)+['tab:orange'])
for i,n in enumerate(names): ax.text(e4[n]['r30']+0.3,i,f"{e4[n]['r30']:.1f}"+(f"  (viol. {100*e4[n]['viol']:.0f}%)" if e4[n]['viol']>0 else ''),va='center',fontsize=7)
ax.set_yticks(y);ax.set_yticklabels(names,fontsize=8);ax.invert_yaxis();ax.set_xlim(160,200);ax.set_xlabel('Tags read after 30 slots (K = 200)')
plt.savefig('../figures/fig7_ablation.png');plt.close()
print('ok')
