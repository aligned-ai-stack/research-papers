"""Regenerate public figures from aggregate data; no participant inputs needed."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def render(data,output):
    output.mkdir(parents=True,exist_ok=True)
    accuracy=pd.read_csv(data/'accuracy.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,conditions in zip(axes,[['near_miss','overbreadth','out_of_scope'],['no_error']]):
        for j,condition in enumerate(['explainable','evaluative']):
            subset=accuracy[(accuracy.condition==condition)&accuracy.trial_type.isin(conditions)]
            if len(conditions)>1:
                n=subset.n_trials.sum(); rate=subset.n_correct.sum()/n
            else:n=int(subset.n_trials.iloc[0]);rate=subset.accuracy.iloc[0]
            z=1.95996398454
            center=(rate+z*z/(2*n))/(1+z*z/n)
            half=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
            ax.bar(j,rate,color=['#b66c4b','#397f99'][j])
            ax.vlines(j,center-half,center+half,color='#222',linewidth=1.6)
            ax.text(j,center+half+.025,f'{rate:.1%}',ha='center')
        ax.set_xticks([0,1],['XAI','EvalAI']);ax.set_ylim(0,1.08)
        ax.spines[['top','right']].set_visible(False)
        ax.set_title('AI error trials' if len(conditions)>1 else 'Correct AI trials')
        ax.set_ylabel('Proportion correct (95% Wilson interval)')
    fig.suptitle('Provision accuracy · 150 participants per AI condition')
    fig.tight_layout();fig.savefig(output/'accuracy.png',dpi=180);plt.close(fig)
    t=pd.read_csv(data/'models/H3b_timing_specifications.csv')
    fig,ax=plt.subplots(figsize=(9,3.8))
    ax.hlines(range(len(t)),t.ci_low,t.ci_high,color='#397f99',linewidth=2)
    ax.scatter(t.coef,range(len(t)),color='#153c50',zorder=3)
    ax.axvline(0,color='#777',linestyle='--',linewidth=.8)
    ax.set_yticks(range(len(t)),['Reported: winsorized + case controls','Companion: raw, no case controls','Companion: raw + case controls'])
    ax.invert_yaxis();ax.set_xlabel('EvalAI − XAI coefficient on log decision time (95% interval)')
    ax.set_title('Timing conclusions depend on the specification')
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(output/'timing_specifications.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('data'))
    p.add_argument('--output',type=Path,default=Path('figures'));a=p.parse_args();render(a.data,a.output)
