"""Verify public summaries; optionally reconstruct panels from approved inputs."""
from pathlib import Path
import argparse, csv, json, subprocess, sys, importlib.util, collections

ROOT=Path(__file__).resolve().parent
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load_module(name,p):
    spec=importlib.util.spec_from_file_location(name,p)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reproduction',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'local/verification.json')
    args=parser.parse_args()
    f=ROOT/'analysis/data/followup_outputs'
    s=json.loads((ROOT/'data/study-summary.json').read_text())
    assert s['trials']==7200 and s['participants']==240
    assert s['zero_update_trials']==4848 and s['absolute_update_below_5_percentage_points_trials']==5500
    for name in ['canonical_aicc_winner_counts.csv','canonical_bic_winner_counts.csv','canonical_future_block_winner_counts.csv']:
        assert sum(int(x['wins']) for x in rows(f/'robust_belief_model_comparison'/name))==656
    thresholds=rows(f/'update_gate_robustness/update_gate_threshold_sensitivity.csv')
    assert int(thresholds[0]['moved_trials'])==7200-4848
    slopes={x['subset']:float(x['point_estimate']) for x in rows(f/'update_gate_robustness/slope_decomposition_bootstrap_summary.csv')}
    assert round(slopes['all'],3)==.494 and round(slopes['nonzero_rows_only'],3)==.949
    report={'public_summary_checks':'passed','full_18_model_grid_refitted':False,'full_hmm_grid_refitted':False}
    if args.reproduction:
        import numpy as np
        import pandas as pd
        base=args.reproduction.resolve();data=base/'analysis/data'
        scripts=base/'analysis/scripts';sys.path[:0]=[str(scripts),str(scripts/'extension')]
        h2=load_module('deposit_h2',scripts/'H2.py')
        sessions=json.loads((data/'intermediate_outputs/valid_sessions.json').read_text())
        assert len(sessions)==240
        assert collections.Counter(s['orderCondition'] for s in sessions)=={i:40 for i in range(6)}
        counts={}
        for tag in ['strict','lenient','hybrid_S5','hybrid_S10','hybrid_S20']:
            policy=tag.split('_')[0];fallback=float(tag.split('_S')[1]) if '_S' in tag else 10.
            panel,qc,_=h2.build_h2_panel_robust(sessions,policy=policy,fallback_S=fallback,fallback_threshold=.30,reliability_delta_tau=.10)
            ref=pd.read_csv(data/f'hypothesis_outputs/H2/h2_panel_{tag}.csv')
            keys=['participant_id','taskType','t']
            pd.testing.assert_frame_equal(panel[ref.columns].sort_values(keys).reset_index(drop=True),ref.sort_values(keys).reset_index(drop=True),check_dtype=False,rtol=1e-10,atol=1e-12)
            qref=pd.read_csv(data/f'hypothesis_outputs/H2/h2_cf_qc_{tag}.csv')
            keys=['participant_id','taskType']
            pd.testing.assert_frame_equal(qc[qref.columns].sort_values(keys).reset_index(drop=True),qref.sort_values(keys).reset_index(drop=True),check_dtype=False,rtol=1e-10,atol=1e-12)
            counts[tag]=len(panel)
        behavior=load_module('extension_behavioral_pipeline',scripts/'extension/extension_behavioral_pipeline.py')
        classification=behavior.load_classification(data/'followup_outputs/robust_belief_model_comparison/canonical_classification.csv')
        regenerated=behavior.add_update_gate_variables(behavior.build_trial_panel(behavior.select_analysis_sessions(sessions),classification))
        reference=pd.read_csv(data/'followup_outputs/behavioral_extension_pipeline/trial_panel_with_updater_types.csv')
        keys=['participant_id','taskType','trialOrdinal']
        pd.testing.assert_frame_equal(regenerated[reference.columns].sort_values(keys).reset_index(drop=True),reference.sort_values(keys).reset_index(drop=True),check_dtype=False,rtol=1e-10,atol=1e-12)
        assert reference.participant_id.nunique()==240 and len(reference)==7200
        robust=load_module('deposit_robustness',scripts/'extension/update_gate_robustness.py')
        h2panel=pd.read_csv(data/'hypothesis_outputs/H2/h2_panel_hybrid_S10.csv').merge(classification[['participant_id','taskType','bic_winner']],on=['participant_id','taskType'],how='left')
        actual=robust.slope_decomposition(h2panel)
        for key,value in slopes.items():assert np.isclose(actual[key],value,rtol=1e-10,atol=1e-12),(key,actual[key],value)
        # Exact participant bootstrap using sufficient statistics. Demeaning
        # occurs within participant-task, so resampling a participant merely
        # repeats their numerator/denominator contribution, including repeated
        # draws with distinct draw IDs in the original implementation.
        reg=robust.add_h2_demeaning(h2panel)
        nonzero=reg.groupby('pid_task').delta_b.transform(lambda s:int((s.abs()>robust.EPS).sum()))
        masks=[np.ones(len(reg),dtype=bool),reg.bic_winner!='no_update',nonzero>=1,nonzero>=2,reg.delta_b.abs()>robust.EPS]
        names=['all','exclude_no_update_winner','at_least_one_nonzero_update_trajectory','at_least_two_nonzero_updates_trajectory','nonzero_rows_only']
        ids=sorted(reg.participant_id.unique())
        sampled=np.random.default_rng(42).choice(len(ids),size=(1000,len(ids)),replace=True)
        values={}
        for name,mask in zip(names,masks):
            part=reg[mask]
            stats=part.assign(num=part.delta_dm*part.norm_dm,den=part.norm_dm**2).groupby('participant_id')[['num','den']].sum().reindex(ids,fill_value=0).to_numpy()
            totals=stats[sampled].sum(axis=1);values[name]=totals[:,0]/totals[:,1]
        values['nonzero_minus_all']=values['nonzero_rows_only']-values['all']
        saved=pd.read_csv(data/'followup_outputs/update_gate_robustness/slope_decomposition_bootstrap_draws.csv')
        max_error=max(float(np.max(np.abs(values[k]-saved[k].to_numpy()))) for k in values)
        assert max_error<1e-12,max_error
        # Cross-check the shortcut against the original resampling procedure.
        direct_summary,direct_draws=robust.bootstrap_slope_decomposition(h2panel,reps=3,seed=42)
        for key in values:np.testing.assert_allclose(direct_draws[key],values[key][:3],rtol=1e-12,atol=1e-12)
        report['bootstrap_1000_draws_max_abs_error']=max_error
        report['bootstrap_sufficient_statistics_crosscheck']='Original algorithm first 3 draws match; all 1000 match stored draws.'
        hmm=load_module('deposit_hmm',scripts/'extension/markov_hmm_analysis.py')
        seq=hmm.load_sequences(data/'followup_outputs/behavioral_extension_pipeline/trial_panel_with_updater_types.csv')
        train,test=hmm.split_sequences(seq,.2,42)
        assert len(seq)==720 and len(train)==576 and len(test)==144
        assert not ({x.key[0] for x in train}&{x.key[0] for x in test})
        report.update({'h2_panel_and_qc_reconstruction':'passed','h2_rows':counts,'behavioral_panel_reconstruction':'passed','slope_point_estimate_reproduction':'passed','hmm_split_integrity':'passed'})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
