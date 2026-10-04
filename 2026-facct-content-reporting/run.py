"""Run the six analyses on approved local, minimized participant inputs."""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, os, subprocess, sys
import pandas as pd

ROOT=Path(__file__).resolve().parent
INPUTS=['final_analysis_0301.csv','final_analysis_0301_cleaned.csv','final_survey_0301.csv',
        'final_sessions_0301.csv','combined_coder_ratings.csv']

def validate(data):
    for name in INPUTS:
        if not (data/name).is_file(): raise ValueError(f'Missing required input: {name}')
    a=pd.read_csv(data/INPUTS[0]); b=pd.read_csv(data/INPUTS[1]); r=pd.read_csv(data/INPUTS[-1])
    required=['id','participantId','postId','correctProvision','condition','errorType','isErrorTrial',
              'accuracyProvision','confusionDistance','decisionTimeMs','chosenAction','outOfScopeChoice']
    if set(a.columns)!=set(required) or set(b.columns)!=set(required):
        raise ValueError('Use the minimized analysis schema in data/input_dictionary.csv')
    pd.testing.assert_frame_equal(a,b)
    if a[required].isna().any().any() or a.id.duplicated().any() or a.duplicated(['participantId','postId']).any():
        raise ValueError('Missing values or duplicate trials')
    if set(a.condition)!= {'manual','evaluative','explainable'}:
        raise ValueError('Expected three assistance conditions')
    if not a.groupby('participantId').condition.nunique().eq(1).all():
        raise ValueError('A participant occurs in multiple conditions')
    if not a.groupby('participantId').size().eq(2).all():
        raise ValueError('Expected two trials per participant')
    for c in ['isErrorTrial','accuracyProvision','outOfScopeChoice']:
        if not a[c].isin([0,1,True,False]).all(): raise ValueError(f'Invalid binary field: {c}')
    if not a.confusionDistance.isin([0,1,2,3]).all() or not a.decisionTimeMs.gt(0).all():
        raise ValueError('Invalid outcome values')
    ai=a[a.condition.ne('manual')]
    if not ai.groupby('participantId').isErrorTrial.sum().eq(1).all():
        raise ValueError('AI participants must have exactly one error trial')
    if not ai[ai.isErrorTrial.eq(1)].errorType.isin(['near_miss','overbreadth','out_of_scope']).all():
        raise ValueError('Unexpected AI error type')
    if r.id.duplicated().any() or set(r.id)!=set(a.id): raise ValueError('Ratings must join one-to-one to all trials')
    ratings=r.drop(columns='id').apply(pd.to_numeric,errors='raise')
    if ratings.isna().any().any() or not ((ratings>=1)&(ratings<=5)).all().all():
        raise ValueError('Invalid coder rating')
    for stem in ['element_coverage','proportionality_reasoning','reasoning_depth','perceived_overall_quality']:
        if not (ratings[stem+'_mean']==ratings[[stem+'_marie',stem+'_rita']].mean(axis=1)).all():
            raise ValueError('Coder means do not match individual ratings')
    return len(a),a.participantId.nunique()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.data=args.data.resolve();args.output=args.output.resolve()
    if args.output.exists(): raise FileExistsError('Choose a new output directory')
    n,people=validate(args.data)
    args.output.mkdir(parents=True)
    env=os.environ.copy();env.update(FACCT_DATA_DIR=str(args.data),FACCT_OUTPUT_DIR=str(args.output),
        MPLBACKEND='Agg',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
    for i in range(1,7):
        result=subprocess.run([sys.executable,'-B',str(ROOT/f'analysis/H{i}.py')],env=env,
                              cwd=args.output,capture_output=True,text=True,encoding='utf-8')
        (args.output/f'H{i}.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode: raise RuntimeError(f'H{i} failed; see its local log')
        print(f'H{i} complete',flush=True)
    failed=list(args.output.rglob('*FAILED*'))
    if failed: raise RuntimeError('A sensitivity model failed; inspect local outputs')
    info={'trials':n,'participants':int(people),'input_sha256':{name:hashlib.sha256((args.data/name).read_bytes()).hexdigest() for name in INPUTS},
          'runtime':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','statsmodels','matplotlib']}}
    (args.output/'run.json').write_text(json.dumps(info,indent=2)+'\n')

if __name__=='__main__':main()
