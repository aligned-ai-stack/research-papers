"""Reproduce aggregate figures or refit the study using approved local inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import kruskal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GROUPS = ['Female_Black', 'Female_White', 'Male_Black', 'Male_White']
OUTCOMES = ['fairness', 'spp', 'privacy_concerns', 'impression_management']
LABELS = {'fairness': 'Fairness', 'spp': 'Social perception (SPP)',
          'privacy_concerns': 'Privacy / creepiness (PER)',
          'impression_management': 'Impression management'}
SCALES = {'ati': (9, 6), 'masi': (30, 5), 'fairness': (10, 7),
          'privacy_concerns': (10, 7), 'impression_management': (4, 5),
          'perceived_warmth': (2, 7), 'perceived_competence': (2, 7),
          'social_presence': (4, 7), 'perceived_outcome': (1, 7)}

def score(participants, items, profile):
    """Items are numeric, pre-reversal responses; align by opaque study_id."""
    if profile not in ('historical', 'corrected'):
        raise ValueError('Unknown scoring profile')
    required = ['study_id', 'age', 'user_demographic', 'interviewer_demographic']
    if not set(required).issubset(participants.columns) or 'study_id' not in items:
        raise ValueError('Required participant fields / study_id missing')
    for frame in (participants, items):
        if frame.study_id.isna().any() or not frame.study_id.is_unique:
            raise ValueError('study_id must be nonmissing and unique')
    if set(participants.study_id) != set(items.study_id):
        raise ValueError('Participant and item IDs must match exactly')
    result = participants[required].copy().reset_index(drop=True)
    if result[required].isna().any().any():
        raise ValueError('Missing participant covariate or condition')
    if not result.user_demographic.isin(GROUPS).all():
        raise ValueError('Unexpected participant group; no implicit exclusions allowed')
    age = pd.to_numeric(result.age, errors='raise')
    if not np.isfinite(age).all() or not age.between(18, 120).all():
        raise ValueError('Invalid adult age')
    result['age'] = age
    aligned = items.set_index('study_id').loc[result.study_id].reset_index(drop=True)
    for scale, (count, maximum) in SCALES.items():
        names = [f'{scale}_item_{i:02d}' for i in range(1, count + 1)]
        block = aligned[names].apply(pd.to_numeric, errors='raise').copy()
        values = block.to_numpy()
        if not (np.isfinite(values).all() and ((values >= 1) & (values <= maximum)).all()
                and (values == np.floor(values)).all()):
            raise ValueError(f'Invalid or missing Likert response: {scale}')
        reverse = {'ati': [3, 6, 8], 'masi': [4, 6]}.get(scale, [])
        if profile == 'corrected' and scale == 'privacy_concerns':
            reverse = [10]
        for i in reverse:
            block.iloc[:, i - 1] = maximum + 1 - block.iloc[:, i - 1]
        result[scale] = block.mean(axis=1)
    if profile == 'historical':
        result['ati'] /= 9
    result['spp'] = result[['perceived_warmth', 'perceived_competence', 'social_presence']].mean(axis=1)
    return result

def descriptives(data, profile):
    rows = []
    for factor in ['user_demographic', 'interviewer_demographic']:
        for group, frame in data.groupby(factor, sort=True):
            for outcome in OUTCOMES:
                rows.append(dict(profile=profile, factor=factor, group=group, outcome=outcome,
                                 n=len(frame), mean=frame[outcome].mean(), sd=frame[outcome].std()))
    return pd.DataFrame(rows)

def fit(data, profile, n_boot=500, seed=42):
    if set(data.user_demographic) != set(GROUPS):
        raise ValueError('All four participant groups are required for these models')
    if n_boot < 2:
        raise ValueError('At least two bootstrap samples are required')
    omnibus, posthoc, mediation = [], [], []
    dummy = data.join(pd.get_dummies(data.user_demographic, dtype=int))
    for factor in ['user_demographic', 'interviewer_demographic']:
        for outcome in OUTCOMES:
            h, p = kruskal(*[g[outcome] for _, g in data.groupby(factor)])
            omnibus.append(dict(profile=profile, factor=factor, outcome=outcome, H=h, p=p))
            if p < .05:
                table = pg.pairwise_gameshowell(data=data, dv=outcome, between=factor)
                table['profile'], table['factor'], table['outcome'] = profile, factor, outcome
                posthoc.append(table)
    for group in GROUPS:
        for outcome in ['fairness', 'privacy_concerns', 'impression_management']:
            table = pg.mediation_analysis(data=dummy, x=group, m='spp', y=outcome,
                                          covar=['age', 'ati', 'masi'], n_boot=n_boot, seed=seed)
            table = table.rename(columns={'CI2.5': 'ci_lower', 'CI97.5': 'ci_upper',
                                          'CI[2.5%]': 'ci_lower', 'CI[97.5%]': 'ci_upper'})
            table['demographic'], table['outcome'], table['profile'] = group, outcome, profile
            table['bootstrap_samples'], table['seed'] = n_boot, seed
            mediation.append(table)
    empty = pd.DataFrame(columns=['A', 'B', 'mean(A)', 'mean(B)', 'diff', 'se', 'T', 'df',
                                  'pval', 'hedges', 'profile', 'factor', 'outcome'])
    return {'descriptives': descriptives(data, profile), 'kruskal': pd.DataFrame(omnibus),
            'posthoc': pd.concat(posthoc, ignore_index=True) if posthoc else empty,
            'mediation': pd.concat(mediation, ignore_index=True)}

def plots(tables, output):
    output.mkdir(parents=True, exist_ok=True)
    desc, med = pd.read_csv(tables / 'descriptives.csv'), pd.read_csv(tables / 'mediation.csv')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': 'hcomp2024'})
    for profile in sorted(desc.profile.unique()):
        fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
        for ax, outcome in zip(axes, OUTCOMES):
            d = desc[(desc.profile == profile) & (desc.factor == 'user_demographic') & (desc.outcome == outcome)].set_index('group').loc[GROUPS]
            ax.bar(range(4), d['mean'], color=['#427f9c', '#78a8b8', '#bc7250', '#d7a47e'])
            ax.set_xticks(range(4), [g.replace('_', '\n') for g in GROUPS])
            ax.set_ylim(0, 5 if outcome == 'impression_management' else 7)
            ax.set_title(LABELS[outcome])
            for i, (_, row) in enumerate(d.iterrows()):
                ax.text(i, row['mean'] + .10, f"{row['mean']:.2f}", ha='center', fontsize=9)
        axes[0].set_ylabel('Mean score')
        total = int(desc[(desc.profile == profile) & (desc.factor == 'user_demographic') & (desc.outcome == 'fairness')]['n'].sum())
        fig.suptitle(f'{profile.capitalize()} scoring · participant group means · N = {total}')
        fig.tight_layout()
        for extension in ['png', 'svg']:
            fig.savefig(output / f'{profile}_means.{extension}', dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
        for ax, outcome in zip(axes, ['fairness', 'privacy_concerns', 'impression_management']):
            d = med[(med.profile == profile) & (med.outcome == outcome) & (med.path == 'Indirect')].set_index('demographic').loc[GROUPS]
            # Plot endpoints directly: asymmetric intervals need not contain the estimate.
            ax.hlines(range(4), d.ci_lower, d.ci_upper, color='#427f9c', linewidth=2)
            ax.scatter(d.coef, range(4), color='#193b51', zorder=3)
            ax.axvline(0, color='#888888', linewidth=.8, linestyle='--')
            ax.set_yticks(range(4), [g.replace('_', ' ') for g in GROUPS])
            ax.invert_yaxis()
            ax.set_title(LABELS[outcome])
            ax.set_xlabel('Indirect coefficient via SPP (95% bootstrap interval)')
        fig.suptitle(f'{profile.capitalize()} scoring · each participant group versus all others')
        fig.tight_layout()
        for extension in ['png', 'svg']:
            fig.savefig(output / f'{profile}_indirect.{extension}', dpi=180)
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plots', help='Regenerate figures from public aggregate tables')
    p.add_argument('--tables', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path, default=Path('figures'))
    p = sub.add_parser('run', help='Score and refit approved participant-level inputs locally')
    p.add_argument('--participants', type=Path, required=True)
    p.add_argument('--items', type=Path, required=True)
    p.add_argument('--profile', choices=['historical', 'corrected'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n-boot', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.command == 'plots':
        plots(args.tables, args.output)
    else:
        # Refuse reuse so input files or a prior analysis cannot be overwritten.
        if args.output.exists():
            raise FileExistsError('Choose a new output directory')
        data = score(pd.read_csv(args.participants), pd.read_csv(args.items), args.profile)
        tables = fit(data, args.profile, args.n_boot, args.seed)
        args.output.mkdir(parents=True)
        data.to_csv(args.output / 'scores_PRIVATE.csv', index=False)
        for name, table in tables.items():
            table.to_csv(args.output / f'{name}.csv', index=False)
        import importlib.metadata
        (args.output / 'run.json').write_text(json.dumps({
            'profile': args.profile, 'n': len(data), 'bootstrap_samples': args.n_boot,
            'seed': args.seed, 'row_order': 'participant input order',
            'runtime': {name: importlib.metadata.version(name) for name in
                        ['numpy', 'pandas', 'scipy', 'pingouin', 'statsmodels']}}, indent=2) + '\n')
        plots(args.output, args.output / 'figures')

if __name__ == '__main__':
    main()
