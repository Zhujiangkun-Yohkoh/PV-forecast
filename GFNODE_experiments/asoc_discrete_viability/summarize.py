import json
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parent
f = pd.read_csv(R / 'benchmark_per_seed.csv')
nums = ['rmse', 'mae', 'r2', 'nrmse']
rows = []
for k, g in f.groupby(['dataset', 'model', 'scope', 'horizon']):
    d = dict(zip(['dataset', 'model', 'scope', 'horizon'], k)); d['parameters'] = int(g.parameter_count.iloc[0])
    for n in nums: d[n + '_mean'], d[n + '_sd'] = g[n].mean(), g[n].std(ddof=1)
    rows.append(d)
s = pd.DataFrame(rows); s.to_csv(R / 'benchmark_mean_sd.csv', index=False)
full = s[s.scope == 'regular_full_timeline'].copy(); ranks = []
for (ds, h), g in full.groupby(['dataset', 'horizon']):
    g = g.sort_values('rmse_mean'); best = g.rmse_mean.iloc[0]
    for i, (_, x) in enumerate(g.iterrows(), 1): ranks.append({'dataset': ds, 'horizon': h, 'model': x.model, 'rmse_mean': x.rmse_mean, 'rank': i, 'gap_to_best_percent': (x.rmse_mean / best - 1) * 100})
r = pd.DataFrame(ranks); r.to_csv(R / 'model_rankings.csv', index=False)
eff = []
for m, g in f.groupby('model'):
    eff.append({'model': m, 'parameters': int(g.parameter_count.iloc[0]), 'mean_training_seconds': g.training_seconds.apply(pd.to_numeric, errors='coerce').mean(), 'mean_best_epoch': g.best_epoch.apply(pd.to_numeric, errors='coerce').mean()})
pd.DataFrame(eff).to_csv(R / 'parameter_efficiency.csv', index=False)
c = r[r.model == 'Discrete Candidate']; avg = float(c['rank'].mean()); h144 = c[c.horizon == 144]; behind = int((c.gap_to_best_percent > 5).sum()); no_last = all(not ((c[c.dataset == ds]['rank'] == 4).all()) for ds in c.dataset.unique())
decision = {'average_rmse_rank': avg, 'H144_top2_datasets': int((h144['rank'] <= 2).sum()), 'combinations_more_than_5pct_behind_best': behind, 'no_dataset_all_horizons_last': bool(no_last), 'verdict': 'PASS' if avg <= 2 and (h144['rank'] <= 2).sum() >= 2 and behind <= 2 and no_last else 'FAIL'}
(R / 'viability_decision.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')
(R / 'ASOC_DISCRETE_VIABILITY_REPORT.md').write_text('\n'.join(['# ASOC Discrete Candidate Viability', '', f"**Verdict: {decision['verdict']}**", '', f"- Average regular-timeline RMSE rank: {avg:.3f}", f"- H144 top-two datasets: {decision['H144_top2_datasets']}", f"- >5% behind best: {behind} combinations", '', 'Full seed metrics, mean±SD, rankings and parameter efficiency are in the accompanying CSV files.']) + '\n', encoding='utf-8')
