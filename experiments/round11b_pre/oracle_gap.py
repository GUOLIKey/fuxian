"""Summarize D0-D4 and compute gaps to the centralized real-latent oracle."""
import csv
import json
from collections import defaultdict

import numpy as np

from common import BASE, write_csv


def mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def num(v):
    try:
        if v in ('', None, 'None', 'nan'):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    rows = list(csv.DictReader((BASE / 'federated_statistics_aggregation.csv').open(newline='')))
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r['model'], r['method'], r['K'], r['covariance'])].append(r)
    summary = {}
    for (model, method, K, cov), rs in grouped.items():
        summary[(model, method, K, cov)] = {
            'model': model, 'method': method, 'K': K, 'covariance': cov,
            'n_rows': len(rs),
            'mean_val_ll': mean([num(r['val_ll']) for r in rs]),
            'mean_rare_proxy': mean([num(r['rare_proxy']) for r in rs]),
            'mean_class2_mmd': mean([num(r['class2_mmd']) for r in rs]),
            'mean_class2_swd': mean([num(r['class2_swd']) for r in rs]),
            'mean_mmd': mean([num(r['mmd_rbf']) for r in rs]),
            'mean_swd': mean([num(r['sliced_wasserstein']) for r in rs]),
        }
    out = []
    for key, item in sorted(summary.items()):
        model = item['model']
        if model == 'centralized':
            ref3 = summary.get((model, 'D4_centralized_full_K3', '3', 'full'))
            ref6 = summary.get((model, 'D4_centralized_full_K6', '6', 'full'))
        else:
            ref3 = summary.get((model, 'D4_oracle_full_K3', '3', 'full'))
            ref6 = summary.get((model, 'D4_oracle_full_K6', '6', 'full'))
        row = dict(item)
        row.update({
            'ref_full_K3_val_ll': ref3['mean_val_ll'] if ref3 else None,
            'ref_full_K3_rare': ref3['mean_rare_proxy'] if ref3 else None,
            'gap_val_ll_full_K3': (item['mean_val_ll'] - ref3['mean_val_ll']
                                   if ref3 and item['mean_val_ll'] is not None and ref3['mean_val_ll'] is not None else None),
            'gap_rare_full_K3': (item['mean_rare_proxy'] - ref3['mean_rare_proxy']
                                 if ref3 and item['mean_rare_proxy'] is not None and ref3['mean_rare_proxy'] is not None else None),
            'ref_full_K6_val_ll': ref6['mean_val_ll'] if ref6 else None,
            'ref_full_K6_rare': ref6['mean_rare_proxy'] if ref6 else None,
            'gap_val_ll_full_K6': (item['mean_val_ll'] - ref6['mean_val_ll']
                                   if ref6 and item['mean_val_ll'] is not None and ref6['mean_val_ll'] is not None else None),
            'gap_rare_full_K6': (item['mean_rare_proxy'] - ref6['mean_rare_proxy']
                                 if ref6 and item['mean_rare_proxy'] is not None and ref6['mean_rare_proxy'] is not None else None),
            'note': 'D4 is a centralized real-latent oracle, not federated',
        })
        out.append(row)
    write_csv(BASE / 'oracle_gap_analysis.csv', out)
    (BASE / 'oracle_gap_summary.json').write_text(json.dumps(out, indent=2) + '\n')
    print('ORACLE_GAP_OK', len(out), flush=True)


if __name__ == '__main__':
    main()
