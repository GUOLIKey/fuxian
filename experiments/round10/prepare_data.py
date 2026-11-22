"""Prepare official UCI Cardiotocography on the remote server only."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path('/root/autodl-tmp/fedvade_project/reproduction_runs/round10_non_iid_generation')
FEATURES = 'LB AC FM UC DL DS DP ASTV MSTV ALTV MLTV Width Min Max Nmax Nzeros Mode Mean Median Variance Tendency'.split()
SOURCE_URL = 'https://archive.ics.uci.edu/static/public/193/cardiotocography.zip'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def grouped_split(indices, labels, groups, fraction, seed):
    unique = np.unique(groups[indices])
    group_labels = []
    for group in unique:
        y = labels[indices[groups[indices] == group]]
        group_labels.append(int(np.bincount(y, minlength=3).argmax()))
    left, right = train_test_split(unique, test_size=fraction, stratify=group_labels,
                                   random_state=seed)
    return indices[np.isin(groups[indices], left)], indices[np.isin(groups[indices], right)]


def partition(labels, mode, seed):
    rng = np.random.RandomState(seed)
    buckets = [[], [], []]
    for cls in range(3):
        ids = np.flatnonzero(labels == cls)
        rng.shuffle(ids)
        if cls != 2 or mode == 'iid':
            pieces = np.array_split(ids, 3)
        elif mode == 'medium':
            first = int(round(len(ids) * .15))
            pieces = (ids[:first], ids[first:2 * first], ids[2 * first:])
        else:
            pieces = (np.array([], dtype=int), np.array([], dtype=int), ids)
        for i, part in enumerate(pieces):
            buckets[i].extend(part.tolist())
    for bucket in buckets:
        rng.shuffle(bucket)
    flat = np.concatenate([np.asarray(x, dtype=int) for x in buckets])
    if len(flat) != len(labels) or len(np.unique(flat)) != len(labels):
        raise RuntimeError('Partition omitted or repeated training records')
    if mode == 'severe' and (np.any(labels[buckets[0]] == 2) or np.any(labels[buckets[1]] == 2)):
        raise RuntimeError('Missing-class scenario not enforced')
    return [np.asarray(x, dtype=int) for x in buckets]


def main():
    output = ROOT / 'dataset_manifest.json'
    if output.exists():
        raise SystemExit('Dataset already prepared; refusing overwrite')
    archive = ROOT / 'raw/cardiotocography.zip'
    workbook = ROOT / 'raw/CTG.xls'
    if not archive.is_file() or not workbook.is_file():
        raise FileNotFoundError('Official UCI archive/workbook missing')
    with zipfile.ZipFile(archive) as package:
        if package.namelist() != ['CTG.xls']:
            raise RuntimeError('Unexpected archive content')
    raw = pd.read_excel(workbook, sheet_name='Raw Data')
    raw = raw.dropna(subset=['NSP']).reset_index(drop=True)
    if len(raw) != 2126 or raw[FEATURES + ['NSP']].isna().any().any():
        raise RuntimeError('Unexpected UCI table shape or missing values')
    X = raw[FEATURES].to_numpy(dtype=np.float64)
    y = raw['NSP'].to_numpy(dtype=int) - 1
    if set(np.unique(y)) != {0, 1, 2} or not np.isfinite(X).all():
        raise RuntimeError('Unexpected class labels or nonfinite features')
    groups_hex = [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in X]
    _, groups = np.unique(groups_hex, return_inverse=True)
    conflict = sum(len(np.unique(y[groups == group])) > 1 for group in np.unique(groups))
    trainval, test = grouped_split(np.arange(len(y)), y, groups, .20, 42)
    train, val = grouped_split(trainval, y, groups, .20, 43)
    if set(groups[train]) & set(groups[val]) or set(groups[train]) & set(groups[test]) or set(groups[val]) & set(groups[test]):
        raise RuntimeError('Duplicate feature vector crossed split boundary')
    scaler = StandardScaler().fit(X[train])
    split = ROOT / 'private_data'
    split.mkdir(exist_ok=True)
    np.savez_compressed(split / 'data.npz', X_train=scaler.transform(X[train]).astype('float32'),
                        y_train=y[train], X_val=scaler.transform(X[val]).astype('float32'),
                        y_val=y[val], X_test=scaler.transform(X[test]).astype('float32'),
                        y_test=y[test], original_train_indices=train, original_val_indices=val,
                        original_test_indices=test, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
    scenarios = {}
    partition_rows = []
    for scenario in ('iid', 'medium', 'severe'):
        paths = partition(y[train], scenario, 42)
        np.savez_compressed(split / f'{scenario}_partitions.npz', **{
            f'client_{i}': ids for i, ids in enumerate(paths)})
        scenarios[scenario] = []
        for client, ids in enumerate(paths):
            counts = np.bincount(y[train][ids], minlength=3).tolist()
            item = {'scenario': scenario, 'client': client, 'n': len(ids),
                    'class_0': counts[0], 'class_1': counts[1], 'class_2': counts[2],
                    'original_index_sha256': hashlib.sha256(
                        np.ascontiguousarray(train[ids]).tobytes()).hexdigest()}
            partition_rows.append(item)
            scenarios[scenario].append(item)
    with (ROOT / 'partition_manifest.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(partition_rows[0]))
        writer.writeheader(); writer.writerows(partition_rows)
    atomic_json(output, {'dataset': 'UCI Cardiotocography', 'doi': '10.24432/C51S4N',
                'url': SOURCE_URL, 'license': 'CC BY 4.0', 'archive_sha256': sha(archive),
                'workbook_sha256': sha(workbook), 'feature_columns': FEATURES,
                'excluded_columns': ['FileName', 'Date', 'SegFile', 'b', 'e', 'LBE',
                                     'DR', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD',
                                     'FS', 'SUSP', 'CLASS', 'NSP'],
                'raw_rows': len(y), 'raw_class_counts': np.bincount(y, minlength=3).tolist(),
                'feature_duplicate_rows': int(len(y) - len(np.unique(groups))),
                'duplicate_groups_with_conflicting_labels': conflict,
                'split_seed': [42, 43], 'partition_seed': 42,
                'split_counts': {name: {'n': len(ids),
                    'class_counts': np.bincount(y[ids], minlength=3).tolist()}
                    for name, ids in [('train', train), ('validation', val), ('test', test)]},
                'scaler_fit': 'train only', 'group_key': 'SHA256 exact 21-feature float64 row',
                'data_npz_sha256': sha(split / 'data.npz'),
                'partition_npz_sha256': {s: sha(split / f'{s}_partitions.npz') for s in scenarios},
                'scenarios': scenarios})
    (ROOT / 'partition_protocol.md').write_text(
        '# Round 10 split and partition protocol\n\n'
        'Official UCI CTG.xls Raw Data sheet, 21 listed physiological features, NSP target (1/2/3 mapped to 0/1/2). '
        'CLASS and all expert pattern indicator columns are excluded. Exact duplicate feature rows are grouped before splitting. '
        'The test set and validation set are sealed before any scenario is built. StandardScaler fits model-training records only. '
        'Training labels are used solely to construct the three controlled client allocations; the model GMM component count is fixed '
        'globally at 3 and chosen locally by unlabeled BIC. Original indices and processed arrays are server-only.\n\n'
        'IID: stratified round-robin per class. Medium: 15% of rare-class training records to A, 15% to B, 70% to C. '
        'Severe: no rare-class records on A or B, all rare-class training records on C. Every training record appears once per scenario.\n')
    print(json.dumps({'raw': len(y), 'classes': np.bincount(y).tolist(),
                      'train_val_test': [len(train), len(val), len(test)],
                      'conflicting_duplicate_groups': conflict,
                      'scenarios': {s: [r['class_2'] for r in rows] for s, rows in scenarios.items()}}),
          flush=True)


if __name__ == '__main__':
    main()
