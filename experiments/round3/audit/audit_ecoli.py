"""Compare the author's Ecoli partition to the UCI feature and label records."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
NPZ = ROOT.parents[1] / 'upstream/FedHCA/dataset/EC.npz'


def describe(x, y, source):
    groups = defaultdict(list)
    for row, label, client in zip(x, y, source):
        groups[tuple(row)].append((str(label), int(client)))
    repeated = [records for records in groups.values() if len(records) > 1]
    return {
        'records': len(x),
        'unique_vectors': len(groups),
        'unique_labeled_vectors': sum(len(set(label for label, _ in group)) for group in groups.values()),
        'duplicate_records_beyond_first': len(x) - len(groups),
        'repeated_vectors': len(repeated),
        'records_in_repeated_groups': sum(map(len, repeated)),
        'conflicting_label_vectors': sum(len({label for label, _ in group}) > 1 for group in repeated),
        'vectors_repeated_across_clients': sum(len({i for _, i in group}) > 1 for group in repeated),
        'vectors_repeated_within_client': sum(len({i for _, i in group}) < len(group) for group in repeated),
        'classes': dict(Counter(map(str, y))),
    }


def main():
    with np.load(NPZ, allow_pickle=True) as dataset:
        clients = [dataset[key].item() for key in dataset.files]
    x = np.concatenate([client['data'] for client in clients])
    y = np.concatenate([client['label'] for client in clients]).ravel()
    source = np.repeat(np.arange(len(clients)), [len(client['data']) for client in clients])
    raw = [line.split() for line in (ROOT / 'uci-ecoli.data').read_text().splitlines() if line.strip()]
    raw_x = np.array([row[1:-1] for row in raw], dtype=float)
    raw_y = np.array([row[-1] for row in raw])
    classes, raw_classes = np.unique(y), np.unique(raw_y)
    matches = {}
    raw_mu, raw_std = raw_x.mean(axis=0), raw_x.std(axis=0)
    standardized = (raw_x - raw_mu) / raw_std
    for name, matrix in {
        'uci_raw': raw_x,
        'uci_global_zscore': standardized,
        'uci_global_minmax': (raw_x - raw_x.min(axis=0)) / np.ptp(raw_x, axis=0),
    }.items():
        lookup = defaultdict(set)
        for row, label in zip(np.round(matrix, 6), raw_y):
            lookup[tuple(row)].add(label)
        matched = [lookup.get(tuple(row), set()) for row in np.round(x, 6)]
        counts = np.array([[sum(label == a and b in candidates for label, candidates in zip(y, matched))
                            for b in raw_classes] for a in classes])
        rows, cols = linear_sum_assignment(-counts)
        mapping = {str(classes[i]): str(raw_classes[j]) for i, j in zip(rows, cols) if counts[i, j]}
        matches[name] = {'matching_features': sum(bool(item) for item in matched),
                         'matching_features_and_labels': int(counts[rows, cols].sum()),
                         'npz_to_uci_label_mapping': mapping,
                         'unique_UCI_vectors_covered': len(set(map(tuple, np.round(x, 6))) & lookup.keys())}
    result = {'sha256': sha256(NPZ.read_bytes()).hexdigest(),
              'client_sizes': [len(client['data']) for client in clients],
              'stored_sizes': [int(client['size']) for client in clients],
              'client_audits': [describe(x[source == i], y[source == i], source[source == i])
                                for i in range(len(clients))],
              'npz': describe(x, y, source), 'uci': describe(raw_x, raw_y, np.zeros(len(raw))),
              'uci_distinct_identifiers': len({row[0] for row in raw}),
              'feature_count': x.shape[1], 'feature_means': x.mean(axis=0).tolist(),
              'feature_stds': x.std(axis=0).tolist(), 'feature_min': x.min(axis=0).tolist(),
              'feature_max': x.max(axis=0).tolist(), 'uci_matching': matches}
    distance = np.max(np.abs(x[:, None, :] - standardized[None, :, :]), axis=2)
    result['max_abs_zscore_residual'] = float(distance.min(axis=1).max())
    result['raw_vectors_missing_from_npz'] = len(raw_x) - matches['uci_global_zscore']['unique_UCI_vectors_covered']
    (ROOT / 'ecoli.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
