"""Validate the two author metric functions without running Fed-HIRE training."""
import ast
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
source = ROOT.parents[1] / 'upstream/Fed-HIRE/opensource/experiment.py'
tree = ast.parse(source.read_text())
tree.body = [line for line in tree.body if not isinstance(line, ast.Assign)]
author = {}
exec(compile(tree, str(source), 'exec'), author)


def scores(truth, prediction):
    _, y = np.unique(truth, return_inverse=True)
    _, p = np.unique(prediction, return_inverse=True)
    counts = np.zeros((y.max()+1, p.max()+1), dtype=int)
    np.add.at(counts, (y, p), 1)
    rows, cols = linear_sum_assignment(counts, maximize=True)
    return counts.max(axis=0).sum()/len(y), counts[rows, cols].sum()/len(y)


fixtures = [
    ([0, 0, 1, 1, 2, 2], [2, 2, 0, 0, 1, 1]),
    ([0, 0, 0, 0, 0, 1, 1, 2, 2, 2], [0, 0, 0, 1, 1, 0, 1, 2, 2, 2]),
    ([0, 0, 0, 1, 1, 1], [0, 1, 2, 3, 3, 3]),
]
results = []
for truth, prediction in fixtures:
    truth, prediction = np.array(truth), np.array(prediction)
    author_scores = author['validation']([{'label': truth, 'predict': prediction, 'data': []}])
    reference = scores(truth, prediction)
    np.testing.assert_allclose(author_scores[[0, 3]], reference)
    results.append({'truth': truth.tolist(), 'prediction': prediction.tolist(),
                    'author_purity': float(author_scores[0]), 'author_acc': float(author_scores[3]),
                    'independent_purity': float(reference[0]), 'independent_acc': float(reference[1])})
(ROOT / 'hire-metric-fixtures.json').write_text(json.dumps(results, indent=2) + '\n')
