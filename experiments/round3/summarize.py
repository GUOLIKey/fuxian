"""Aggregate the fixed paired protocols and verify every saved final trace."""
import csv
import json
from math import isclose
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parent
NEW = ROOT / 'autodl-results' / 'formal'
OLD = ROOT.parent / 'round2/autodl-results/formal'
CLASSICAL = ('purity', 'ari', 'nmi', 'acc')
DEEP = ('purity_150', 'ari_150', 'purity', 'ari')


def read(path):
    return json.loads(path.read_text())


def write_rows(name, rows):
    with (ROOT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_trace(folder, data, schedule):
    rows = [json.loads(s) for s in (folder / 'rounds.jsonl').read_text().splitlines()]
    assert [row['round'] for row in rows] == list(range(1, 301))
    assert rows[149] == data['round_150'] and rows[-1] == data['last']
    assert (folder / 'round_300_server.pt').stat().st_size > 8_000_000
    defaults = data['params']
    initial, gamma, step = (defaults[key] for key in ('initial_lr', 'gamma', 'stepsize'))
    counts = [item['batches'] for item in data['partition']]
    for index, row in enumerate(rows):
        assert len(counts) == len(row['client_batch_lrs']) == len(row['client_end_lrs'])
        for n, rates, ending in zip(counts, row['client_batch_lrs'], row['client_end_lrs']):
            assert len(rates) == n
            for offset, rate in enumerate(rates):
                updates = index * n + offset if schedule == 'author' else index
                assert isclose(rate, initial * gamma ** (updates // step))
            updates = (index + 1) * n if schedule == 'author' else index + 1
            assert isclose(ending, initial * gamma ** (updates // step))
        assert isclose(row['server_lr'], initial * gamma ** (index // step))
        assert isclose(row['server_end_lr'], initial * gamma ** ((index + 1) // step))


def select(name, seed, prior_count):
    folder = (OLD if seed < prior_count else NEW) / f'{name}_seed{seed}'
    if not (folder / 'status.json').exists():
        return None
    status = read(folder / 'status.json')
    if status['status'] != 'ok':
        return None
    data = read(folder / 'result.json')
    reference = read(OLD / f'{name}_seed0' / 'result.json')
    for key in ('params', 'partition', 'source_hashes', 'python', 'numpy', 'device', 'torch'):
        assert data.get(key) == reference.get(key), (folder, key)
    assert data['seed'] == seed and data['device'] == 'cpu'
    return folder, data


def summary(rows, keys):
    return {key: dict(mean=mean(values), sd=pstdev(values), min=min(values), max=max(values))
            for key in keys if (values := [row[key] for row in rows])}


def main():
    hire, fedvade = [], []
    for rate in ('0.0001', '0.03', '0.05'):
        for seed in range(10):
            selected = select(f'A_lr{rate}', seed, 2)
            if selected:
                _, data = selected
                hire.append({'rate': rate, 'seed': seed, **{key: data[key] for key in CLASSICAL},
                             'elapsed_seconds': data['elapsed_seconds']})
    for schedule in ('author', 'epoch'):
        for seed in range(5):
            selected = select(f'B_{schedule}', seed, 1)
            if selected:
                folder, data = selected
                validate_trace(folder, data, schedule)
                deep = {key + '_150': data['round_150'][key] for key in ('purity', 'ari')}
                deep.update({key: data['last'][key] for key in ('purity', 'ari', 'client_loss', 'server_loss')})
                fedvade.append({'schedule': schedule, 'seed': seed, **deep, 'elapsed_seconds': data['elapsed_seconds']})
    results = {}
    for rows, model, axis, metrics, comparisons in [
        (hire, 'hire', 'rate', CLASSICAL, [('0.03', '0.0001'), ('0.05', '0.0001'), ('0.05', '0.03')]),
        (fedvade, 'fedvade', 'schedule', DEEP, [('epoch', 'author')]),
    ]:
        groups = {}
        for row in rows:
            groups.setdefault(row[axis], {})[row['seed']] = row
        results[model] = {'completed': len(rows), 'groups': {}, 'paired': {}}
        for group, by_seed in groups.items():
            loss_metrics = ('client_loss', 'server_loss') if model == 'fedvade' else ()
            stats = summary(list(by_seed.values()), metrics + loss_metrics)
            results[model]['groups'][group] = {'n': len(by_seed), **stats}
        differences = []
        for treatment, base in comparisons:
            high, low = groups.get(treatment, {}), groups.get(base, {})
            key = f'{treatment}-{base}'
            paired = [{'comparison': key, 'seed': seed,
                       **{metric: high[seed][metric] - low[seed][metric] for metric in metrics}}
                      for seed in sorted(high.keys() & low.keys())]
            differences.extend(paired)
            results[model]['paired'][key] = {'n': len(paired), **summary(paired, metrics),
                                             'strictly_positive': {m: sum(p[m] > 0 for p in paired) for m in metrics}}
        if rows:
            write_rows(f'{model}-all-seeds.csv', rows)
        if differences:
            write_rows(f'{model}-paired.csv', differences)
    (ROOT / 'statistics.json').write_text(json.dumps(results, indent=2) + '\n')
    print('Verified:', {name: value['completed'] for name, value in results.items()})


if __name__ == '__main__':
    main()
