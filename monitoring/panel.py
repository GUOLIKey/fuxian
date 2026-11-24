"""Read-only panel for the F01 frozen Multi-CA experiment."""
from __future__ import annotations

import json
import sys
from pathlib import Path

E = Path(sys.argv[1])


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def main():
    st = load(E / 'status.json') or {}
    m = load(E / 'results/metrics.json')
    print('F01 + Multi-CA frozen aggregation | NEW server')
    print(f"Stage: {st.get('stage','--')} | {st.get('message','')} | "
          f"{'DONE' if st.get('done') else 'RUNNING'}")
    if m:
        print(f"Counts: A={m['component_counts']['A']} B={m['component_counts']['B']} "
              f"C={m['component_counts']['C']} union={m['component_counts']['union']}")
        print(f"A validation vs historical per-seed: exact={m['validation_A']['exact_1e-9']}")
        print()
        print('Model   K   ARI(0-2)  ARI(0-9)   NMI    Purity     NLL    GenMMD   GenSWD  Sup  Ghost  Mass<20')
        print('-----  --  --------  --------  ------  ------  -------  -------  -------  ---  -----  -------')
        for name in ('A', 'B', 'C', 'union'):
            s = m['per_model'][name]
            print(f"{name:5}  {s['K']:2}  {s['ari_0_2_mean']:.4f}    "
                  f"{s['ari_0_9_mean']:.4f}   {s['nmi_0_2_mean']:.4f}  "
                  f"{s['purity_0_2_mean']:.4f}  {s['nll']:.4f}  "
                  f"{s['gen_mmd_mean']:.5f}  {s['gen_swd_mean']:.5f}   "
                  f"{s['n_supported_20']:2}   {s['n_ghost']:2}   {s['mass_unsupported']:.4f}")
        eq = m.get('b_c_equivalence', {})
        print()
        print('B=C equivalence: leaf={} weights={} means={} logvar={} macro={} metrics={}'.format(
            eq.get('leaf_partitions_equal'), eq.get('weights_allclose'),
            eq.get('means_allclose'), eq.get('log_sigma2_allclose'),
            eq.get('macro_map_equal'), eq.get('metrics_identical')))
        a = load(E / 'results/aggregation.json')
        if a:
            c = a['models']['C']
            print(f"C stop: tau={c.get('stop_threshold', 0):.4f} "
                  f"allocation={c.get('macro_allocation')} K={c.get('n_micro')}")
        lc = m.get('latent_compatibility', {})
        print(f"Latent client-vs-server: meanL2={lc.get('max_mean_l2_to_server', 0):.4f} "
              f"MMD={lc.get('max_mmd_rbf_to_server', 0):.4f}")
        print('Key result: Multi-CA == same-capacity Flat GMM; no hierarchy gain.')
    else:
        print('Results not written yet.')
    if (E / 'error.txt').exists():
        print('ERROR: ' + (E / 'error.txt').read_text().strip().splitlines()[-1])


if __name__ == '__main__':
    main()
