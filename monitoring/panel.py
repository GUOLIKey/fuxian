"""Read-only panel for the frozen ACD experiment."""
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
    units = sorted((E / 'results/units').glob('client*.json'))
    total_units = 30
    print('F01 frozen ACD + GMM parameter adaptation | NEW server')
    print(f"Stage: {st.get('stage','--')} | {st.get('message','')} | "
          f"{'DONE' if st.get('done') else 'RUNNING'}")
    print(f"Current: method={st.get('method','--')} client={st.get('client','--')} "
          f"seed={st.get('seed','--')} phase={st.get('phase','--')}")
    print(f"Progress: {len(units)}/{total_units} units "
          f"({100.0*len(units)/total_units:.0f}%)")
    if units:
        u = load(units[-1])
        if u:
            print(f"Latest unit: client {u.get('client')} seed {u.get('seed')} "
                  f"n={u.get('n')} originalK={u.get('original_K')}")
            print('Method   K   ARI     NMI     Purity   Low5  Ghost  HoldNLL   MMD     SWD')
            print('------  --  ------  ------  -------  ----  -----  --------  ------  ------')
            for m in ('A0', 'A1', 'A2', 'A3', 'A4', 'A5'):
                r = u.get('methods', {}).get(m)
                if not r:
                    continue
                f = r['full']; g = r['generation']; h = r['holdout']
                print(f"{m:6}  {f['K']:2}  {f['ari']:.4f}  {f['nmi']:.4f}  "
                      f"{f['purity']:.4f}  {f['n_low5']:4}  {f['n_ghost']:5}  "
                      f"{h['nll']:8.3f}  {g['mmd']:.4f}  {g['swd']:.4f}")
            if u.get('failures'):
                print(f"Unit failures: {len(u['failures'])}")
    fail_path = E / 'results/failures.jsonl'
    if fail_path.exists():
        lines = [x for x in fail_path.read_text().splitlines() if x.strip()]
        print(f"Total recorded failures: {len(lines)}")
        if lines:
            print('Last failure: ' + lines[-1][:300])
    if (E / 'error.txt').exists():
        print('ERROR: ' + (E / 'error.txt').read_text().strip().splitlines()[-1])


if __name__ == '__main__':
    main()
