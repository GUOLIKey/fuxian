"""Read-only panel for the frozen-client global ACD experiment."""
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
    runs = sorted((E / 'results/runs').glob('*_seed*.json'))
    done = 0
    partial = 0
    for p in runs:
        r = load(p) or {}
        if r.get('status') == 'DONE':
            done += 1
        elif r.get('status'):
            partial += 1
    print('F01 frozen-client global ACD experiment | NEW server')
    print(f"Stage: {st.get('stage','--')} | {st.get('message','')} | "
          f"{'DONE' if st.get('done') else 'RUNNING'}")
    print(f"Current: group={st.get('group','--')} seed={st.get('seed','--')} "
          f"round={st.get('round','--')}/{st.get('rounds','10')}")
    print(f"Runs: {done} done, {partial} partial / 15")
    latest = None
    rp = E / 'results/rounds.jsonl'
    if rp.exists():
        lines = [x for x in rp.read_text().splitlines() if x.strip()]
        if lines:
            latest = json.loads(lines[-1])
    if latest:
        print(f"Latest: {latest['group']} seed {latest['seed']} round "
              f"{latest['round']} | ARI {latest['ari']:.4f} "
              f"(SD {latest.get('ari_sd_inference',0):.4f}) | "
              f"NMI {latest['nmi']:.4f} | Purity {latest['purity']:.4f} | "
              f"loss {latest.get('server_loss') if latest.get('server_loss') is not None else '--'}")
        print(f"Gen: MMD {latest['gen_mmd']:.5f} | SWD {latest['gen_swd']:.5f} "
              f"| NLL {latest['global_nll']:.4f}")
    # Highest ARI of the current (or latest) run.
    if latest:
        rp2 = E / 'results/runs' / f"{latest['group']}_seed{latest['seed']}.json"
        r = load(rp2)
        if r:
            hist = max(r.get('rounds', []), key=lambda x: x.get('ari', -1))
            print(f"Current run highest ARI: {hist['ari']:.4f} (round {hist['round']})")
    if done == 15 and latest:
        print('All 15 runs complete; summary files are in results/.')
    if (E / 'error.txt').exists():
        print('ERROR: ' + (E / 'error.txt').read_text().strip().splitlines()[-1])


if __name__ == '__main__':
    main()
