# C13 server DPGMM multigranularity experiment

This isolated package prepares experimental arm `C_server_dpgmm`:

- client Local GMM: unchanged author/C13 ordinary diagonal GMM;
- server input: latent candidates sampled from client GMMs;
- server partition: three independent DPGMM fits with concentration priors
  `0.1`, `1.0`, and `10.0`;
- final server sample count: 100, allocated 34/33/33 across the three levels;
- client Decoder routing: posterior-responsibility mass by source client;
- server Decoder, VaDE loss, aggregation, broadcast, datasets and evaluation:
  unchanged from the completed C13 A/B protocol.

This is a DPGMM implementation of multigranularity exploration.  It is not a
reproduction of Multi-CA's competitive-penalty learning algorithm and uses no
complete-linkage tree.

## Protocol

Formal settings match the completed A/B experiment:

- Iris: VaDE alpha `0.01`;
- Breast Cancer and Banknote Authentication: VaDE alpha `0.05`;
- 10 clients, latent width 10, batch size 200, learning rate 0.002;
- 50 AE pretraining epochs, 300 federated rounds, sample count 100;
- the launcher requires one explicit `TRAINING_SEED` in 0 through 4 and does
  not treat inference draws as independent training seeds;
- round-300 evaluation on `unique_full` and `legacy_clients` with 10
  cluster-assignment inference seeds per checkpoint.

The DPGMM configuration is explicit in `configs/server_dpgmm.json`.  A
label-free structural smoke must show at least one server call with two or more
different active K values across the three levels.  The gate never reads true
labels or ARI and does not tune concentration from final performance.

## Verification performed on the server

Date: 2026-11-27.  Mac training/smoke was not used.

Server code:
`/root/autodl-tmp/fedvade_project/reproduction_runs/c13_server_dpgmm_multigranularity_v1/review_code/`

Environment: Python server CPU environment with torch 2.0.1+cpu, NumPy 1.25.2
and scikit-learn 1.7.2.

- 26/26 unit tests passed; `scripts/run_formal.sh` passed Bash syntax checking
  and rejects launch when a single training seed is not explicitly supplied.
- Iris, Breast Cancer and Banknote each completed 2 AE pretraining epochs and
  3 federated rounds with status `ok` and zero non-finite events.
- Both evaluation populations completed 10 inference draws for every smoke
  checkpoint.
- Active K by server call (initialization, rounds 1-3):
  - Iris: `[4,3,4]`, `[5,3,3]`, `[4,5,4]`, `[3,3,3]`;
  - Breast Cancer: `[3,2,4]`, `[3,9,3]`, `[2,4,3]`, `[2,3,4]`;
  - Banknote: `[2,2,2]`, `[3,3,6]`, `[3,6,6]`, `[3,4,3]`.

The Banknote initialization call did not produce different K across levels,
although all three training-round calls did.  Different concentration priors
therefore do not guarantee different active K at every call.  The code records
every call rather than claiming otherwise.

Smoke ARI is an engineering-path check after only three rounds and is not
method evidence.

## Formal result status

The seed-1 formal campaign completed 300 rounds on Iris, Breast Cancer and
Banknote.  A post-hoc baseline-selected sensitivity check also completed for
Iris seed 4 and Banknote seed 4.  All five runs report status `ok`, zero
non-finite events and both evaluation populations with 10 inference draws.

The result tables, structural statistics, evidence limits and conclusions are
recorded in `docs/C_SERVER_DPGMM_RESULTS_20261127.md`.  In brief, the server
DPGMM arm does not show a stable improvement over C13: Iris is lower than the
baseline at seeds 1 and 4, Breast Cancer seed 1 is nearly unchanged only under
the overlapping-client evaluation, and Banknote's seed-1 improvement reverses
sharply at seed 4.  The three concentration-conditioned partitions also do not
form a reliably ordered coarse-to-fine hierarchy on individual server calls.

## Preservation

`/Users/cheeseowl/Desktop/科研/fuxian/FedVaDE_source` and the completed
`c13_dpgmm_3datasets_v1` code/results were not modified.  The four author Python
files in `vendor/author/` are frozen byte-identical copies recorded by
`SOURCE_MANIFEST.json`; each run patches only its private output copy.
