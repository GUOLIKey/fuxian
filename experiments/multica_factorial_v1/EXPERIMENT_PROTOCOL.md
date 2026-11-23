# Stage A Experiment Protocol (Pre-registered)

## 1. Dataset stage

Stage A is Breast Cancer only. Stage B (controlled rare-client modes) and Stage C
(CTG) are **not** started in this run.

BC has 569 rows, 31 features (including the author A1 identifier), and 2 classes
(212 minority, 357 majority). The author protocol repeats records across 10
clients (2129 client records total). **There is no independent BC test split.**
All reported BC metrics are therefore author-protocol in-sample metrics unless
explicitly labeled as generated-sample proxies. No generalization claim is allowed.

## 2. 2×2 factorial

| Group | Training | Aggregation |
|---|---|---|
| A | Author config: alpha=0.01, client scheduler per minibatch | original FedVaDE |
| B | Author config | Multi-CA K=2 |
| C | F01 config: alpha=0.05, client scheduler per epoch | original FedVaDE |
| D | F01 config | Multi-CA K=2 |

F01 is taken from
`reproduction_runs/round7_hyperparameter_search/fine/F01/config.json` and
`source.diff`: lr=0.002, alpha=0.05, `schedule=epoch`; no other changes.
F01's historical ARI is exploratory and is not an expectation for this experiment.

## 3. Seeds and pairing

- Training seeds: 0, 1, 2.
- Data partition seed: 42 for all seeds (author dataloader default); all four
  groups for a given seed use the same partition.
- GMM initialization seed: 42 (author local/global sklearn GMM).
- Pretrain: 50 epochs, author protocol, once per training seed.
- A/B/C/D for a seed load the **same pretrain checkpoint**.
- Pretrain includes server/client VAE and local GMMs, plus RNG states.
- Formal 300 rounds.
- Generation inference seeds: 0–4; these are not training seeds.
- Multi-CA macro sampling uses a dedicated RNG
  `seed*100000 + round`; it does not consume the global training RNG.

## 4. Capacity and control

- Multi-CA target/max/min components = 2; author global GMM K=2.
- No component-count increase in Stage A; no capacity-confound claim.
- Same decoder, same M=100 server samples, same clients/data/rounds.
- Flat-GMM K=2 and plain Wasserstein K=2 are included in the **frozen PoC** as
  diagnostics, not as extra formal training arms.

## 5. Main result and checkpoints

- Primary result: round 300.
- Tracked: rounds 100, 200, 300; no best-checkpoint selection.
- Server checkpoint every 25 rounds.
- Full checkpoint (server+clients+optimizers+schedulers+RNG) at 100/200/300.
- Merge tree for ON at 100/200/300.
- `rounds.jsonl` and `metrics.jsonl` every round.

## 6. Metrics

Training/tracking:

- client loss, server loss;
- author-protocol client-record Purity/ARI;
- unique-row deterministic ARI/NMI/Purity/ACC;
- server GMM NLL and weights;
- local component count, macro component count, merge count, unsafe merges,
  max merge distance, max normalized Bures;
- client/server LRs and runtime.

Final analysis:

- ARI, NMI, Purity (main clustering metrics);
- GMM NLL;
- overall MMD/SWD against real BC;
- minority-class proxy fraction, minority-class MMD/SWD;
- merge counts/unsafe merges;
- paired B−A, D−C, C−A, D−B, interaction = (D−C)−(B−A).

Generated-sample minority metrics use an **in-sample** BC classifier and are
exploratory probes, not true rare-mode precision/recall. BC cannot prove rare
socio-clinical submodes; Stage B is required.

## 7. Statistical reporting

- Report every seed, mean ± SD, and same-seed paired differences.
- Three seeds are preliminary stability evidence, not a significance claim.
- Report failures and negative results explicitly.
- No result may be selected by test ARI, rare proxy, or generation metric.

## 8. Stop conditions

Before formal training:

- core unit tests pass;
- four-group smoke forward/backward and checkpoints pass;
- frozen PoC shows the aggregation path changes the server sampling/training
  distribution;
- no source diff changes model.py/dataloader.py/experiment.py.

If a formal run fails, mark FAILED/INTERRUPTED; do not silently reinitialize.
Full checkpoints at 100/200/300 are the only valid resume points.

## 9. Stage B/C (not started)

- Stage B: controlled rare-client partition; same-capacity Flat-GMM controls for
  any K_macro>K_global; adaptive K and stopping can be tested only there.
- Stage C: CTG only after Stage A/B mechanism evidence; no cross-dataset ARI
  comparison.

## 10. Pre-registered hypotheses

- H1: Multi-CA changes server GMM training distribution and component geometry.
- H2: At same K, any improvement over original aggregation is due to moment-matched
  component aggregation, not adaptive stopping (frozen PoC already suggests
  Multi-CA ≡ plain merge at K=2).
- H3: If Multi-CA only improves NLL but not ARI/MMD/SWD, it is not sufficient.
- H4: Any benefit that requires K>2 must be validated with a Flat-GMM K control
  in Stage B; otherwise it is a capacity effect.
