# FedVaDE Round 9 optimization

All processing and training ran on the remote Linux server. No author source, data,
or historical checkpoint was modified. This is an exploratory search on the author
Breast Cancer evaluation records, **not independent test performance**.

Seed-0 plan: 9 initialization/pretraining groups
(8 new, one reused),
8 hyperparameter groups; at most 2 full-300
reruns and at most 6 extra training-seed validation runs. M search is not planned
because this already reaches the 24-new-training upper budget.

The author code sets each client GMM component count using the number of unique
true labels. This count remains fixed by the inherited protocol; labels are never
used to rank GMM starts. Ten inference seeds repeat one fixed checkpoint and are
not ten independent training runs.

| Config | Stage | Seed | LR | Alpha | Pretrain | GMM init | Fixed ARI | Exploratory best ARI | Best epoch |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| I01 | initialization | 0 | 0.002 | 0.05 | 50 | kmeans seed 42 n_init=1 | 0.5779 | 0.5779 | 150 |
| I02 | initialization | 0 | 0.002 | 0.05 | 50 | kmeans seed 7 n_init=1 | 0.5095 | 0.5588 | 80 |
| I03 | initialization | 0 | 0.002 | 0.05 | 50 | kmeans seed 123 n_init=1 | 0.5049 | 0.5595 | 80 |
| I04 | initialization | 0 | 0.002 | 0.05 | 50 | kmeans seed 42 n_init=5 | 0.5295 | 0.5798 | 80 |
| I05 | initialization | 0 | 0.002 | 0.05 | 50 | random seed 42 n_init=10 | 0.5185 | 0.5721 | 70 |
| I06 | initialization | 0 | 0.002 | 0.05 | 30 | kmeans seed 42 n_init=1 | 0.3608 | 0.3957 | 50 |
| I07 | initialization | 0 | 0.002 | 0.05 | 75 | kmeans seed 42 n_init=1 | 0.2303 | 0.4441 | 80 |
| I08 | initialization | 0 | 0.002 | 0.05 | 100 | kmeans seed 42 n_init=1 | 0.3410 | 0.5485 | 70 |
| I09 | initialization | 0 | 0.002 | 0.05 | 50 | k-means++ seed 42 n_init=5 | 0.5097 | 0.5558 | 80 |
| H01 | hyperparameter | 0 | 0.002 | 0.02 | 50 | kmeans seed 42 n_init=1 | 0.5243 | 0.5508 | 120 |
| H02 | hyperparameter | 0 | 0.002 | 0.03 | 50 | kmeans seed 42 n_init=1 | 0.5191 | 0.5721 | 70 |
| H03 | hyperparameter | 0 | 0.002 | 0.075 | 50 | kmeans seed 42 n_init=1 | 0.5259 | 0.6013 | 80 |
| H04 | hyperparameter | 0 | 0.0015 | 0.05 | 50 | kmeans seed 42 n_init=1 | 0.5083 | 0.5376 | 40 |
| H05 | hyperparameter | 0 | 0.0008 | 0.05 | 50 | kmeans seed 42 n_init=1 | 0.5344 | 0.5883 | 20 |
| H06 | hyperparameter | 0 | 0.0005 | 0.05 | 50 | kmeans seed 42 n_init=1 | 0.5576 | 0.5835 | 30 |
| H07 | hyperparameter | 0 | 0.0003 | 0.05 | 50 | kmeans seed 42 n_init=1 | 0.5734 | 0.5868 | 80 |
| H08 | hyperparameter | 0 | 0.002 | 0.05 | 50 | kmeans seed 42 n_init=1 | 0.4991 | 0.5754 | 90 |

Round 7 F01 epoch100 exploratory reference: ARI 0.6140.
Author supplied weight reference: approximately 0.6295.
Publication comparison is exploratory: configuration and epoch selection reused
the evaluation labels. No held-out test set was created.

Server-only checkpoints were saved every 10 formal rounds. Full optimizer,
scheduler, client and RNG state was saved at each run final round. More frequent
full checkpoints would exceed the available 27 GB disk budget.

The 300-round extensions are fresh runs with the same seed/configuration, not
exact continuation of their 150-round runs. Pretraining length and GMM starts
are changed one factor at a time relative to F01; the eight H groups vary only
training hyperparameters relative to F01.

Best seed-0 checkpoint: /root/autodl-tmp/fedvade_project/reproduction_runs/round9_fedvade_optimization/phase1/H03/epoch_080_server.pt
ARI 0.6013 ± 0.0058; Purity 0.8889 ± 0.0018; epoch 80; config H03.
Change vs F01 epoch100: -0.0127 ARI.

Best 150-round new configuration: H03, exploratory ARI 0.6013 at epoch 80.
Its initial reconstruction MSE is 0.053405, GMM log likelihood per client record -15.901818.
The reused F01 baseline lacks an initial-state likelihood snapshot, so the
initial likelihood comparison to that baseline remains unavailable.

Across the 16 new seed-0 phase-1 configurations, the
Pearson correlation of initial GMM log likelihood and fixed-150 ARI is 0.5591.
This association does not establish causality.

15 of 16 phase-1 runs lost more
than 0.02 ARI between their exploratory best saved point and epoch 150.

No additional training-seed validation passed the predeclared improvement
gate; training stability beyond seed 0 remains unverified.

## Files

See `experiment_plan.json`, `experiment_configs.csv`,
`initialization_comparison.csv`, `hyperparameter_comparison.csv`,
`all_checkpoint_metrics.csv`, `inference_repeats.csv`,
`multiseed_validation.csv`, `checkpoint_inventory.csv`,
per-run `config.json`, `initialization.json`, `rounds.jsonl`, and raw `.log` files.
The server retains all model weights. GitHub contains only text artifacts.

GitHub publication verified: https://github.com/GUOLIKey/fuxian/commit/016d5a2cf134c4d93dc3d8123fe665d957457304
