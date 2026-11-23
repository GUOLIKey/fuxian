# Feasibility and Warm-Start Audit

The author-delivered `.pk` and the F01 `epoch_100_server.pt` contain **server
weights only**. They do not contain the 10 client models, local GMMs,
optimizers, schedulers, or RNG states. Exact continuation of the historical
federated trajectories is therefore **not possible**.

This experiment uses a controlled warm start: the delivered server
encoder/decoder and GMM are loaded; for each client, the encoder/decoder are
copied and the local GMM is initialized by selecting the top local-class-count
components of the same server GMM. No local GMM is fitted, no pretraining is
run, and no retrained baseline is substituted. The result is a new 2x2
warm-start experiment, not a reproduction of the historical author/F01 runs.

| Model | Source | Server K | Initial ARI | Historical ref | K_micro |
|---|---|---:|---:|---:|---:|
| AUTHOR | `server_model_breast_cancer.pk` | 2 | 0.6317 | 0.6395 | 10 |
| F01 | `epoch_100_server.pt` | 2 | 0.6090 | 0.6140 | 10 |

Client-state hashes are identical between OFF and ON snapshots; the
ON server GMM differs by construction because it is the Multi-CA
macro-micro mixture. See `init_manifest.json` for all hashes.

BC has no independent test split; all metrics are author-protocol in-sample.
