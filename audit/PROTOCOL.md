# Corrected evaluation protocol

## Scope

This experiment evaluates the two delivered models exactly as they are:

| Group | Weight | Multi-CA | Operation |
|---|---|---|---|
| A | `FedVaDE_source/model/server_model_breast_cancer.pk` | OFF | direct inference only |
| C | `round7.../fine/F01/epoch_100_server.pt` | OFF | direct inference only |
| B | A's weight | ON | **blocked** — no client states |
| D | C's weight | ON | **blocked** — no client states |

A and C are not trained, continued, warmed up, or modified. No client state is
constructed or fitted. The checkpoint files are read only.

## Sources

| Model | sha256 | Payload |
|---|---|---|
| AUTHOR | `168ad5a4cd619ccee527a4a6c4a6816d35166e3a6bb6bc70b4bbbaa838c2ea85` | 21 server tensors (`rho_`, `mu_c`, `log_sigma2_c`, encoder/decoder) |
| F01 | `41f425203990644c6321b68c83b7b5a7343cafe50a56215c7f505775e9912573` | `{round: 100, server: 21 tensors}` |

Neither payload contains client models, local GMMs, optimizers, schedulers, or
RNG state.

## Data and evaluation

- Author partition: `make_noniid_clients(csv, 10, 200, 'breast_cancer', seed=42)`
  — 2129 client records, 569 unique rows, sizes
  `[115, 209, 326, 56, 218, 155, 350, 239, 138, 323]`.
- The model is constructed and loaded **once**.
- For each inference seed: `torch.manual_seed(seed)`, then iterate every client
  `DataLoader` batch by batch and call `model.predict(batch)`. Predictions and
  true labels are concatenated in that order.
- This is the author's own evaluation loop (`main.py`) and the Round 6
  protocol. It is not a one-shot prediction over a concatenated matrix: batch
  boundaries and DataLoader shuffling change the latent-sampling RNG stream.

## Protocol validation

F01 epoch-100 per-seed ARIs under this loop:

```
0.5973312491202104, 0.6063097913439999, 0.6136900737551098,
0.6181640900291924, 0.6182834047722384, 0.6107263814227599,
0.6107263814227599, 0.6121295294170352, 0.6270997306857246,
0.6256158722596407
```

They match `round7.../fine/F01/inference.csv` bit for bit; the mean is
`0.6140076504228672`, the recorded historical value 0.6140. The evaluation
script asserts this and would fail otherwise.

Two independent runs of `evaluate_baselines.py` produced byte-identical
`baselines.json` (md5 `1dae9b84d72361a36e8f71834fe3104b`).

## Metrics

- ARI, Purity, NMI, ACC on the 2129 client records, 10 inference seeds.
- A secondary ARI on the 569 unique rows (same seeds) is reported separately;
  it is not the historical protocol.
- Latent-GMM NLL on the 569 unique rows using the encoder mean.
- Generation: 5000 samples from the server GMM, server decoder; MMD (RBF,
  median-distance bandwidth) and SWD (64 directions) against the 569 real
  standardized rows, 5 generation seeds.
- Per-component support and the lower-weight component's generation metrics.
  The baselines have K=2 only; there is no micro layer, so these are a
  descriptive "rare-component" proxy, not Multi-CA micro retention.

## Command

```
cd /root/autodl-tmp/fedvade_project/experiments/multica_pretrained_corrected_v1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/autodl-tmp/fedvade_project/.venv-cpu/bin/python scripts/evaluate_baselines.py
```

Outputs: `results/baselines.json`, `results/baselines.csv`, `logs/eval.log`.
