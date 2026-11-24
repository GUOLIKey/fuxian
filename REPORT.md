# FedVaDE corrected 2×2: A/C direct inference, B/D blocked

**Status:** A and C evaluated directly from the delivered weights with zero
training. B and D **not run**: the original client states do not exist, so
Multi-CA cannot be computed on the original weights without reconstructing
clients, which the instruction forbids.

## 1. Relation to the previous experiment

The previous directory `experiments/multica_pretrained_2x2_v1` continued all
12 runs for 300 more rounds from reconstructed clients; its ARI collapsed and
it is **not** used as the baseline here. It is preserved unchanged.

This experiment separates three things that were previously mixed:

1. **Original weights, direct inference** — A and C below.
2. **Multi-CA on the original weights** — B and D; blocked.
3. **Matched OFF continuation** — not run; would need the same missing client
   initialization and must never replace A/C.

## 2. Original baselines (A/C), direct inference only

Weights are read-only; no pretraining, continuation, warm start, client
reconstruction, or weight modification. Protocol: author client partition
(seed 42, 2129 records), model built once, then for each inference seed
`torch.manual_seed(seed)` and per-batch `model.predict` over the author's
client DataLoaders. This is the author loop and reproduces the recorded F01
per-seed values exactly.

| Group | Model | ARI seed 0 | ARI seed 1 | ARI seed 2 | ARI 0–2 (mean ± SD) | ARI 0–9 (mean ± SD) | Purity | NMI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | Author `.pk` | 0.6258 | 0.6289 | 0.6305 | 0.6284 ± 0.0024 | **0.6295 ± 0.0088** | 0.8973 | 0.5043 |
| C | F01 epoch 100 | 0.5973 | 0.6063 | 0.6137 | 0.6058 ± 0.0082 | **0.6140 ± 0.0088** | 0.8925 | 0.4867 |

| Group | NLL | Gen MMD | Gen SWD | K | Support≥20 | Ghost | Rare-comp mass | Rare MMD | Rare SWD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 17.1594 | 0.19783 | 0.31006 | 2 | 2 | 0 | 0.2999 | 0.32326 | 0.47275 |
| C | 17.6040 | 0.18776 | 0.27281 | 2 | 2 | 0 | 0.2901 | 0.35187 | 0.54232 |

NLL is the latent-GMM NLL on the 569 unique rows. Generation is 5000 samples
from the server GMM through the server decoder, compared with the 569 real
standardized rows (MMD RBF, SWD over 64 directions, 5 generation seeds).
"Rare-comp" is the lower-weight of the two components; with K=2 there is no
micro layer, so it is a descriptive proxy, not Multi-CA micro retention.

### Historical comparison

| Group | Historical reference | Actual (0–9) | Difference |
|---|---:|---:|---:|
| A | 0.6395 | 0.6295 | −0.0100 |
| C | 0.6140 | 0.6140 | 0.0000 |

C reproduces its historical value exactly. A's delivered `.pk` gives 0.6295
under the same author loop; the 0.6395 reference is not exactly reproducible
from this file. No seed was selected to close the gap and no weight was
changed. The previous report's "initial ARI 0.6317" came from a one-shot
prediction over concatenated records, not the author batch loop; it is
superseded by 0.6295 here.

## 3. B/D (Multi-CA): blocked

Multi-CA aggregates the 10 client local GMMs into a global Macro–Micro mixture
(`K_macro=2`, micros retained) and samples it for server training. Both target
payloads are server-only:

- AUTHOR: 21 server tensors only.
- F01 epoch 100: `{round: 100, server: 21 tensors}` only.

The only full checkpoints on the server belong to **different** server states:
F01 round 300 (ARI 0.5138) and the Round 5 reproduction (ARI 0.4778). Their
clients are not A's or C's clients.

Consequently B/D cannot be computed without reconstructing client states or
training from scratch, both explicitly forbidden. B/D have **no numbers**.
No matched OFF continuation was run either, because it would require the same
missing initialization and could not serve as a control for the original
baseline. Details and the artifacts needed to unblock are in
[`audit/BLOCKER_B_D.md`](audit/BLOCKER_B_D.md).

## 4. Reproduction and integrity

- F01 epoch-100 per-seed ARIs match `inference.csv` bit for bit; mean
  `0.6140076504228672`; the script asserts this and fails otherwise.
- Two independent evaluation runs produced byte-identical `baselines.json`
  (md5 `1dae9b84d72361a36e8f71834fe3104b`).
- Source sha256 values are recorded in `results/baselines.json`:
  AUTHOR `168ad5a4…`, F01 `41f42520…`.
- No checkpoint was written; no dataset, weight, or key is included in the
  repository.

## 5. Artifacts

- `results/baselines.json` — full per-seed metrics, per-component support,
  generation seeds, source hashes, payload keys.
- `results/baselines.csv` — compact table.
- `audit/PROTOCOL.md` — exact evaluation protocol and validation.
- `audit/BLOCKER_B_D.md` — why B/D are blocked and what is needed.
- `scripts/evaluate_baselines.py` — inference-only evaluator.
- `monitoring/remote_status.sh` — live panel (A/C DONE, B/D BLOCKED).
- `logs/eval.log` — evaluation log.
