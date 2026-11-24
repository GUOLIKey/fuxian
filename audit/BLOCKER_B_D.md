# B/D blocker: no original client states

## Requirement

B and D must add Multi-CA **on top of the original A/C weights**, with the
Macro–Micro structure retained. If Multi-CA needs training, a matched OFF
continuation with the same initialization and conditions must be kept
separate. The A/C baselines must not be overwritten.

The instruction also says: if the real client states are missing, report the
limitation and do **not** reconstruct clients or train from scratch.

## What Multi-CA needs

Multi-CA builds the global Macro–Micro mixture by aggregating the 10 client
local GMMs (each client's `rho_`, `mu_c`, `log_sigma2_c`, together with its
encoder/decoder). It cannot be computed from a global server GMM alone: with
one component set there is no client-level partition to aggregate and no
micro structure to preserve.

Continuing the federated loop additionally needs the client models,
optimizers, schedulers, and RNG state at the same round as the server weight.

## What is actually present

| Artifact | Server state hash (first 20) | Clients? | Model |
|---|---|---|---|
| `server_model_breast_cancer.pk` | `b59b9d5e5129478fa87b` | **no** | A (target) |
| `F01/epoch_100_server.pt` | `650a49cea683618f6687` | **no** | C (target) |
| `F01/last_checkpoint.pt` | `a913b5585c72de8b305c` | yes, 10 | round 300, ARI 0.5138 |
| `round5.../round_300_full.pt` | `392b0f6c0f1915c7ac41` | yes, 10 | round 300, ARI 0.4778 |

- The two target payloads are server-only. AUTHOR has exactly 21 tensors; F01
  epoch-100 has `round` and `server` only.
- The only full checkpoints belong to **different server states** (different
  hashes and different ARIs). Their clients are not A's or C's clients.
- The `federated_clustering_handoff/checkpoints/independent_fedvade/fedvade_BC_*.pt`
  files are a different independent implementation (keys `weights`, `split`,
  `dataset`, `seed`, `standardization_*`), not author VaDE client states.

Therefore no existing artifact contains the client states of either target
model.

## Consequence

- **B and D were not run.** There is no Multi-CA result for them.
- **No matched OFF continuation was run either.** It would require the same
  missing client initialization; without it, an OFF continuation is not the
  original baseline and cannot serve as a control.
- No client state was reconstructed, no local GMM was fitted, and no model was
  trained. The previous warm-start experiment (`multica_pretrained_2x2_v1`)
  is preserved unchanged; it is a different experiment and is not substituted
  here.

## What would unblock B/D

1. **Preferred:** the original client checkpoints.
   - A: the 10 client VaDE state dicts from the author run that produced
     `server_model_breast_cancer.pk`. The delivered weight has no round
     metadata, so the full checkpoint (server + clients + optimizers) is the
     safest.
   - C: the 10 client state dicts at **epoch 100** of the F01 run. The Round 7
     runner saved only the server at epoch 100; `last_checkpoint.pt` is epoch
     300 and is a different model (ARI 0.5138).
2. **Alternative, if exact continuation is not required:** explicitly
   authorize a reconstructed-client warm start with a matched OFF control and
   label it as a new experiment, not as the original model's Multi-CA. This is
   what the instruction currently forbids, so it was not done.
3. **Alternative method (not the paper's Multi-CA):** a server-only
   macro–micro variant fitted to the frozen global latents, with no client
   aggregation. This is a different method and was not run.

The A/C direct-inference results in `REPORT.md` are complete and unaffected by
this blocker.
