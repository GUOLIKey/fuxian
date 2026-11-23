# Original FedVaDE vs Multi-CA Integration

## 1. Source of truth

Author source:

`/root/autodl-tmp/fedvade_project/fuxian-round2/FedVaDE_source/`

Hashes:

| File | SHA256 |
|---|---|
| `main.py` | `33bd9927437c444f3f5355813dba8de1896fd59e2eb3fbd9f16960ff3041b87b` |
| `model.py` | `6cdef32b831d090ee79fa952a6f55386bba2e37a6c0c38d6ebc865441c1379d9` |
| `dataloader.py` | `28fc737a243ea550dc4c18ec98928a51af2701132452c68d1e7ef0d1db9c9cad` |
| `experiment.py` | `979739dacb8950e074d8cbc775722aaa5ff8c304a1dba13efc64d97d9da4d758` |

The broadcast audit is at
`experiments/author_broadcast_code_audit/BROADCAST_CODE_AUDIT.md`; its label is
`TRANSIENT_MISMATCH_WITH_RECOVERY`. Multi-CA does not "fix" the author broadcast.

## 2. Original server path

1. Clients train local ELBO; local GMMs and client VAEs are updated.
2. `clients_gmm_param` collects local GMMs (`main.py:160-165`).
3. `gmm_sampling(clients_gmm_param, weight, args.sample)` samples from the raw
   client-union mixture (`main.py:166`; `model.py:48-101`).
4. Server and client decoders decode the samples (`main.py:171,196`).
5. `ELBO_Loss_server` trains server VAE + server GMM (`main.py:204-209`).
6. `vae_avg` averages client VAE parameters, skipping all GMM keys
   (`model.py:31-45`); clients get that averaged VAE (`main.py:212-213`).
7. Client local GMMs are retained; the next round continues local training.

The server global GMM is K=2 on BC. It is not broadcast to clients. Client local
K is the author label-assisted `len(unique(local_y))`.

## 3. Multi-CA ON path

Only step 3 is replaced:

```python
# OFF (A/C): exact author call
Z_hat, y_hat, lengths, _, _, _ = gmm_sampling(...)

# ON (B/D): Multi-CA hook
Z_hat, y_hat, lengths, _mc_info = _sample_union(...)
```

The hook:

1. flattens client local components with global weight
   `a_{c,k}=(n_c/Σn)π_{c,k}`;
2. computes W2² and normalized Bures distances;
3. performs deterministic complete-linkage agglomerative matching to K=2;
4. merges by weighted moment matching;
5. samples M=100 from the macro mixture;
6. maps each macro component to its dominant client decoder;
7. returns `Z_hat, y_hat, lengths` in the exact format expected by the author
   server loop.

Everything after step 3 is the author `main.py`/`model.py` code:

- server decoder and client decoders unchanged;
- `ELBO_Loss_server` unchanged;
- server global GMM K=2 unchanged;
- server optimizer/scheduler unchanged;
- broadcast unchanged;
- client ELBO/local GMM updates unchanged.

## 4. What actually changes

| Element | OFF (A/C) | ON (B/D) |
|---|---|---|
| Client training | author | author |
| Client local GMMs | author | author |
| Server sampling distribution | raw client-union mixture | Multi-CA macro K=2 |
| Server GMM K | 2 | 2 |
| Server ELBO | author | author |
| Server GMM init | author pretrain | shared pretrain |
| Broadcast | averaged VAE, no GMM | same |
| Client decoder mapping | per local component | dominant owner per macro component |

Thus Multi-CA is not an extra post-hoc clustering on an unaffected old GMM:
the server's training samples, and therefore server VAE/GMM updates, come from
the macro mixture.

## 5. Capacity-matched decision

Stage A uses K_macro=2. This avoids a component-count confound and needs no extra
Flat-GMM capacity control for the 2×2 comparison.

The price is explicit: Stage A cannot show that fine-grained component-count
preservation helps. The frozen PoC also showed Multi-CA and plain average-linkage
Wasserstein merging are identical at K=2 on the available BC full checkpoints.
Therefore:

- Stage A tests **capacity-matched moment-matched aggregation** end-to-end;
- Stage A does **not** establish a Multi-CA adaptive-stopping advantage over plain
  hierarchical merging;
- Stage B must test adaptive K>2 with same-capacity Flat-GMM controls before any
  fine-grained-mode claim.

## 6. Code diffs

Every run writes `source.diff`. Expected differences:

- A/C: pretrain-load hook and `_observe` only; C additionally moves client
  `StepLR.step()` outside the minibatch loop (F01).
- B/D: above plus the single `gmm_sampling` call replaced by `_sample_union`.
- `model.py`, `dataloader.py`, `experiment.py` are unchanged in all runs.

No author file is modified in place; every run uses an isolated copy.
