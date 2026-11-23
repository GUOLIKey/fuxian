# Multi-CA v1 — Method Definition

## 1. Scope

Multi-CA v1 is a **server-side, label-free aggregation of client Gaussian
components**. It does not change client training, client ELBO, client optimizer,
decoder, broadcast, or the number of global components in Stage A.

Stage A uses:

- `target_components = 2`
- `max_components = 2`
- `min_components = 2`
- `allow_early_stop = False`
- merge guard: normalized Bures distance `1.0`

This is a **capacity-matched** comparison: Multi-CA cannot gain from having more
Gaussian components than the author global K=2. A capacity-increasing variant is
reserved for Stage B and must include a same-capacity Flat-GMM control.

## 2. Inputs

For client `c` with `n_c` records and local components `k=1..K_c`:

- local weight `π_{c,k}`;
- local mean `μ_{c,k} ∈ R^10`;
- local diagonal variance `σ²_{c,k} ∈ R^10`;
- client record count `n_c`.

Global component mass before normalization:

```text
a_{c,k} = (n_c / Σ_c n_c) * π_{c,k}
```

Normalize all `a_{c,k}` to sum to 1. This is computed only from client size and
local mixture weight; no true label is used.

## 3. Gaussian 2-Wasserstein distance

For diagonal Gaussians:

```text
G_i = N(μ_i, diag(σ_i²)),  G_j = N(μ_j, diag(σ_j²))

W2²(G_i, G_j) = ||μ_i − μ_j||² + ||σ_i − σ_j||²
```

`σ` is the standard deviation. Variances are clipped below by `1e-8`; NaN/Inf and
negative variances are rejected by `sanitize_gmm`.

A scale-normalized Bures distance is also computed for diagnostics and the merge
guard:

```text
BW_norm(i,j) = sqrt(W2²(i,j)) / (median(σ_i) + median(σ_j))
```

Multi-CA Stage A uses `BW_norm` for pair ordering and `W2²` is recorded in the
merge tree. This prevents a very narrow component from dominating the merge order.

## 4. Agglomerative matching + weighted moment matching

1. Start with one cluster per client component.
2. For each cluster pair, compute **complete-linkage** distance: the maximum
   component-pair `BW_norm` between the two member sets. Raw `W2²` is recorded.
3. Select the pair with the smallest `(BW_norm, W2², cluster_i, cluster_j)`;
   this is deterministic and resolves ties.
4. Merge the pair by weighted moment matching:
   ```text
   w_new = w_i + w_j
   μ_new = (w_i μ_i + w_j μ_j) / w_new
   Σ_new = [w_i(Σ_i + (μ_i−μ_new)(μ_i−μ_new)^T)
            + w_j(Σ_j + (μ_j−μ_new)(μ_j−μ_new)^T)] / w_new
   ```
   Keep the diagonal of `Σ_new`; clip variance to `>= 1e-8`.
5. Repeat until 2 macro components remain.
6. Normalize macro weights to sum to 1.

`merge_events` records step, member indices, weight, raw W2², normalized Bures,
unsafe flag, mean, and variance.

No function accepts `y` or any label.

## 5. Adaptive stopping / capacity

General implementation supports:

- `target_components`: desired final K;
- `max_components`: capacity ceiling;
- `min_components`: lower bound;
- `merge_threshold`: normalized-Bures guard;
- `allow_early_stop`: if true, stop before reaching target when the best merge is
  above threshold; otherwise continue to target.

For Stage A, capacity is fixed at K=2 and `allow_early_stop=False`. The guard still
flags every merge above `1.0` as `unsafe_merge`; no merge is silently called safe.

**Stage A does not test adaptive component-count growth.** That is a Stage B
question and requires Flat-GMM controls at the same K.

## 6. Sampling and decoder ownership

After merging, sample `M=100` latent points from the macro mixture:

```text
k ~ Categorical(w_macro)
z = μ_macro[k] + ε ⊙ σ_macro[k],  ε ~ N(0,I)
```

The author server loop expects `lengths[client]` adjacent component blocks so it
can choose a client decoder per sample. A macro component may contain components
from several clients. Multi-CA assigns the macro component to its **dominant owner
client** (largest summed component mass; ties by lower client id), and records the
dominant fraction. This is the minimal compatible mapping; no server loop
restructuring is done.

## 7. What Multi-CA changes in FedVaDE

Original:

```text
Local GMM → gmm_sampling (raw union) → Z_hat
         → server/client decoders → server ELBO → server GMM
         → broadcast averaged VAE
```

Multi-CA ON:

```text
Local GMM → component flattening with client mass
         → Multi-CA macro K=2 → macro sampling → Z_hat
         → same server/client decoders → same server ELBO → same server GMM
         → same broadcast averaged VAE
```

Only the **server latent sampling distribution** is replaced. Server GMM
initialization, server ELBO, server GMM K=2, client training, and broadcast are
unchanged.

## 8. Complexity

Let `J = Σ_c K_c` be the number of client components (`J≈14` on BC) and `d=10`.

- Pairwise distances: `O(J² d)`.
- One merge: recompute pair distances `O(J²)` with complete-linkage member checks
  `O(J² d)`; at most `J−2` merges.
- Total: `O(J⁴ d)` worst case, but with `J≈14` negligible relative to neural
  training. Memory `O(J²)` for distances plus `O(J)` members.

## 9. Guarantees and tests

- `W2²(G,G)=0`; symmetry; non-negative.
- Weighted moment matching conserves total mass and closed-form mean/variance.
- Merges deterministic; tie-breaking explicit.
- Macro weights normalized; variances positive.
- Macro sampling layout sums to K.
- NaN/Inf/negative-variance inputs rejected.
- No label parameter exists in any merge function.

Unit tests: `tests/test_multica_core.py`; runner: `tests/run_tests.py`.
