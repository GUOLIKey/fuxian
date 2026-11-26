# C13 server DPGMM multigranularity design

> Status: this document records the pre-execution design.  Its original
> local-only scope was later superseded by explicit authorization for isolated
> server verification and formal runs.  Results are in
> `../C_SERVER_DPGMM_RESULTS_20261127.md`.

## Objective

Implement experimental arm `C_server_dpgmm` without changing the completed
client-DPGMM A/B/C package.  This arm keeps the author's client Local GMM and
replaces the server sampling distribution with three independent DPGMM
partitions of sampled latent points.  It tests the server-side interpretation
of the supervisor's suggestion.  It does not claim to reproduce the original
Multi-CA competitive-penalty algorithm.

## Preserved protocol

- Author source of record: `/Users/cheeseowl/Desktop/科研/fuxian/FedVaDE_source`.
- Client Local GMM remains the author's diagonal `GaussianMixture`, including
  the author's label-count choice of local K.
- Datasets, partitions, model width, optimizer, 50 AE pretraining epochs, 300
  federated rounds, learning rate, VaDE alpha and final sample count match the
  completed A/B experiments.
- Supported training seeds are 0 through 4.  Formal evaluation uses round 300
  and 10 cluster-assignment inference seeds for both `unique_full` and
  `legacy_clients` populations.
- Existing source, code and result directories are immutable inputs.  New code
  and outputs use this sibling experiment directory.

## Server data flow

For initialization and every federated round:

1. Collect the client GMM weights, means and diagonal variances exactly as the
   author implementation does.
2. Use the author's client-size and local-mixture weights to sample a candidate
   latent pool.  Each candidate point retains its source client identifier.
3. Fit three diagonal `BayesianGaussianMixture` models to the same candidate
   latent pool.  They share all settings except the DP concentration prior.
4. Treat the three fits as coarse, medium and fine independent partitions.
   They are not required to be nested and no agglomerative linkage is used.
5. Prune components below the fixed server weight threshold and renormalize
   each level.
6. Draw the final `args.sample` latent points in a deterministic 34/33/33 split
   when `args.sample=100`; the general allocation differs by at most one point.
7. For every active server component, aggregate its candidate-point posterior
   responsibilities by source client.  A newly drawn latent point is routed to
   a client Decoder by sampling from that component's normalized client mass.
8. Return author-loop-compatible latent points, client routing and lengths;
   keep the server Decoder, client Decoder targets and VaDE loss unchanged.

## Initial server configuration

The initial, pre-registered candidate settings are:

- concentration priors: `0.1`, `1.0`, `10.0` for coarse, medium and fine;
- truncation upper bound: `10` components at every level;
- covariance: diagonal;
- active-component weight threshold: `0.01`;
- convergence controls: `max_iter=1000`, `tol=0.001`, `reg_covar=1e-6`;
- dedicated deterministic seeds derived from training seed, round and level.

These are server settings and are separate from the VaDE loss alpha and the
completed client-DPGMM settings.  Before formal training, a structural precheck
on seed 0 may inspect active component counts, convergence and finite values.
It must not inspect labels or ARI.  At least two distinct active component
counts must occur across the three levels for each dataset; otherwise the
method has not produced multiple granularities and formal training must stop.
No silent parameter substitution is permitted.

## Method identity and comparison

`C_server_dpgmm` means author client Local GMM plus server multigranularity
DPGMM.  It is distinct from group `C` in the completed client-DPGMM package,
where ordinary GMM was matched to the client DPGMM component count.

The primary comparison is `C_server_dpgmm` against the existing author-style
baseline A under identical training seed and evaluation population.  Existing
A artifacts are reused only after their protocol and hashes are checked.

## Recorded evidence

Each run records source hashes, full server configuration, candidate-pool hash,
per-round and per-level active K, convergence status, component weights,
client-mass matrices, final level allocation, non-finite events, checkpoint
hashes and evaluation draws.  A run is incomplete if a level has no active
component, client mass cannot be normalized, a fitted value is non-finite, or
the structural multigranularity gate fails.

## Tests

- The server module rejects non-finite inputs, invalid weights and invalid
  configuration.
- The three levels receive the same candidate pool and never use labels.
- Different levels use only their declared concentration prior.
- Active weights and component-to-client masses normalize to one.
- Final sample counts equal `args.sample`; routing indices are valid.
- Fixed inputs and seeds reproduce identical output and provenance.
- A synthetic mixed-client component routes according to soft responsibility
  mass rather than an arbitrary dominant-client-only rule.
- Source instrumentation replaces the server sampling call exactly once and
  leaves the client Local GMM block unchanged.
- Static checks reject `complete`, `linkage` and agglomerative clustering in the
  new server implementation.
- An isolated server smoke runs preparation, a few rounds and 10 inference
  draws without modifying completed experiments.

## Scope boundary

The initial design task prepared code only.  Formal server training was later
authorized separately and wrote to isolated output directories.  Neither the
design nor the completed runs overwrite prior experiments, claim that DPGMM is
the original Multi-CA algorithm, or select the server concentrations from final
ARI.
