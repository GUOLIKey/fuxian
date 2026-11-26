# Server DPGMM Multigranularity Implementation Plan

> Status: this is the initial implementation plan.  The original local-only
> verification boundary was superseded by explicit authorization to verify and
> run on the server.  Actual formal results and evidence limits are recorded in
> `../C_SERVER_DPGMM_RESULTS_20261127.md`.

**Goal:** Prepare and verify experimental arm `C_server_dpgmm`, which preserves the author client Local GMM and replaces server latent sampling with three DPGMM granularities.

**Architecture:** A pure NumPy/scikit-learn server module samples a provenance-bearing latent candidate pool from the author client mixtures, fits three diagonal DPGMMs with fixed concentration priors, resamples equally across levels and routes each new sample to a client Decoder through responsibility-derived client mass. Exact-once source hooks patch only copied author files and return `y_hat=client_id` with one label block per client, so the author Decoder and loss loop remains unchanged.

**Tech Stack:** Python 3, NumPy, scikit-learn `BayesianGaussianMixture`, PyTorch author runtime, `unittest`, Bash.

---

### Task 1: Freeze reusable execution sources

**Files:**
- Create: `vendor/author/{main.py,model.py,dataloader.py,experiment.py}`
- Create: `vendor/audited_scripts/common/`
- Create: `SOURCE_MANIFEST.json`
- Create: `tests/test_source_snapshot.py`

**Step 1:** Write a failing test that requires the new source manifest, verifies every declared SHA-256 and asserts that the author snapshot equals `/Users/cheeseowl/Desktop/科研/fuxian/FedVaDE_source`.

**Step 2:** Run `python3 -m unittest tests.test_source_snapshot -v`; expect failure because the snapshot and manifest are absent.

**Step 3:** Mechanically copy the four author Python files and the already-audited common runner from the completed A/B package.  Generate explicit hashes without modifying either source directory.

**Step 4:** Rerun the test; expect all source snapshot tests to pass.

### Task 2: Validate the server protocol configuration

**Files:**
- Create: `configs/server_dpgmm.json`
- Create: `scripts/protocol.py`
- Create: `tests/test_protocol.py`

**Step 1:** Write failing tests for the exact concentration list `[0.1,1.0,10.0]`, shared upper bound 10, prune 0.01, diagonal covariance, finite positive convergence controls, three unique level names and no unknown keys.

**Step 2:** Run `python3 -m unittest tests.test_protocol -v`; expect failure because `validate_server_config` does not exist.

**Step 3:** Implement strict validation and the explicit JSON configuration.  Reject implicit defaults in the formal runner.

**Step 4:** Rerun the protocol tests; expect pass.

### Task 3: Sample a provenance-bearing candidate pool

**Files:**
- Create: `scripts/server_multigranularity.py`
- Create: `tests/test_server_multigranularity.py`

**Step 1:** Write failing tests using two synthetic clients.  Require normalized client/local weights, finite diagonal variances, deterministic samples under a fixed seed, valid source-client IDs and rejection of malformed mixtures.

**Step 2:** Run the single candidate-pool test; expect failure because `sample_client_mixture_pool` does not exist.

**Step 3:** Implement the minimal local-RNG sampler.  Do not call labels or mutate NumPy's global RNG.

**Step 4:** Run the candidate-pool tests; expect pass.

### Task 4: Fit independent DPGMM granularities

**Files:**
- Modify: `scripts/server_multigranularity.py`
- Modify: `tests/test_server_multigranularity.py`

**Step 1:** Write failing tests that fit all levels to the identical candidate array, verify each level uses only its declared concentration, prunes and renormalizes active components, produces finite positive variances and records convergence metadata.

**Step 2:** Run the level-fit tests; expect failure because `fit_levels` does not exist.

**Step 3:** Implement `BayesianGaussianMixture` fitting with dedicated per-level seeds and explicit warning capture.  A non-converged or empty active fit is an error, not a silent fallback.

**Step 4:** Run the level-fit tests and then the full module tests; expect pass.

### Task 5: Derive soft client ownership and final samples

**Files:**
- Modify: `scripts/server_multigranularity.py`
- Modify: `tests/test_server_multigranularity.py`

**Step 1:** Write failing tests for responsibility aggregation by source client, row-normalized component-client mass, deterministic level allocation summing to `args.sample`, valid client routing and fixed-seed repeatability.

**Step 2:** Add a synthetic mixed-client component where the expected client mass is not one-hot; verify the implementation does not use dominant-client-only routing.

**Step 3:** Implement `component_client_mass`, balanced level allocation and `sample_multigranular`.  Return author-compatible `(Z_hat, client_ids, [1] * n_clients, metadata)`.

**Step 4:** Run all server module tests; expect pass.

### Task 6: Instrument copied author source exactly once

**Files:**
- Modify: `scripts/protocol.py`
- Create: `tests/test_instrumentation.py`

**Step 1:** Write failing tests against the frozen author source.  Require one initialization hook in `model.py`, one per-round hook in `main.py`, preservation of the original client `GaussianMixture` block, and rejection of duplicate or missing hook points.

**Step 2:** Require the generated source to contain no `complete-linkage`, `AgglomerativeClustering` or linkage call.

**Step 3:** Implement exact string hooks that replace only the two `gmm_sampling` calls with `_server_multigranular_sample`, passing round `-1` during initialization and `epoch` during training.

**Step 4:** Run instrumentation tests; expect pass and inspect the generated unified diffs.

### Task 7: Build the isolated runner and provenance log

**Files:**
- Create: `scripts/server_runtime.py`
- Create: `scripts/run_experiment.py`
- Create: `tests/test_runner_contract.py`

**Step 1:** Write failing contract tests requiring method ID `C_server_dpgmm`, author local GMM, datasets Iris/BC/Banknote, training seeds 0-4, dataset-specific VaDE alpha, explicit server configuration, exclusive output paths and source/data hashes.

**Step 2:** Implement the runtime context, exact source patching, JSONL per-call records, candidate hashes, active K, client-mass matrices, level allocation and non-finite failure records.  Reuse the completed audited trainer without modifying it.

**Step 3:** Implement preparation/checkpoint restoration compatible with the A/B pre-GMM boundary while leaving local fitting untouched.

**Step 4:** Run runner-contract and full unit tests; expect pass.

### Task 8: Add structural precheck and formal launcher

**Files:**
- Create: `scripts/precheck.py`
- Create: `scripts/run_formal.sh`
- Create: `tests/test_precheck.py`

**Step 1:** Write failing tests that require at least two distinct active K values across levels, reject labels and ARI in precheck input/output, and fail on non-convergence or non-finite output.

**Step 2:** Implement the seed-0 structural precheck and receipt.  Formal launch requires a matching receipt hash and never silently changes configuration.

**Step 3:** Add a launcher for seeds 0-4, 50 pretraining epochs, 300 rounds, sample 100 and the same dataset alphas as A/B.  Evaluation must call the existing evaluator at round 300 with 10 inference seeds for both populations.

**Step 4:** Run `bash -n scripts/run_formal.sh` and the full unit suite; expect pass.

### Task 9: Server smoke and documentation audit

**Files:**
- Create: `README.md`
- Create: `REVIEW.md`
- Create: `verification/`

**Step 1:** Run all unit tests in the isolated server environment and save the output.

**Step 2:** Run server CPU smokes with two pretraining epochs, three federated rounds and 10 inference draws on each dataset.  Verify status, round count, finite values, source diffs, active-K records and output hashes.

**Step 3:** Audit that no existing A/B source or result file changed and every server output used an isolated new directory.

**Step 4:** Document the proposed/executed/verified distinction, configuration, method boundary and smoke limitations.  Formal training was launched only after separate user authorization; see the final result report.
