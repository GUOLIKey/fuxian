# FedVaDE Round 10: Non-IID generation and rare-mode audit

This round tests a generative-coverage hypothesis rather than optimizing Breast Cancer ARI.
All data processing, model training, inference and evaluation ran on the remote server.
No original author material or previous run was modified. Generated samples are not assigned
true clinical labels; all generated-class results below are reference-classifier proxies.

## Phase A: frozen Breast Cancer models

The author protocol contains the ID feature, full-data StandardScaler and repeated client records.
These are in-sample distribution diagnostics, **not held-out generation tests**.

| Weight | MMD (lower) | Sliced Wasserstein (lower) | Generated-to-real NN mean (lower) |
| --- | ---: | ---: | ---: |
| author_weight | 0.0749 | 0.3331 | 1.9144 |
| round7_F01_epoch100 | 0.0655 | 0.3029 | 1.9699 |

The three distances disagree; neither weight is unambiguously better at generation.
Neither ARI nor GMM-component assignment establishes the true class of a generated sample.

## Dataset and protocol

Official [UCI Cardiotocography](https://archive.ics.uci.edu/dataset/193/cardiotocography), DOI 10.24432/C51S4N, CC BY 4.0. Raw workbook SHA256: `d6aa62e82625e59f9d5b05bc8fc52af9ea67bdf2a23f1faca8511d5618fcb421`.
Rows 2126; NSP class counts [1655, 295, 176]; exact feature duplicates 16; duplicate groups with conflicting labels 2.
Train/validation/test sizes: [1360, 339, 427]. Exact duplicate feature vectors are kept within one split. CLASS, expert pattern indicators
and record identifiers are excluded. The scaler is fitted on model-training records only.
The new model is `label_blind_fedvade_baseline`: global K=3 is fixed in advance; local K
is selected by unlabeled BIC on encoder means. The author source originally uses true local
labels to determine K, so this is a documented protocol adaptation, not a literal source rerun.
IID, medium and severe partitions use the same model-training records; only assignment to clients changes.
In the severe case A and B contain no NSP=3 records and C contains all rare-class training records.

## Fixed-model held-out test and generation

Checkpoint selection uses validation reconstruction MSE only; the sealed test split is used once
after all runs finish. Test ARI/Purity and reconstruction are held-out metrics. Generated-class
coverage is **exploratory** because generated samples have no true labels.

| Model | Selected epoch | Test ARI | Test Purity | Test recon MSE | Rare proxy mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| fedvade_iid_seed0 |  | 0.1024 | 0.7799 | 0.0584 | 0.0026 |
| fedvade_medium_seed0 |  | 0.0566 | 0.7799 | 0.0630 | 0.0043 |
| fedvade_severe_seed0 |  | 0.0452 | 0.7799 | 0.0724 | 0.0014 |
| fedvade_severe_seed1 |  | -0.0162 | 0.7799 | 0.0710 | 0.0000 |
| fedvade_severe_seed2 |  | 0.0860 | 0.8033 | 0.0656 | 0.0004 |
| local_A | 150 | -0.0085 | 0.7799 | 0.1001 | 0.0000 |
| centralized | 150 | 0.1203 | 0.7799 | 0.0363 | 0.0032 |

Independent reference-classifier rare-class test precision 1.000, recall 0.861 on 36 held-out records.
A rare-mode proxy requires classifier rare probability >=0.8 and a generated-to-real
nearest-neighbor distance within the 95th percentile of real rare-class nearest-neighbor distances.
This is a conservative, classifier-dependent proxy, not true synthetic precision or recall.

Rare-mode proxy means at training seed 0: IID 0.0026, medium 0.0043, severe 0.0014; local A 0.0000.
Only severe has three independent training seeds. The IID-to-severe comparison is therefore
a controlled single-seed scenario contrast, not a stable trend estimate.

Severe-seed0 rare test latent centroid distance to other classes 4.4912; rare reconstruction MSE 0.1453; mean GMM responsibility [0.4089251756668091, 0.46962037682533264, 0.12145447731018066].
These diagnostics help separate encoder overlap, prior mass, sampling frequency and decoder
quality but do not alone identify a causal failure point.

## Simple baseline gate

Reference reliable=True; severe rare proxy=0.0014; test rare fraction=0.0843; flat-GMM comparison executed=True.
If executed, the K=3/K=6 fits use pooled training latent vectors with the encoder/decoder frozen.
This is a centralized diagnostic upper bound, not a privacy-preserving federated method.

## Research decision

E: evidence remains uncertain; next work should validate the generated-class evaluator and repeat scenario controls across training seeds before choosing ACD, Multi-CA or decoder changes.
ACD and Multi-CA were not implemented. Current evidence does not justify either mechanism
as a proven remedy. If rare-mode weakness is confirmed with a reliable evaluator, inspect latent
class separation and reconstruction first; only if representations are sound but mixture mass
or server sampling fails should server probability aggregation become the first modification.
Simple flat GMM fitting and central/local references are diagnostics, not evidence of novelty.

## Evidence and limits

- `source_audit.md`: author source/paper method and protocol differences.
- `dataset_manifest.json`, `partition_protocol.md`, `partition_manifest.csv`: source, splits and clients.
- `breast_cancer_generation.csv`, `breast_latent_diagnostics.csv`: author-protocol frozen-weight diagnostics.
- `noniid_baseline_metrics.csv`: held-out fixed-checkpoint test clustering/reconstruction.
- `generation_inference_records.csv`, `classwise_generation_metrics.csv`, `rare_mode_coverage.csv`: fixed-budget exploratory generation proxies.
- `latent_diagnostics.csv`, `simple_baseline_comparison.csv`: mechanistic checks and gated simple baseline.
- Per-run `result.json`, `config.json`, `rounds.jsonl`, `*.log` and server-only checkpoints retain failed attempts and provenance.
- Generation validity is judged in standardized tabular feature space. No image metric is used.
- Severe vs IID/medium has only one common training seed; missing-mode recovery has no ground-truth labels for synthetic samples.
- The supplied six-page paper PDF lacks a full Appendix; Table II protocol remains partly unknown.
