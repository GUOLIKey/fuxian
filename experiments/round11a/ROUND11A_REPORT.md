# FedVaDE Round 11A: Frozen-model diagnostics

**No VaDE/FedVaDE training was performed in this round.** All results use frozen Round 10 checkpoints. The CTG test split has been used in prior exploratory analysis; test rows below are exploratory, not new independent test scores.

## Status

- Primary evaluator: calibrated_isotonic, tau=0.50, radius=5.185265 (train-rare self-NN reference).
- Validation positive control (real rare): joint pass 0.7857.
- Validation negative control (real common): joint pass 0.0032.
- Test positive control (exploratory): joint pass 0.8333.
- Round 10-style test positive control (uncalibrated, tau=0.80, test-rare q95 radius): 0.6667.

## E1: evaluator calibration

| Classifier | Split | Rare precision | Rare recall | Rare support |
| --- | --- | ---: | ---: | ---: |
| calibrated_isotonic | val | 0.9565 | 0.7857 | 28.0 |
| calibrated_isotonic | test | 1.0000 | 0.8611 | 36.0 |
| uncalibrated | test | 1.0000 | 0.8611 | 36.0 |

- Chosen tau rule: validation Youden-best tau=0.50 for calibrated_isotonic; no test labels used.
- Chosen radius rule: fallback train-rare q95 (no quantile met val positive>=0.90, negative<=0.05).
- Real rare validation pass at the primary rule: 0.7857; real common false-positive pass: 0.0032.
- The evaluator remains a classifier + distance proxy, not a true synthetic-label metric. It is conservative; the positive-control pass is the ceiling reference.

## E2: real reconstruction and latent separability

| Model | Rare recon MSE | Common recon MSE | Rare/common | Rare recon proxy | Latent silhouette | Latent 1-NN | Rare centroid distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| centralized | 0.0845 | 0.0356 | 2.3732 | 0.6429 | 0.0685 | 0.9027 | 3.8638 |
| fedvade_severe_seed0 | 0.1882 | 0.0690 | 2.7271 | 0.7143 | 0.0939 | 0.8938 | 6.2071 |
| fedvade_severe_seed1 | 0.1716 | 0.0678 | 2.5309 | 0.6786 | 0.0758 | 0.9056 | 5.0659 |
| fedvade_severe_seed2 | 0.1353 | 0.0651 | 2.0770 | 0.6786 | 0.0836 | 0.9086 | 5.3332 |

- `rare recon proxy` is the same calibrated evaluator applied to reconstructed real rare samples; it is not a true label.
- A high rare MSE with high latent separability points toward the decoder; low latent separability points toward the encoder/representation.

## E2/D0: global GMM vs client-union MoG

| Model | Compatibility | CKA mean | Union NLL | Global NLL | Global proxy | Union/server proxy | Union/client proxy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fedvade_severe_seed0 | OK | 0.9991 | 17.9595 | 19.1701 | 0.0130 | 0.0165 | 0.0144 |
| fedvade_severe_seed1 | OK | 0.9993 | 17.7173 | 19.0254 | 0.0021 | 0.0052 | 0.0039 |
| fedvade_severe_seed2 | OK | 0.9996 | 16.8573 | 17.4490 | 0.0117 | 0.0150 | 0.0135 |

- `Union/server` decodes client-union MoG samples with the same server decoder (required D0 comparison).
- `Union/client` decodes each sampled component with its matched client decoder (supplementary compatibility check).
- If compatibility is INCOMPATIBLE or LIMITED, the union comparison must not be interpreted as evidence that aggregation loses rare modes.

## E3: frozen-latent GMM capacity (diagonal, centralized diagnostic)

| Model | K | Val log-lik | BIC | Effective K | Rare proxy | Assigned rare | Class-2 MMD | Class-2 SWD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| centralized | 3 | -15.0986 | 41901.0 | 3 | 0.0192 | 0.0380 | 0.0843 | 0.6596 |
| centralized | 6 | -14.6758 | 40794.0 | 4 | 0.0438 | 0.0536 | 0.0555 | 0.4699 |
| centralized | 10 | -14.4009 | 39943.6 | 8 | 0.0548 | 0.0626 | 0.0762 | 0.5351 |
| fedvade_severe_seed0 | 3 | -16.7270 | 46650.7 | 3 | 0.0248 | 0.0490 | 0.0944 | 0.6148 |
| fedvade_severe_seed0 | 6 | -16.1544 | 45198.6 | 5 | 0.0446 | 0.0606 | 0.0636 | 0.4941 |
| fedvade_severe_seed0 | 10 | -15.7943 | 44411.9 | 8 | 0.0553 | 0.0655 | 0.0669 | 0.4892 |
| fedvade_severe_seed2 | 3 | -15.8915 | 44344.4 | 3 | 0.0271 | 0.0495 | 0.0992 | 0.6813 |
| fedvade_severe_seed2 | 6 | -15.3507 | 42686.2 | 6 | 0.0587 | 0.0763 | 0.0669 | 0.4805 |
| fedvade_severe_seed2 | 10 | -14.9045 | 41994.4 | 9 | 0.0516 | 0.0677 | 0.0384 | 0.3890 |

- GMM fits use frozen training latents only; true labels are never used for fitting or K selection.
- These are centralized mixture-capacity diagnostics, not deployable federated algorithms.

## Answers

1. **Is the evaluator trustworthy?** See validation positive/negative controls above. The primary rule gives positive 0.7857 and negative 0.0032. It is usable as a conservative diagnostic if the positive control is high and the negative control is low; otherwise the evaluator must be revised first.

2. **Can real rare samples be reconstructed?** See the rare vs common reconstruction MSE and rare recon proxy. High rare MSE alone is not proof of decoder failure; compare with latent separability and the linear-probe rows in the CSV.

3. **Global GMM vs client-union MoG?** Compare the three proxy columns. A union/client gain with a union/server loss suggests cross-space decoding or compatibility problems; a union/server gain over global supports the aggregation hypothesis.

4. **K=3 vs K=6 vs K=10?** Compare validation log-likelihood/BIC and rare proxy/class-2 distances. If K=6/10 improves rare proxy without worsening class-2 MMD/SWD, mixture capacity is a plausible bottleneck. If only assigned rare fraction rises while class-2 quality does not, the gain is mainly sampling proportion.

5. **Can we prove the problem is server aggregation?** Only if the union/server comparison is compatible and clearly better than global, or if a flat same-K GMM beats the original global GMM in a controlled way. The current design cannot prove it if compatibility is LIMITED/INCOMPATIBLE.

6. **What evidence supports ACD?** All Round 10 local K values saturated at the K=3 cap; the frozen-latent K capacity test is indirect evidence about local mixture capacity, but this round did not implement ACD. ACD is not yet proven necessary.

7. **What evidence supports Multi-CA?** The union vs global comparison is the relevant evidence. Even if flat K=6/10 helps, Multi-CA must still beat a flat GMM with the same number of components and the same parameters.

8. **Should the base VaDE be optimized first?** If real rare reconstruction is poor and latent separability is weak, or if IID/centralized references also fail, the base encoder/decoder/ELBO is the first target. Round 10 already showed IID and centralized rare coverage near zero.

9. **Which conclusions are exploratory?** All test-split comparisons, the classifier proxy, the union compatibility interpretation, the single-seed IID/medium evidence, and the component-conditioned diagnostics.

10. **Minimal next training experiment?** Start with the direction selected below, using 3 seeds, the same partitions, the same evaluator, and one same-capacity control. Do not start with ACD+Multi-CA combined.

## Provisional recommendation

**优先改善服务器概率聚合 / Multi-CA.** 相同 server Decoder 下，客户端并集 MoG 在 3 个 severe seed 上均高于全局 GMM（增益 +0.0035, +0.0031, +0.0033）；但集中式扁平 K=6/10 的稀有代理也提高约 +0.0316，因此 Multi-CA 必须与同分量数扁平 GMM 正面对照，不能只靠“分量更多”。

The recommendation is a rule-based reading of this exploratory round, not a confirmed causal claim. The server-generated CSVs remain the source of truth; no test-based model selection was used.

## Evidence limits

- No new training; no gradient updates.
- Test split is not independent for new comparisons; all test rows are exploratory.
- Rare coverage is a classifier + distance proxy, not true synthetic class accuracy.
- The client-union MoG may not share the server decoder latent space; compatibility is audited and reported.
- Reference runs have no intermediate checkpoints; formal runs do.
- Single training seed for IID/medium; severe has 3 seeds in Round 10 but this round focuses on severe0/2.
