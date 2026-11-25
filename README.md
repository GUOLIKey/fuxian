# F01 + ACD frozen local-GMM experiment

This experiment loads the real F01 epoch-100 full checkpoint and asks whether
the author Fed-HIRE ACD (`FCPL`) improves local clustering on FedVaDE client
latent features. No FedVaDE training, client reconstruction, or checkpoint
modification was performed.

## Headline result (negative)

ACD over-segments the frozen latent space:

| Method | K (mean) | ARI (4 multi-class clients) | Holdout NLL | Support<5 components | Stability |
|---|---:|---:|---:|---:|---:|
| A1 original K + GMM | 1.4 | 0.461 | 16.10 | 0 | 1.00 |
| A2 BIC original | 6.7 | 0.212 | 15.59 | 1.2 | 0.53 |
| A4 ACD + moments | 91.2 | 0.011 | 30.25 | 85.0 | 0.31 |
| A5 same-K GMM | 91.2 | 0.015 | 38.15 | 78.6 | 0.48 |

ACD found more components than BIC in 30/30 units, but 85/91 components have
support<5 and 85% of its mixture mass sits on those components. Its ARI is near
zero; the same-K ordinary GMM matches it on clustering. A4's Gaussian
parameters are compatible with FedVaDE (30/30 full and holdout interface
checks), but compatibility is not effectiveness.

## Source and caveats

- ACD is the author `FCPL` code at `hongdream/Fed-HIRE`, pinned commit
  `c116c8354f988d4c3080bbd9b1ab685117e78787`, used unmodified.
- The paper's Eqs. (6)-(7) differ from the source in the weight shift,
  reward/penalty granularity, and elimination rule. This negative result is
  for the source version, not a reimplementation of the paper equations.
- Labels are used only post-hoc for ARI/NMI/Purity, component majority and
  minority diagnostics; they never drive ACD, BIC, GMM fitting or selection.
- The frozen encoder saw the original BC training data; holdout NLL only
  measures GMM generalization, not an independent FedVaDE test.

Read [`REPORT.md`](REPORT.md) for full tables, per-client results, paired
contrasts, failure records, and figures. `audit/ACD_SOURCE.md` and
`audit/INTERFACE.md` document the ACD provenance and parameter conversion.

No model weights, full checkpoints, raw datasets or keys are uploaded.
