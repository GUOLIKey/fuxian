# Round 11B-pre source and checkpoint audit

No neural network training is performed in this round.

## Data

- `data.npz`: OK (`ad73dc9a3ffdbae7e7e1d91f5caac9b98c62b38c63697bd8de3beb896edaef48`)
- `iid_partitions.npz`: OK (`67ab84f489069b3639847edbbaa25781dcf57148464e1b9d31f96725a840ed73`)
- `medium_partitions.npz`: OK (`65ae0f85ce0bca72b1a4c4f3e041cf170727edb1f4516ee70c407781fb754789`)
- `severe_partitions.npz`: OK (`1e694a7d3a3277cb6318ea1c8271a9de002f42ada8d4dc97ea8da94b87cfc04c`)

## Fixed Round 11A evaluator

- Primary classifier: `calibrated_isotonic`
- Tau: 0.5
- Radius: 5.1852652072906436
- Tau rule: validation Youden-best tau=0.50 for calibrated_isotonic; no test labels used
- Radius rule: fallback train-rare q95 (no quantile met val positive>=0.90, negative<=0.05)

## Frozen models

| Model | Seed | Kind | Selected epoch | Global K | Clients | Local Ks | SHA256 |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| fedvade_severe_seed0 | 0 | fedvade | 150 | 3 | 3 | 3;3;3 | `f74c3c0305aa64bd…` |
| fedvade_severe_seed1 | 1 | fedvade | 150 | 3 | 3 | 3;3;3 | `7abb61c424de8a98…` |
| fedvade_severe_seed2 | 2 | fedvade | 150 | 3 | 3 | 3;3;3 | `dfe443457e398626…` |
| centralized | 0 | reference | 150 | 3 | 0 | — | `32e12da255a20753…` |

## Round 11A files reused

- `ROUND11A_REPORT.md`: `a7d1f5dd9fa585438b30e9c68488233a1cb3871fd809330c40302ed0f844170f`
- `evaluation_protocol.md`: `87c9a6d2bfc10b8fdb9bb5d3a9770dab854cf503f4f33366c35641504516c90c`
- `evaluator_config.json`: `eae8a3ddad1c208315d0ac87092297be0039e0130d7f47dabfc6d180a8f55d25`
- `gmm_k_comparison.csv`: `4bbefce981f27f75338e42117d3f9e2d6ab5dcbffa55143469cd6066ad48d606`
- `global_vs_union_mog.csv`: `91d34c47bf8952696ff48501e790383a829b2a9806dc6a9fced0e0311723f4c7`
- `evaluator_calibration.csv`: `3f3d258d6e569a13c78624f4c302a3f041eb7f7f9b4588e207b5f11f9bb94274`
- `reconstruction_diagnostics.csv`: `ffa5f02ad1e13d7aa30ad63ba5c8581d701f820454c573554f60de2abd0965fa`
- `latent_separability.csv`: `743bf14d971a2fa8a8c06ea30b14dea69a2d69168969460df1711beb0bb4a76d`
- `summary.json`: `f4c6773ee37768a1d85af49476473bdc9db7c988bb15f5c29c4c78e408b5d70e`
