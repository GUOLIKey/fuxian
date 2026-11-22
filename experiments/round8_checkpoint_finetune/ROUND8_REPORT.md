# FedVaDE Round 8 weight-initialized fine-tuning

This is not exact continuation from Round 7 epoch 100: its checkpoint has only the server state. Local client GMMs were refitted in that latent space with the author method; all optimizers, schedulers and RNGs were newly initialized and shared across arms. No author source or prior checkpoint was modified.

All ARI-selected checkpoints use the same evaluation labels; these values are exploratory, not independent test performance. Ten inference seeds are repeats of one fixed model, not ten training seeds.

| Group | LR | Alpha | Epoch | ARI mean±sd | Purity mean±sd |
|---|---:|---:|---:|---:|---:|
| FT00 | 0 | 0.05 | 100 | 0.6140±0.0084 | 0.8925±0.0026 |
| FT01 | 0.0003937488086814452 | 0.05 | 5 | 0.6026±0.0063 | 0.8891±0.0020 |
| FT01 | 0.0003937488086814452 | 0.05 | 10 | 0.5814±0.0091 | 0.8823±0.0029 |
| FT01 | 0.0003937488086814452 | 0.05 | 15 | 0.5648±0.0091 | 0.8771±0.0029 |
| FT01 | 0.0003937488086814452 | 0.05 | 20 | 0.5631±0.0077 | 0.8765±0.0025 |
| FT02 | 0.0001968744043407226 | 0.05 | 5 | 0.5860±0.0073 | 0.8838±0.0023 |
| FT02 | 0.0001968744043407226 | 0.05 | 10 | 0.5792±0.0079 | 0.8817±0.0025 |
| FT02 | 0.0001968744043407226 | 0.05 | 15 | 0.5560±0.0062 | 0.8742±0.0020 |
| FT02 | 0.0001968744043407226 | 0.05 | 20 | 0.5580±0.0064 | 0.8749±0.0021 |
| FT03 | 7.874976173628904e-05 | 0.05 | 5 | 0.5560±0.0079 | 0.8742±0.0026 |
| FT03 | 7.874976173628904e-05 | 0.05 | 10 | 0.5595±0.0069 | 0.8753±0.0023 |
| FT03 | 7.874976173628904e-05 | 0.05 | 15 | 0.5567±0.0094 | 0.8745±0.0031 |
| FT03 | 7.874976173628904e-05 | 0.05 | 20 | 0.5558±0.0087 | 0.8742±0.0028 |
| FT04 | 3.937488086814452e-05 | 0.05 | 5 | 0.5535±0.0063 | 0.8735±0.0021 |
| FT04 | 3.937488086814452e-05 | 0.05 | 10 | 0.5532±0.0072 | 0.8733±0.0023 |
| FT04 | 3.937488086814452e-05 | 0.05 | 15 | 0.5542±0.0095 | 0.8736±0.0031 |
| FT04 | 3.937488086814452e-05 | 0.05 | 20 | 0.5548±0.0103 | 0.8738±0.0034 |

FT00 reference: ARI 0.6140±0.0084, Purity 0.8925±0.0026.
Conditional decisions: [{"FT05": "skipped", "reason": "No FT01-FT04 checkpoint met the prespecified improvement/purity gate"}, {"FT06": "skipped", "reason": "FT05 not run"}]
Author supplied weight reference (Round 6): ARI approximately 0.6295 under the same author-protocol evaluation. Checkpoint selection on current labels biases comparison. Generation quality was not measured.

For a strict evaluation, construct train/validation/test splits before model or hyperparameter selection, fit preprocessing on training data only, and report performance once on untouched test records across multiple training seeds.
