# FedVaDE source and paper method audit

Baseline: user-confirmed author-delivered FedVaDE_source. Author files are read only.
Paper PDF SHA256: `e89b6c91303c0b09e97d260bef6cbb9cefd01284bfc80408e2f60e44284bcc3e`; six pages, no full Appendix in this copy.
`main.py` SHA256: `33bd9927437c444f3f5355813dba8de1896fd59e2eb3fbd9f16960ff3041b87b`.
`model.py` SHA256: `6cdef32b831d090ee79fa952a6f55386bba2e37a6c0c38d6ebc865441c1379d9`.
`dataloader.py` SHA256: `28fc737a243ea550dc4c18ec98928a51af2701132452c68d1e7ef0d1db9c9cad`.

- Encoder/decoder: `model.py` Encoder and Decoder use fully connected widths 500/500/2000 and a latent Gaussian of width 10 in the executed baselines. Decoder returns continuous values without a tabular validity projection. Paper p.3 describes the VAE latent representation.
- Global prior: `VaDE` holds `rho_`, `mu_c`, `log_sigma2_c` for a diagonal Gaussian mixture. The GMM posterior is used for clustering assignment in `predict`; this is distinct from generation.
- Author pretraining: `main.py` calls `pre_train(..., pre_epoch=50)`; `model.py` fits client diagonal GMMs to encoder means. It uses **true local labels to set K_l**, then fits a global GMM on latent samples. This is why the new CTG experiment is labeled `label_blind_fedvade_baseline`.
- Server synthetic training: `main.py` calls `gmm_sampling` on the **client GMM states**, weighted by client record count. It decodes those latent samples with the server and respective client decoders, then optimizes `ELBO_Loss_server`. Paper pp.2–4 describes parameter exchange, latent sampling, decoding and server alignment. The server global GMM prior sampled for this diagnostic is not identical to the training-time mixture of client priors.
- Client/server state: encoder/decoder parameters are weighted-averaged, server VAE is updated with generated samples, and the author code distributes the **pre-server-update** averaged VAE state to clients. GMM parameters are not part of that VAE average.
- Generation vs assignment: (1) draw z from the chosen diagonal GMM, (2) decode z into x, (3) separately compute GMM posterior/argmax for a real encoded x. A generated x has no known true NSP label. GMM-component-conditioned generation is not true-class-conditioned generation.
- No independent tabular generation-quality evaluation was found in the author entry code. The existing `validation` function reports clustering metrics. Paper p.5 discusses synthetic-sample count M but does not provide a held-out rare-mode generation protocol.
- CTG adaptation: replace only the data loader with a fixed preprocessed partition and local K selection with unlabeled BIC over 1–3 components. Keep the author VAE, ELBO, client training, server resampling/aggregation, schedule and parameter distribution. Fixed global K=3 is a preregistered model capacity, not computed from run labels.
- The author Breast Cancer CSV has a sample ID as one of 31 inputs, full-data standardization, overlapping client records, and label-assisted local K. Round 10 Breast Cancer generation results are explicitly `author_protocol_diagnostic`, not held-out test evidence.
