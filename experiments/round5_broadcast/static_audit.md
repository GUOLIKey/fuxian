# Round 5 FedVaDE parameter flow audit

Author baseline: /root/autodl-tmp/fedvade_project/fuxian-round2/FedVaDE_source
Third-round runner base: /root/autodl-tmp/fedvade_project/fuxian-round3/runner.py
Experiment: CPU, breast_cancer, fixed seed 0, 10 clients, 50 pretrain rounds, 300 training rounds, batch 200, author StepLR.

1. model.py:30-45 vae_avg computes sample-count-weighted averages of all encoder and decoder state_dict tensors, omitting rho_, mu_c, log_sigma2_c. main.py:153-155 loads those averages into server VAE; server GMM stays at its existing value.
2. main.py:194-210 decodes 100 sampled latent points, computes ELBO_Loss_server, and Adam updates the server VAE and its GMM. server_vade_param is a separate CPU tensor dictionary, so the update does not change that prior average.
3. main.py:212-213 downloads server_vade_param to clients, meaning clients receive the pre-server-update weighted average. Their GMM is retained because strict=False loads only VAE keys. B changes only that line to load the same VAE keys from the post-optimizer server state_dict.
4. The post-update server VAE and GMM are used for prediction in the current round (main.py:217-229). At the start of the next round, client VAE training starts from the downloaded values; a fresh weighted average overwrites server VAE again. Thus in A the prior server VAE update is not propagated to client training, but server GMM, optimizer moments and scheduler state persist. In B the server VAE update propagates through the clients. Server GMM is never downloaded in either group.
5. B retains identical model keys/shapes and client-specific GMM. Existing Adam optimizer states persist when weights are replaced, following the author's own broadcast convention; this is a methodological caveat, but not a broken state_dict. No optimizer is recreated or reset.
6. A and B 3-pretrain/2-training-round smoke tests with seed 0 both completed. Round 1 pre- vs post-server VAE max absolute delta = 0.002000004; A clients exactly match pre-server average and B clients exactly match post-server state. First-round server/client GMM weights and prediction occupancy match across groups. Round 2 shows divergent losses and occupancy, consistent with the single changed broadcast. All observed model tensors finite and optimizer state counts intact.
7. Formal runs have only the single broadcast difference plus identical observation/checkpoint instrumentation. No real labels influence training or checkpoint selection. Predictions and ARI/Purity are recorded at fixed rounds 150 and 300. Full checkpoints store server/clients, optimizer/scheduler and RNG state at these rounds. CSV partition signature and source hashes appear in command/result JSON.

Smoke data are diagnostic only, never formal results. Historical author source and results remain untouched.
