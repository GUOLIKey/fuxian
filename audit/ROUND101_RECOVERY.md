# Round-101 deterministic recovery check

- Checkpoint: `/tmp/f01_101_check/epoch_100_full.pt`
- Next round: 101
- Exact at 1e-6: **True**
- Delta: `{"client_loss": 4.7683715642676816e-08, "server_loss": 0.0, "purity": 0.0, "ari": 0.0}`
- Got: `{"client_loss": 4.866437602043152, "server_loss": 2.112144947052002, "purity": 0.8830436824800376, "ari": 0.5831924944280963}`
- Expected: `{"client_loss": 4.866437554359436, "server_loss": 2.112144947052002, "purity": 0.8830436824800376, "ari": 0.5831924944280963}`

This check restores the full checkpoint (server, 10 clients, optimizers, schedulers, RNG), runs the original round-101 loop, and compares with the F01 rounds.jsonl. It is an interface/path check, not a formal result.
