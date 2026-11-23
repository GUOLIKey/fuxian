# Stage A analysis notes

- Primary final result: round 300.
- BC has no independent test split; metrics are author-protocol in-sample.
- `gen_*` is generation from the trained server GMM.
- `agg_*` is generation from the final aggregation distribution: Multi-CA macro for B/D, raw client union for A/C.
- Minority metrics use an in-sample classifier and are exploratory proxies.
- Paired effects use same training seed: B-A, D-C, C-A, D-B, interaction.
- No metric was used to select checkpoints.
