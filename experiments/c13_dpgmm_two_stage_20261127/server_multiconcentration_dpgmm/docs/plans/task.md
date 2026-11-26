| Task | Status | Evidence |
| --- | --- | --- |
| 1. Freeze reusable execution sources | completed | `python3 -m unittest tests.test_source_snapshot -v` |
| 2. Validate server protocol configuration | completed | `python3 -m unittest tests.test_protocol -v` |
| 3. Sample provenance-bearing candidate pool | completed | `CandidatePoolTests` 3/3 pass |
| 4. Fit independent DPGMM granularities | completed | `LevelFitTests` 2/2 pass |
| 5. Derive soft client ownership and samples | completed | `OwnershipAndSamplingTests` 3/3 pass |
| 6. Instrument copied author source | completed | `InstrumentationTests` 2/2 pass |
| 7. Build isolated runner and provenance log | completed | `RunnerContractTests` 3/3 pass |
| 8. Add structural precheck and launcher | completed | `PrecheckTests` 4/4 pass; `bash -n` pass |
| 9. Server smoke and documentation audit | completed | Server: 26/26 unit tests; 3 datasets each 2+3-round smoke; 10 inference draws x 2 populations; one explicit training seed required |
| 10. Separately authorized formal runs | completed | seed1 on 3 datasets; baseline-selected seed4 sensitivity on Iris and Banknote; see `../C_SERVER_DPGMM_RESULTS_20261127.md` |
