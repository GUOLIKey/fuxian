# Round 10 split and partition protocol

Official UCI CTG.xls Raw Data sheet, 21 listed physiological features, NSP target (1/2/3 mapped to 0/1/2). CLASS and all expert pattern indicator columns are excluded. Exact duplicate feature rows are grouped before splitting. The test set and validation set are sealed before any scenario is built. StandardScaler fits model-training records only. Training labels are used solely to construct the three controlled client allocations; the model GMM component count is fixed globally at 3 and chosen locally by unlabeled BIC. Original indices and processed arrays are server-only.

IID: stratified round-robin per class. Medium: 15% of rare-class training records to A, 15% to B, 70% to C. Severe: no rare-class records on A or B, all rare-class training records on C. Every training record appears once per scenario.
