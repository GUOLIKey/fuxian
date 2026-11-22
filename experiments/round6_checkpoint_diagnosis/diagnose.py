"""Read-only FedVaDE checkpoint inventory, paired evaluation, and state diagnostics."""
from __future__ import annotations
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from argparse import Namespace

ROOT = Path("/root/autodl-tmp/fedvade_project")
OUT = ROOT / "reproduction_runs/round6_checkpoint_diagnosis"
SOURCE = ROOT / "fuxian-round2/FedVaDE_source"
sys.path.insert(0, str(SOURCE))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from dataloader import make_noniid_clients
from experiment import validation
from model import VaDE

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
PARAMS = Namespace(input_dim=31, hid_dim=10, nClusters=2, alpha=.01, cuda=False)
CSV = SOURCE / "dataset/breast_cancer.csv"
EVAL_SEEDS = tuple(range(10))

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None

def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def candidate_paths():
    yield ("author_weight", SOURCE / "model/server_model_breast_cancer.pk", None, "author_provided_state_dict")
    r5 = ROOT / "reproduction_runs/round5_broadcast"
    for arm in ("author", "post_server"):
        run = r5 / f"formal_{arm}_seed0"
        for rnd in (150, 300):
            yield (f"r5_{arm}_{rnd}", run / f"round_{rnd}_full.pt", run, "full_server_client_checkpoint")
    for stage in ("fuxian-round2", "fuxian-round3"):
        for run in sorted((ROOT / stage / "formal").glob("B_*_seed*")):
            path = run / "round_300_server.pt"
            if path.exists():
                yield (f"{stage.replace('fuxian-','r')}_{run.name}", path, run, "server_only_checkpoint")

def load_state(path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "server" in payload:
        state = payload["server"]
        kind = "full" if "clients" in payload else "server_only"
        embedded_round = payload.get("round")
        embedded_seed = payload.get("seed")
    else:
        state = payload
        kind = "bare_state_dict"
        embedded_round = None
        embedded_seed = None
    state = {k: v.detach().cpu().clone() for k, v in state.items()}
    del payload
    gc.collect()
    return state, kind, embedded_round, embedded_seed

def data():
    raw = pd.read_csv(CSV)
    if raw.shape != (569, 32):
        raise ValueError(f"Unexpected CSV shape {raw.shape}")
    X_raw = raw.iloc[:, :-1].to_numpy()
    y = raw.iloc[:, -1].to_numpy(dtype=np.int64)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)
    nc, clients, _, sizes = make_noniid_clients(str(CSV), 10, 200, "breast_cancer", seed=42)
    assert nc == 2 and sum(sizes) == 2129
    # Verify both protocols use the exact same full-dataset fitted StandardScaler.
    for loader in clients:
        x_local = loader.dataset.tensors[0].numpy()
        assert all(np.isfinite(x_local.ravel()))
    all_clients = torch.cat([d.dataset.tensors[0] for d in clients])
    unique_rows = len(np.unique(np.round(all_clients.numpy(), 7), axis=0))
    return raw, X, y, clients, sizes, unique_rows, scaler

def evaluate(model, protocol, X, y, clients, seed):
    torch.manual_seed(seed)
    if protocol == "A_author":
        streams = clients
    else:
        streams = [DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                              batch_size=200, shuffle=False)]
    preds, truth = [], []
    with torch.no_grad():
        for loader in streams:
            for x, labels in loader:
                preds.append(model.predict(x))
                truth.append(labels.numpy())
    pred = np.concatenate(preds)
    true = np.concatenate(truth)
    purity, ari = validation(true, pred)
    nc = len(model.rho_)
    counts = np.bincount(pred, minlength=nc).tolist()
    labels, label_counts = np.unique(true, return_counts=True)
    table = [[int(((true == label) & (pred == c)).sum()) for c in range(nc)] for label in labels]
    return {"ari": float(ari), "purity": float(purity), "n": len(pred), "cluster_counts": counts,
            "true_counts": dict(zip(map(int, labels), map(int, label_counts))), "contingency": table}

def state_diagnostics(model, X, y, identity):
    with torch.no_grad():
        x = torch.from_numpy(X)
        mu, logvar = model.encoder(x)
        recon = model.decoder(mu)
        weights = torch.softmax(model.rho_, dim=0).numpy()
        means = model.mu_c.numpy()
        vars_ = torch.exp(model.log_sigma2_c).numpy()
        logp = torch.log_softmax(model.rho_, dim=0).unsqueeze(0) + model.gaussian_pdfs_log(mu, model.mu_c, model.log_sigma2_c)
        posterior = torch.softmax(logp, dim=1).numpy()
        entropy = -(posterior * np.log(np.maximum(posterior, 1e-30))).sum(axis=1)
        assignment = posterior.argmax(axis=1)
        labels = np.unique(y)
        class_centroids = np.stack([mu.numpy()[y == label].mean(axis=0) for label in labels])
        within = np.sqrt(np.mean([np.sum((mu.numpy()[y == label] - class_centroids[i])**2, axis=1).mean()
                                  for i, label in enumerate(labels)]))
        between = np.linalg.norm(class_centroids[0] - class_centroids[1])
        avg_var = (vars_[0] + vars_[1]) / 2
        delta = means[0] - means[1]
        bhatt = .125 * np.sum(delta**2 / avg_var) + .5 * np.sum(np.log(avg_var / np.sqrt(vars_[0] * vars_[1])))
        gmm = {
            "model": identity, "mixture_weights": json.dumps(weights.tolist()),
            "component_means": json.dumps(means.tolist()),
            "component_variances": json.dumps(vars_.tolist()),
            "component_distance_euclidean": float(np.linalg.norm(delta)),
            "component_distance_scaled": float(np.sqrt(np.sum(delta**2 / avg_var))),
            "component_bhattacharyya_coefficient": float(np.exp(-bhatt)),
            "variance_min": float(vars_.min()), "variance_max": float(vars_.max()),
            "posterior_argmax_occupancy_unique": json.dumps(np.bincount(assignment, minlength=2).tolist()),
            "posterior_entropy_mean": float(entropy.mean()),
            "posterior_entropy_normalized_mean": float(entropy.mean() / math.log(2)),
            "posterior_maxprob_mean": float(posterior.max(axis=1).mean()),
        }
        z = mu.numpy()
        latent = {
            "model": identity, "latent_mu_mean": float(z.mean()), "latent_mu_std": float(z.std()),
            "latent_mu_min": float(z.min()), "latent_mu_max": float(z.max()),
            "latent_logvar_mean": float(logvar.mean()), "latent_logvar_min": float(logvar.min()),
            "latent_logvar_max": float(logvar.max()),
            "class_centroid_distance": float(between), "within_class_rms": float(within),
            "class_separation_ratio": float(between / within) if within > 0 else float("nan"),
            "reconstruction_mse_unique": float(torch.mean((recon - x)**2)),
            "decoder_output_min": float(recon.min()), "decoder_output_max": float(recon.max()),
            "input_min": float(x.min()), "input_max": float(x.max()),
            "latent_norm_p50": float(np.median(np.linalg.norm(z, axis=1))),
            "latent_norm_p99": float(np.quantile(np.linalg.norm(z, axis=1), .99)),
            "latent_norm_max": float(np.linalg.norm(z, axis=1).max()),
        }
    return gmm, latent

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw, X, y, clients, sizes, unique_rows, scaler = data()
    source_hash = {p.name: digest(p) for p in SOURCE.glob("*.py")}
    source_hash["breast_cancer.csv"] = digest(CSV)
    (OUT / "evaluation_protocol.md").write_text(
        "# Shared FedVaDE checkpoint evaluation\n\n"
        "A_author: author CSV, first 31 columns including A1 ID, last column labels; "
        "StandardScaler fitted on all 569 CSV rows before the author's seeded non-IID client "
        "partition (seed 42). Evaluate concatenated 10 client loaders (2129 records, duplicates "
        "weighted), batch size 200, shuffle=True, CPU float32. For each independent evaluation "
        "seed 0..9, reset torch RNG, call unmodified VaDE.predict (sample z from encoder Gaussian, "
        "GMM posterior argmax), then unmodified experiment.validation (labelMapping, Purity, ARI). "
        "These are ten inference draws of each fixed checkpoint, NOT ten training seeds. "
        "Labels enter metrics only during evaluation; author training uses labels in partition "
        "and local GMM component-count initialization. The author checkpoint may have unknown "
        "training scaler/feature/partition provenance.\n\n"
        "B_unique_diagnostic: same CSV, columns, all-data StandardScaler, checkpoint, "
        "VaDE.predict and metrics, but evaluate each of 569 CSV rows exactly once in original "
        "order, batch size 200, shuffle=False. This isolates evaluation duplicate weighting, "
        "but does not remove label-assisted training or all-data scaler exposure; it is "
        "NOT an independent held-out generalization test.\n\n"
        f"CSV SHA256: {source_hash['breast_cancer.csv']}; Python source hashes: {json.dumps(source_hash)}.\n"
        "Prediction uses one stochastic latent sample per record, not encoder-mean argmax. "
        "Posterior/latent/reconstruction diagnostics separately use deterministic encoder means. "
        "Scores across A and B protocols must not be mixed. Table II ten runs likely refer "
        "to training repeats, not these evaluation draws.\n"
    )
    manifest, eval_rows, gmms, latents = [], [], [], []
    candidates = list(candidate_paths())
    for identity, path, run, origin in candidates:
        if not path.exists():
            continue
        state, kind, embedded_round, embedded_seed = load_state(path)
        result = read_json(run / "result.json") if run else None
        command = read_json(run / "command.json") if run else None
        if run and (not result or result.get("status") != "ok"):
            raise ValueError(f"No successful training record: {run}")
        model = VaDE(PARAMS).cpu()
        model.load_state_dict(state, strict=True)
        model.eval()
        if tuple(state["encoder.encoder.0.weight"].shape) != (500,31):
            raise ValueError(f"Input dimension mismatch: {identity}")
        gmm, latent = state_diagnostics(model, X, y, identity)
        gmms.append(gmm)
        latents.append(latent)
        for protocol in ("A_author", "B_unique_diagnostic"):
            for eval_seed in EVAL_SEEDS:
                score = evaluate(model, protocol, X, y, clients, eval_seed)
                eval_rows.append({
                    "model": identity, "protocol": protocol, "evaluation_seed": eval_seed,
                    **{k: (json.dumps(v, sort_keys=True) if isinstance(v,(list,dict)) else v)
                       for k,v in score.items()}})
        shape = {k: list(v.shape) for k,v in state.items()}
        dtype = {k: str(v.dtype) for k,v in state.items()}
        seed = embedded_seed if embedded_seed is not None else (result.get("seed") if result else None)
        round_ = embedded_round
        if round_ is None and kind != "bare_state_dict":
            raise ValueError(f"Checkpoint lacks round metadata: {identity}")
        manifest.append({
            "model": identity, "absolute_path": str(path.resolve()), "sha256": digest(path),
            "bytes": path.stat().st_size, "file_format": kind, "role": "author_supplied_unknown_training_role" if not run else "server_after_optimizer_update",
            "model_keys": json.dumps(list(state.keys())), "tensor_shapes": json.dumps(shape),
            "tensor_dtypes": json.dumps(dtype), "total_parameters": sum(v.numel() for v in state.values()),
            "gmm_components": int(state["rho_"].shape[0]), "latent_dim": int(state["mu_c"].shape[1]),
            "input_dim": int(state["encoder.encoder.0.weight"].shape[1]),
            "known_seed": seed if seed is not None else "UNKNOWN", "known_round": round_ if round_ is not None else "UNKNOWN",
            "seed_evidence": "embedded_checkpoint" if embedded_seed is not None else ("associated_result.json" if result else "UNKNOWN"),
            "round_evidence": "embedded_checkpoint" if embedded_round is not None else "UNKNOWN",
            "source_dir": str((run / "source").resolve()) if run else str(SOURCE.resolve()),
            "source_main_sha256": digest(run / "source/main.py") if run else source_hash["main.py"],
            "dataset_sha256": digest(run / "source/dataset/breast_cancer.csv") if run else source_hash["breast_cancer.csv"],
            "command_json": str(run / "command.json") if run else "UNKNOWN",
            "result_json": str(run / "result.json") if run else "UNKNOWN",
            "training_config": json.dumps(result.get("params",{}),sort_keys=True) if result else "UNKNOWN",
            "missing_provenance": "training seed, round, training partition/scaler, selection rule, optimizer/RNG history" if not run else
                ("client, optimizer, scheduler and RNG state" if kind == "server_only" else "none for current wrapper"),
        })
        print("EVALUATED", identity, kind, "seed", seed, "round", round_, flush=True)
        del model, state
        gc.collect()
    write_csv(OUT / "checkpoint_manifest.csv", manifest)
    write_csv(OUT / "unified_evaluation.csv", eval_rows)
    write_csv(OUT / "gmm_diagnostics.csv", gmms)
    write_csv(OUT / "latent_diagnostics.csv", latents)
    summary = {}
    for item in manifest:
        model = item["model"]
        summary[model] = {}
        for protocol in ("A_author", "B_unique_diagnostic"):
            rows = [r for r in eval_rows if r["model"] == model and r["protocol"] == protocol]
            summary[model][protocol] = {
                "ari_mean": float(np.mean([r["ari"] for r in rows])),
                "ari_std": float(np.std([r["ari"] for r in rows])),
                "ari_min": float(np.min([r["ari"] for r in rows])),
                "ari_max": float(np.max([r["ari"] for r in rows])),
                "purity_mean": float(np.mean([r["purity"] for r in rows])),
                "purity_std": float(np.std([r["purity"] for r in rows])),
                "first_draw_ari": rows[0]["ari"], "first_draw_purity": rows[0]["purity"],
                "n": rows[0]["n"], "first_draw_cluster_counts": json.loads(rows[0]["cluster_counts"]),
                "first_draw_true_counts": json.loads(rows[0]["true_counts"]),
                "first_draw_contingency": json.loads(rows[0]["contingency"]),
            }
    findings = {
        "protocol_summary": {
            "csv_sha256": source_hash["breast_cancer.csv"], "csv_shape": list(raw.shape),
            "input_columns": list(raw.columns[:-1]), "label_column": raw.columns[-1],
            "client_sizes": list(map(int,sizes)), "client_records": int(sum(sizes)),
            "unique_scaled_feature_vectors_in_client_records": int(unique_rows),
            "full_csv_unique_rows": int(len(np.unique(X,axis=0))),
            "scaler_mean_first": float(scaler.mean_[0]),
            "model_predict_path": "encoder sample z -> GMM posterior argmax",
            "model_validation_path": "labelMapping -> per-cluster purity, sklearn adjusted_rand_score",
            "device": "cpu", "dtype": "float32", "evaluation_seeds": list(EVAL_SEEDS),
            "torch": torch.__version__, "numpy": np.__version__, "sklearn": __import__("sklearn").__version__,
            "python": platform.python_version(),
        },
        "summary": summary,
        "gmm": {r["model"]:r for r in gmms},
        "latent": {r["model"]:r for r in latents},
    }
    (OUT / "round6_findings.json").write_text(json.dumps(findings,indent=2,allow_nan=False)+"\n")
    lines = ["# FedVaDE Round 6 checkpoint diagnosis", "",
             "All scores below are ten fixed checkpoint inference draws, not training repeats.",
             "The first draw uses torch seed 0. A protocol weights the author's 2129 client records;",
             "B protocol uses 569 unique original CSV rows. Neither is held-out evaluation.", "",
             "| Checkpoint | A ARI mean±sd | A Purity mean±sd | B ARI mean±sd | B Purity mean±sd |",
             "|---|---:|---:|---:|---:|"]
    for item in manifest:
        name = item["model"]
        a,b = summary[name]["A_author"],summary[name]["B_unique_diagnostic"]
        lines.append(f"| {name} | {a['ari_mean']:.4f}±{a['ari_std']:.4f} | {a['purity_mean']:.4f}±{a['purity_std']:.4f} | {b['ari_mean']:.4f}±{b['ari_std']:.4f} | {b['purity_mean']:.4f}±{b['purity_std']:.4f} |")
    lines += ["", "Author checkpoint training seed, round, partition/scaler, selection rule are UNKNOWN.",
              "A and B evaluation protocols share CSV, model, prediction and metrics; they differ only in",
              "evaluation multiplicity/order. No model training or source modification was performed.",
              "Use the full CSV/JSON files for posterior, latent, decoder and provenance diagnostics."]
    (OUT / "round6_report.md").write_text("\n".join(lines)+"\n")
    print("COMPLETE",len(manifest),"checkpoints",len(eval_rows),"evaluation rows",flush=True)

if __name__ == "__main__":
    main()
