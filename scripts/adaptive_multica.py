"""Adaptive Macro-Micro aggregation on a frozen set of client local GMMs.

Uses the Multi-CA primitives in `multica_core.py`: complete-linkage
normalized-Bures 2-Wasserstein distance, weighted moment matching, and the
same merge guard.  The leaf count is chosen by an elbow in the merge-distance
sequence (largest relative jump), with a minimum of two micros per macro.
Nothing is trained; only the frozen GMM parameters are merged.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from multica_core import (Cluster, _clusters_to_model, _pair_distance,
                          agglomerate, merge_two, sanitize)

EPS = 1e-10


def merge_sequence(weights, means, log_sigma2, component_clients,
                   linkage="complete"):
    """Full agglomerative merge sequence; returns every intermediate state."""
    w, mu, lv = sanitize(weights, means, log_sigma2, normalize=False)
    var = np.exp(lv)
    cc = np.asarray(component_clients, dtype=int)
    clusters = [Cluster(float(w[i]), mu[i].copy(), var[i].copy(), [i],
                        {int(cc[i]): float(w[i])}) for i in range(len(w))]
    snapshots = [copy.deepcopy(clusters)]
    steps, distances = [], []
    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                raw, norm = _pair_distance(clusters[i], clusters[j], mu, var,
                                           linkage)
                key = (norm, raw, i, j)
                if best is None or key < best[0]:
                    best = (key, i, j, raw, norm)
        _, i, j, raw, norm = best
        merged = merge_two(clusters[i], clusters[j])
        n_before = len(clusters)
        steps.append({
            "merge": len(steps) + 1,
            "members_a": list(clusters[i].members),
            "members_b": list(clusters[j].members),
            "merged_members": list(merged.members),
            "distance": float(norm), "raw": float(raw),
            "weight": float(merged.weight), "n_before": n_before,
            "n_after": n_before - 1,
        })
        distances.append(float(norm))
        new = [c for k, c in enumerate(clusters) if k not in (i, j)]
        new.append(merged)
        new.sort(key=lambda c: min(c.members))
        clusters = new
        snapshots.append(copy.deepcopy(clusters))
    return snapshots, steps, distances


def choose_stop(distances, n_components, min_leaves=2):
    """Choose the merge count by the largest relative jump in distance.

    Leaves = n_components - merges.  Never merges below `min_leaves` per
    macro.  If no clear jump exists, merges down to the minimum.
    """
    max_merges = n_components - int(min_leaves)
    if max_merges <= 0 or len(distances) < 2:
        return 0, 1.0
    limit = min(max_merges, len(distances) - 1)
    best_m, best_score = 0, 1.0
    for m in range(1, limit + 1):
        prev = distances[m - 1]
        nxt = distances[m]
        score = (nxt / prev) if prev > 0 else (nxt if nxt > 0 else 1.0)
        if score > best_score:
            best_m, best_score = m, score
    if best_score <= 1.05:
        return max_merges, best_score
    return best_m, best_score


def robust_tau(distances):
    """Robust upper threshold: median + 1.4826 * MAD."""
    arr = np.asarray(distances, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return float(max(med + 1.4826 * mad, 0.0)), med, mad


def elbow_ratio(distances, n_components, min_leaves=2):
    """Diagnostic only: largest relative jump in the merge distances."""
    max_merges = n_components - int(min_leaves)
    if max_merges <= 0 or len(distances) < 2:
        return 1.0
    limit = min(max_merges, len(distances) - 1)
    best = 1.0
    for m in range(1, limit + 1):
        prev = distances[m - 1]
        nxt = distances[m]
        score = (nxt / prev) if prev > 0 else (nxt if nxt > 0 else 1.0)
        best = max(best, score)
    return float(best)


def adaptive_hierarchical(components, k_macro=2, min_micro=2,
                          linkage="complete", guard_threshold=1.0):
    """Build a Macro-Micro GMM with adaptive micro stopping.

    Stop rule: pool the within-macro merge distances, excluding each macro's
    final merge-to-one (which is uninformative), and compute
    tau = median + 1.4826*MAD.  Within each macro, merge the closest pair
    while its normalized Bures distance <= tau; stop at the first larger
    distance.  Never go below `min_micro` micros per macro.  This preserves
    distinct micros instead of forcing the minimum or a fixed K.
    """
    w = components["weights"]
    mu = components["means"]
    lv = components["log_sigma2"]
    cc = components["component_clients"]

    macro_clusters, macro_events = agglomerate(
        w, mu, lv, cc, target=k_macro, linkage=linkage,
        guard_threshold=guard_threshold, use_guard=True, stage="macro")
    macro_clusters.sort(key=lambda c: min(c.members))
    macro_sizes = [len(c.members) for c in macro_clusters]
    macro_masses = [float(c.weight) for c in macro_clusters]

    sequences = []
    for m, parent in enumerate(macro_clusters):
        idx = list(parent.members)
        snapshots, steps, distances = merge_sequence(
            w[idx], mu[idx], lv[idx], cc[idx], linkage=linkage)
        sequences.append({"macro": m, "idx": idx, "snapshots": snapshots,
                          "steps": steps, "distances": distances})

    pooled = []
    for seq in sequences:
        ds = seq["distances"]
        if len(ds) > 1:
            pooled.extend(ds[:-1])
    tau, pooled_med, pooled_mad = robust_tau(pooled)

    micros, macro_ids, stop_info, micro_events = [], [], [], []
    for seq in sequences:
        m = seq["macro"]; idx = seq["idx"]; ds = seq["distances"]
        n = len(idx)
        max_merges = max(0, n - min_micro)
        m_stop = 0
        for d in ds:
            if m_stop >= max_merges:
                break
            if d <= tau:
                m_stop += 1
            else:
                break
        clusters = seq["snapshots"][m_stop]
        for c in clusters:
            c.members = [idx[j] for j in c.members]
        for step in seq["steps"][:m_stop]:
            micro_events.append({
                "stage": f"micro_m{m}",
                "n_before": step["n_before"],
                "members": [idx[j] for j in step["merged_members"]],
                "weight": step["weight"],
                "raw_w2": step["raw"],
                "normalized_bures": step["distance"],
                "unsafe": bool(step["distance"] > guard_threshold),
            })
        micros.extend(clusters)
        macro_ids.extend([m] * len(clusters))
        stop_info.append({
            "macro": m, "n_components": n,
            "component_members": idx,
            "merge_distances": ds,
            "chosen_merges": int(m_stop),
            "leaves": int(len(clusters)),
            "threshold": float(tau),
            "pooled_median": float(pooled_med),
            "pooled_mad": float(pooled_mad),
            "elbow_ratio": elbow_ratio(ds, n, min_micro),
            "min_leaves": int(min_micro),
            "stop_rule": ("merge while normalized Bures <= median+1.4826*MAD "
                          "of non-final within-macro distances"),
        })

    order = sorted(range(len(micros)),
                   key=lambda i: (min(micros[i].members), micros[i].weight))
    micros = [micros[i] for i in order]
    macro_ids = [macro_ids[i] for i in order]
    model = _clusters_to_model(micros, macro_ids, macro_events + micro_events)
    model.update({
        "n_micro": int(len(micros)),
        "n_macro": int(len(set(macro_ids))),
        "macro_allocation": [int(sum(1 for x in macro_ids if x == m))
                             for m in sorted(set(macro_ids))],
        "macro_sizes": macro_sizes,
        "macro_masses": macro_masses,
        "linkage": linkage,
        "guard_threshold": float(guard_threshold),
        "micro_members": [c.members for c in micros],
        "macro_events": macro_events,
        "micro_events": micro_events,
        "stop_info": stop_info,
        "stop_threshold": float(tau),
        "stop_pooled_median": float(pooled_med),
        "stop_pooled_mad": float(pooled_mad),
        "adaptive": True,
    })
    return model


def evaluation_macro_map(clusters, target=2):
    """Label-free 2-cluster grouping used only to score a flat GMM."""
    n = len(clusters)
    if n <= target:
        return np.arange(n, dtype=int)
    w = np.array([c.weight for c in clusters], dtype=np.float64)
    mu = np.stack([c.mean for c in clusters])
    var = np.stack([c.var for c in clusters])
    lv = np.log(np.maximum(var, EPS))
    groups, _ = agglomerate(w, mu, lv, np.zeros(n, dtype=int), target=target,
                            linkage="complete", use_guard=True,
                            stage="evaluation_macro")
    groups.sort(key=lambda c: min(c.members))
    out = np.zeros(n, dtype=int)
    for m, c in enumerate(groups):
        for j in c.members:
            out[j] = m
    return out


def flat_fixed(components, target, linkage="complete"):
    """Flat GMM with exactly `target` leaves and no macro constraint."""
    w = components["weights"]
    mu = components["means"]
    lv = components["log_sigma2"]
    cc = components["component_clients"]
    clusters, events = agglomerate(
        w, mu, lv, cc, target=target, linkage=linkage,
        guard_threshold=1.0, use_guard=True, stage="flat")
    macro_map = evaluation_macro_map(clusters, target=2)
    model = _clusters_to_model(clusters, macro_map, events)
    model.update({
        "n_micro": int(len(clusters)),
        "n_macro": int(len(set(macro_map.tolist()))),
        "linkage": linkage,
        "evaluation_macro_only": True,
        "micro_members": [c.members for c in clusters],
        "merge_events": events,
        "adaptive": False,
        "target_leaves": int(target),
    })
    return model


def union_model(components):
    """The raw union of client components, no merging."""
    return flat_fixed(components, target=len(components["weights"]))
