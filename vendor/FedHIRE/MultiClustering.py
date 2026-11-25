import statistics

import numpy as np

np.seterr(divide='ignore', invalid='ignore')

MAX_ITER = 100


def similarity(x, C):
    diff = np.sum((C - x) ** 2, axis=1)
    s = 1 / (1 + diff)
    return s


def _initial_k(k_ratio, n):
    k = int(np.round(k_ratio * n))
    return max(1, min(k, n))


def _cluster_sums(X, labels, k):
    d = X.shape[1]
    totalData = np.zeros((k, d), dtype=X.dtype)
    np.add.at(totalData, labels, X)
    cluster_size = np.bincount(labels, minlength=k)
    return totalData, cluster_size


def _centroids(totalData, cluster_size):
    C = np.zeros_like(totalData, dtype=float)
    non_empty = cluster_size > 0
    C[non_empty] = totalData[non_empty] / cluster_size[non_empty, np.newaxis]
    return C


def _score_matrix(X, C, belta, cnt):
    diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]
    dist = np.sum(diff ** 2, axis=2)
    s = 1 / (1 + dist)
    w = 1 / (1 + np.exp(-10 * belta + 5))
    gamma = 1 - cnt / np.sum(cnt)
    return s * w[np.newaxis, :] * gamma[np.newaxis, :]


def _batch_cpl(X, rate, k, random_state=None):
    n, d = np.shape(X)
    rng = np.random.default_rng(random_state)
    seed = rng.choice(n, k, replace=False)

    y = np.full(n, -1, dtype=int)
    y[seed] = np.arange(k)

    unlabeled = y == -1
    if np.any(unlabeled):
        y[unlabeled] = rng.choice(k, size=np.sum(unlabeled), replace=True)

    totalData, cluster_size = _cluster_sums(X, y, k)
    cnt = np.full(k, 1, dtype=float)
    belta = np.full(k, 1, dtype=float)

    for _ in range(MAX_ITER):
        C = _centroids(totalData, cluster_size)
        score = _score_matrix(X, C, belta, cnt)
        new_y = np.argmax(score, axis=1)

        winner_count = np.bincount(new_y, minlength=k).astype(float)
        belta += rate * winner_count
        cnt += winner_count

        if k > 1:
            top2 = np.argpartition(-score, kth=1, axis=1)[:, :2]
            first_is_best = score[np.arange(n), top2[:, 0]] >= score[np.arange(n), top2[:, 1]]
            runner_up = np.where(first_is_best, top2[:, 1], top2[:, 0])
            runner_penalty = np.zeros(k, dtype=float)
            np.add.at(runner_penalty, runner_up, score[np.arange(n), runner_up])
            belta -= rate * runner_penalty

        if np.array_equal(new_y, y):
            break

        y = new_y
        totalData, cluster_size = _cluster_sums(X, y, k)

    index = np.where(cluster_size > 0)[0]
    label = np.full(n, -1, dtype=int)
    theta = []
    for new_label, old_label in enumerate(index):
        member_index = np.where(y == old_label)[0]
        label[member_index] = new_label
        theta.append(totalData[old_label] / cluster_size[old_label])

    if len(theta) == 0:
        return np.empty((0, d)), label, 0

    return np.stack(theta), label, len(index)


def FCPL(X, rate, k_0, random_state=42):
    n, _ = np.shape(X)
    k = _initial_k(k_0, n)
    return _batch_cpl(X, rate, k, random_state)


def calWeight(X, C, label):
    matched_centers = C[label]

    s_matrix = np.sum(X == matched_centers, axis=0)
    w = s_matrix / np.sum(s_matrix, axis=0)
    w = w[np.newaxis, :]

    return w


def labelassign(X, C, w):
    same = (X[:, np.newaxis, :] == C[np.newaxis, :, :]).astype(float)
    weighted_sim = same * w[:, np.newaxis, :]
    similarity = np.sum(weighted_sim, axis=2)
    result = np.argmax(similarity, axis=1)

    return result


def fed_hire(labelData, true_k, random_state=42):
    rng = np.random.default_rng()
    N, d = np.shape(labelData)
    unique_rows = np.unique(labelData, axis=0)

    if true_k > unique_rows.shape[0]:
        return [], [], [], "error"

    idx = rng.choice(unique_rows.shape[0], true_k, replace=False)
    C = unique_rows[idx]

    convergence = False
    w = np.ones((1, d)) * (1 / d)
    result = labelassign(labelData, C, w)
    iter = 0

    while convergence == False:
        iter += 1
        old_result = result
        Centroid = []
        for t in range(true_k):
            index = np.where(result == t)
            if len(index[0]) == 0:
                Centroid.append(np.ones(d) * -1)
                continue
            modes = []
            for i in range(d):
                x = labelData[index, i].tolist()
                modes.append(statistics.mode(x[0]))
            Centroid.append(modes)
        Centroid = np.array(Centroid)
        w = calWeight(labelData, Centroid, old_result)

        result = labelassign(labelData, Centroid, w)
        if np.array_equal(result, old_result):
            convergence = True

        if iter > MAX_ITER:
            break

    predict = result
    return predict


def MCPL(X, rate, k_1):
    n, _ = np.shape(X)
    k = _initial_k(k_1, n)

    Label = []
    K = []
    epoch = 0
    while True:
        epoch += 1
        _, label, length = _batch_cpl(X, rate, k)

        if length == k:
            if epoch == 1:
                Label.append(label)
                K.append(length)
            convergence = True
        else:
            Label.append(label)
            K.append(length)
            k = length
            convergence = False

        if length < 2:
            convergence = True
        if convergence == True:
            break

    return Label, K
