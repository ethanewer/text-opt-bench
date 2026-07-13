"""CPU-only ANN recall under deterministic memory and distance budgets."""

from __future__ import annotations

import argparse
import time

import numpy as np

from common import dump


def make_data(seed, n, queries, dim):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(24, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    labels = rng.integers(0, len(centers), size=n)
    data = centers[labels] + .22 * rng.normal(size=(n, dim)).astype(np.float32)
    qlabels = rng.integers(0, len(centers), size=queries)
    query = centers[qlabels] + .22 * rng.normal(size=(queries, dim)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    return data, query


def truth(data, query, k):
    scores = query @ data.T
    return np.argpartition(-scores, k - 1, axis=1)[:, :k]


def recall_at_k(found, expected, k):
    return sum(len(set(a[:k]) & set(b[:k])) for a, b in zip(found, expected)) / (len(found) * k)


def kmeans(data, clusters, iterations, seed):
    rng = np.random.default_rng(seed)
    centroids = data[rng.choice(len(data), clusters, replace=False)].copy()
    for _ in range(iterations):
        assignment = np.argmax(data @ centroids.T, axis=1)
        for cluster in range(clusters):
            members = data[assignment == cluster]
            if len(members):
                centroids[cluster] = members.mean(axis=0)
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8)
    return centroids, np.argmax(data @ centroids.T, axis=1).astype(np.int16)


def ivf(data, query, candidate_budget, seed):
    clusters = max(8, int(np.sqrt(len(data))))
    centroids, assignment = kmeans(data, clusters, 6, seed)
    lists = [np.flatnonzero(assignment == value) for value in range(clusters)]
    found, distances = [], 0
    for q in query:
        order = np.argsort(-(centroids @ q))
        candidates = []
        for cluster in order:
            candidates.extend(lists[int(cluster)].tolist())
            if len(candidates) >= candidate_budget:
                break
        candidates = np.asarray(candidates[:candidate_budget], dtype=np.int32)
        distances += len(candidates) + clusters
        local = candidates[np.argsort(-(data[candidates] @ q))]
        found.append(local)
    index_bytes = centroids.nbytes + assignment.nbytes
    return found, distances, index_bytes


def lsh(data, query, candidate_budget, seed):
    rng = np.random.default_rng(seed)
    bits = 24
    planes = rng.normal(size=(bits, data.shape[1])).astype(np.float32)
    data_codes = (data @ planes.T) > 0
    query_codes = (query @ planes.T) > 0
    found, distances = [], 0
    for q, code in zip(query, query_codes):
        hamming = np.count_nonzero(data_codes != code, axis=1)
        candidates = np.argpartition(hamming, candidate_budget - 1)[:candidate_budget]
        distances += len(candidates)
        found.append(candidates[np.argsort(-(data[candidates] @ q))])
    packed = np.packbits(data_codes, axis=1)
    return found, distances, planes.nbytes + packed.nbytes


def random_search(data, query, candidate_budget, seed):
    rng = np.random.default_rng(seed)
    found = []
    for q in query:
        candidates = rng.choice(len(data), candidate_budget, replace=False)
        found.append(candidates[np.argsort(-(data[candidates] @ q))])
    return found, len(query) * candidate_budget, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12000)
    parser.add_argument("--queries", type=int, default=160)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--candidates", type=int, default=240)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    data, query = make_data(10, args.n, args.queries, args.dim)
    started = time.perf_counter()
    expected = truth(data, query, args.k)
    truth_seconds = time.perf_counter() - started
    results = []
    for name, method in [("random", random_search), ("lsh", lsh), ("ivf", ivf)]:
        started = time.perf_counter()
        found, calls, index_bytes = method(data, query, args.candidates, 99)
        results.append({
            "method": name,
            "recall_at_10": recall_at_k(found, expected, args.k),
            "distance_or_centroid_evaluations": calls,
            "index_bytes": index_bytes,
            "diagnostic_wall_seconds": time.perf_counter() - started,
        })
    dump({
        "n": args.n,
        "queries": args.queries,
        "dim": args.dim,
        "candidate_budget_per_query": args.candidates,
        "truth_seconds": truth_seconds,
        "results": sorted(results, key=lambda value: -value["recall_at_10"]),
    })


if __name__ == "__main__":
    main()
