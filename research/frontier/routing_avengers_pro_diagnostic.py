"""Local Avengers-Pro-style semantic cluster routing diagnostic.

The official code obtains embeddings from an external service.  For a fully
local, reproducible run this adapter uses TF-IDF features, while retaining the
published mechanism: K-means semantic clusters and cluster-wise model
performance/cost trade-offs.  Cluster count is selected on the visible scoring
rows and then frozen for held-out validation.
"""

import json
import statistics
from pathlib import Path

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from bench import heldout


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "bench/tasks/llm_routing/data"
PENALTIES = (0.0, 0.1, 0.3, 1.0, 3.0)


def fit(rows, clusters):
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                 min_df=2, max_features=8192,
                                 sublinear_tf=True, norm="l2")
    matrix = vectorizer.fit_transform(row[0] for row in rows)
    kmeans = MiniBatchKMeans(n_clusters=clusters, random_state=0,
                            n_init=10, batch_size=512).fit(matrix)
    n_models = len(rows[0][1])
    sums_q = [[[0.0, 0.0] for _ in range(n_models)] for _ in range(clusters)]
    for label, (_, quality, costs) in zip(kmeans.labels_, rows):
        for model in range(n_models):
            sums_q[label][model][0] += quality[model]
            sums_q[label][model][1] += costs[model]
    counts = [0] * clusters
    for label in kmeans.labels_:
        counts[label] += 1
    for cluster in range(clusters):
        for model in range(n_models):
            sums_q[cluster][model][0] /= max(1, counts[cluster])
            sums_q[cluster][model][1] /= max(1, counts[cluster])
    return vectorizer, kmeans, sums_q


def score(state, rows, scale):
    vectorizer, kmeans, stats = state
    labels = kmeans.predict(vectorizer.transform(row[0] for row in rows))
    prompt_gaps = []
    accuracies = []
    costs = []
    for label, (_, quality, cost) in zip(labels, rows):
        gaps = []
        for penalty in PENALTIES:
            predicted = stats[label]
            choice = max(range(len(predicted)), key=lambda i:
                         predicted[i][0] - penalty * predicted[i][1] / scale)
            utility = quality[choice] - penalty * cost[choice] / scale
            oracle = max(q - penalty * c / scale for q, c in zip(quality, cost))
            gaps.append(oracle - utility)
            if penalty == 0.3:
                accuracies.append(quality[choice])
                costs.append(cost[choice])
        prompt_gaps.append(sum(gaps) / len(gaps))
    ordered = sorted(prompt_gaps)
    tail = ordered[-max(1, len(ordered) // 10):]
    mean = sum(prompt_gaps) / len(prompt_gaps)
    return {"score": mean,
            "mean_oracle_utility_gap": mean,
            "worst_decile_prompt_gap": sum(tail) / len(tail),
            "avg_accuracy": sum(accuracies) / len(accuracies),
            "avg_cost": sum(costs) / len(costs)}


def main():
    visible = json.loads((DATA / "train.json").read_text())
    validation = heldout.read(DATA / "heldout_val.bin")
    fit_rows = visible["fit"]
    scale = statistics.median(c for _, _, costs in fit_rows for c in costs if c > 0)
    candidates = []
    for clusters in (8, 16, 32, 64):
        state = fit(fit_rows, clusters)
        candidates.append((score(state, visible["score"], scale)["score"],
                           clusters, state))
    _, clusters, state = min(candidates, key=lambda row: row[0])
    print(json.dumps({"clusters": clusters,
                      "embedding": "local TF-IDF adapter",
                      "train": score(state, visible["score"], scale),
                      "validation": score(state, validation, scale)},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
