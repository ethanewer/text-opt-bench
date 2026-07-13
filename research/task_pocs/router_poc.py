"""Offline LLM routing on RouterBench's precomputed outcomes.

No LLM or judge is called during this experiment. Accuracy and realized API
cost are the paper-standard RouterBench/LLMRouterBench metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


MODELS = [
    "mistralai/mistral-7b-chat",
    "WizardLM/WizardLM-13B-V1.2",
    "mistralai/mixtral-8x7b-chat",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
]


def split_id(sample_id):
    value = int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:8], "big") % 10
    return "train" if value < 6 else "validation" if value < 8 else "test"


def arrays(frame):
    quality = frame[MODELS].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
    quality = np.clip(quality, 0, 1)
    cost = frame[[f"{model}|total_cost" for model in MODELS]].to_numpy(float)
    return quality, cost


def route_points(probabilities, quality, costs, cost_scale):
    points = []
    for penalty in [0, .03, .1, .3, 1, 3, 10, 30]:
        choices = np.argmax(probabilities - penalty * costs / cost_scale, axis=1)
        rows = np.arange(len(choices))
        points.append({
            "penalty": penalty,
            "accuracy": float(quality[rows, choices].mean()),
            "mean_cost": float(costs[rows, choices].mean()),
        })
    # Remove points dominated in both accuracy and cost.
    frontier = []
    for point in points:
        if not any(other["accuracy"] >= point["accuracy"] and
                   other["mean_cost"] <= point["mean_cost"] and
                   (other["accuracy"] > point["accuracy"] or
                    other["mean_cost"] < point["mean_cost"])
                   for other in points):
            frontier.append(point)
    return sorted(frontier, key=lambda value: value["mean_cost"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-features", type=int, default=6000)
    args = parser.parse_args()
    started = time.perf_counter()
    frame = pd.read_pickle(args.data)
    frame = frame.dropna(subset=["sample_id", "prompt", "eval_name"]).copy()
    frame["split"] = frame.sample_id.map(split_id)
    train = frame[frame.split == "train"]
    test = frame[frame.split == "test"]
    train_quality, train_cost = arrays(train)
    test_quality, test_cost = arrays(test)
    scale = float(np.median(train_cost[train_cost > 0]))

    # Best-single probabilities are constant train accuracies.
    best_single_prob = np.tile(train_quality.mean(axis=0), (len(test), 1))

    # Dataset-mean is deliberately excluded from the intended benchmark API,
    # but diagnoses how much of the dataset can be solved through coarse IDs.
    global_mean = train_quality.mean(axis=0)
    by_dataset = {}
    for dataset, group in train.groupby("eval_name"):
        by_dataset[dataset] = arrays(group)[0].mean(axis=0)
    dataset_prob = np.stack([
        by_dataset.get(name, global_mean) for name in test.eval_name
    ])

    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5), min_df=3,
        max_features=args.max_features, sublinear_tf=True)
    x_train = vectorizer.fit_transform(train.prompt)
    x_test = vectorizer.transform(test.prompt)
    prompt_predictions = []
    for column in range(len(MODELS)):
        labels = (train_quality[:, column] > .5).astype(int)
        if labels.min() == labels.max():
            prompt_predictions.append(np.full(len(test), labels[0], dtype=float))
            continue
        classifier = LogisticRegression(max_iter=150, C=.5, n_jobs=1)
        classifier.fit(x_train, labels)
        prompt_predictions.append(classifier.predict_proba(x_test)[:, 1])
    prompt_prob = np.stack(prompt_predictions, axis=1)

    oracle_prob = test_quality.copy()
    methods = {
        "best_single": best_single_prob,
        "dataset_id_mean_diagnostic": dataset_prob,
        "prompt_tfidf": prompt_prob,
        "oracle": oracle_prob,
    }
    results = []
    for name, probabilities in methods.items():
        results.append({
            "method": name,
            "frontier": route_points(probabilities, test_quality, test_cost, scale),
        })
    print(json.dumps({
        "rows": len(frame),
        "train_rows": len(train),
        "test_rows": len(test),
        "models": MODELS,
        "metric": "accuracy-cost Pareto frontier",
        "results": results,
        "eval_seconds": time.perf_counter() - started,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
