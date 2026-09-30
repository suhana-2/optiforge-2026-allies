"""Drift-robust, fair, lightweight classifier via evolutionary hyperparameter search.

SDG 9 (Industry, Innovation and Infrastructure): resilient, resource-efficient models.
An individual is [n_estimators, max_depth, min_samples_split] for a RandomForest.
"""
import random
from typing import Dict, List, Tuple

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

# ---- CONFIG ----
SEED = 42
POP_SIZE = 20
GENERATIONS = 15
MUTATION_RATE = 0.2
TOURNAMENT_K = 3
ELITE = 2

GENE_RANGES: Dict[str, Tuple[int, int]] = {
    "n_estimators": (10, 200),
    "max_depth": (2, 20),
    "min_samples_split": (2, 20),
}
MAX_SIZE = GENE_RANGES["n_estimators"][1] * GENE_RANGES["max_depth"][1]

Individual = List[int]


def load_data(seed: int = SEED) -> tuple:
    """Return train/val splits, a drift-perturbed val copy, and a subgroup mask."""
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )
    rng = np.random.RandomState(seed)
    X_val_drift = X_val + 0.5 + rng.normal(0, 0.5, X_val.shape)
    group_val = X_val[:, 0] > np.median(X_val[:, 0])
    return X_train, X_val, X_val_drift, y_train, y_val, group_val


def random_individual() -> Individual:
    """Create one random individual within GENE_RANGES."""
    return [random.randint(*GENE_RANGES[k]) for k in GENE_RANGES]


def init_population(size: int) -> List[Individual]:
    """Create the starting population."""
    return [random_individual() for _ in range(size)]


def build_model(ind: Individual, seed: int = SEED) -> RandomForestClassifier:
    """Build a RandomForest from an individual's genes."""
    n_est, depth, min_split = ind
    return RandomForestClassifier(
        n_estimators=n_est,
        max_depth=depth,
        min_samples_split=min_split,
        random_state=seed,
        n_jobs=-1,
    )


def subgroup_gap(model, X_val: np.ndarray, y_val: np.ndarray, group_val: np.ndarray) -> float:
    """Absolute accuracy gap between two subgroups (lower means fairer)."""
    pred = model.predict(X_val)
    acc_a = (pred[group_val] == y_val[group_val]).mean()
    acc_b = (pred[~group_val] == y_val[~group_val]).mean()
    return float(abs(acc_a - acc_b))


def fitness(ind: Individual, data: tuple) -> float:
    """Higher is better: accuracy (clean and drifted) minus instability, unfairness, size."""
    X_train, X_val, X_val_drift, y_train, y_val, group_val = data
    model = build_model(ind)
    cv_std = float(cross_val_score(model, X_train, y_train, cv=3).std())
    model.fit(X_train, y_train)
    clean_acc = model.score(X_val, y_val)
    drift_acc = model.score(X_val_drift, y_val)
    gap = subgroup_gap(model, X_val, y_val, group_val)
    size_penalty = (ind[0] * ind[1]) / MAX_SIZE
    return float(
        0.4 * clean_acc + 0.4 * drift_acc - 1.0 * cv_std - 0.3 * gap - 0.1 * size_penalty
    )


def tournament_select(pop: List[Individual], scores: List[float]) -> Individual:
    """Pick TOURNAMENT_K random individuals and return a copy of the best."""
    idx = random.sample(range(len(pop)), TOURNAMENT_K)
    best = max(idx, key=lambda i: scores[i])
    return pop[best][:]


def crossover(p1: Individual, p2: Individual) -> Individual:
    """Uniform crossover: each gene comes from either parent."""
    return [random.choice(pair) for pair in zip(p1, p2)]


def mutate(ind: Individual) -> Individual:
    """Resample each gene within its range with probability MUTATION_RATE."""
    for i, key in enumerate(GENE_RANGES):
        if random.random() < MUTATION_RATE:
            ind[i] = random.randint(*GENE_RANGES[key])
    return ind


def run_ga(data: tuple, pop_size: int = POP_SIZE, generations: int = GENERATIONS):
    """Run the GA. Returns (best_individual, best_score, per-generation best history)."""
    random.seed(SEED)
    np.random.seed(SEED)
    pop = init_population(pop_size)
    history: List[float] = []
    best_ind, best_score = pop[0], float("-inf")
    for gen in range(generations):
        scores = [fitness(ind, data) for ind in pop]
        ranked = sorted(zip(scores, pop), key=lambda t: t[0], reverse=True)
        best_score, best_ind = ranked[0]
        history.append(best_score)
        print(f"Gen {gen + 1:2d} | best {best_score:.4f} | mean {np.mean(scores):.4f} | {best_ind}")
        new_pop = [ind[:] for _, ind in ranked[:ELITE]]
        while len(new_pop) < pop_size:
            child = crossover(
                tournament_select(pop, scores), tournament_select(pop, scores)
            )
            new_pop.append(mutate(child))
        pop = new_pop
    return best_ind, best_score, history


if __name__ == "__main__":
    best_ind, best_score, _ = run_ga(load_data())
    print("FINAL BEST:", best_ind, "fitness:", round(best_score, 4))
