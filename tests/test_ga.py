"""Unit tests for the evolutionary hyperparameter optimizer."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ga_optimizer as ga  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return ga.load_data()


def test_random_individual_within_ranges():
    for _ in range(50):
        ind = ga.random_individual()
        assert len(ind) == 3
        for gene, (lo, hi) in zip(ind, ga.GENE_RANGES.values()):
            assert lo <= gene <= hi


def test_init_population_size():
    assert len(ga.init_population(7)) == 7


def test_crossover_genes_come_from_parents():
    p1, p2 = [10, 2, 2], [200, 20, 20]
    for _ in range(20):
        child = ga.crossover(p1, p2)
        assert all(c in (a, b) for c, a, b in zip(child, p1, p2))


def test_mutation_keeps_genes_in_range():
    for _ in range(100):
        ind = ga.mutate(ga.random_individual())
        for gene, (lo, hi) in zip(ind, ga.GENE_RANGES.values()):
            assert lo <= gene <= hi


def test_load_data_shapes(data):
    X_train, X_val, X_val_drift, y_train, y_val, group_val = data
    assert X_train.shape == (700, 20)
    assert X_val.shape == X_val_drift.shape == (300, 20)
    assert group_val.shape == (300,)
    assert not np.allclose(X_val, X_val_drift)


def test_subgroup_gap_is_valid_probability(data):
    X_train, X_val, _, y_train, y_val, group_val = data
    model = ga.build_model([10, 3, 2]).fit(X_train, y_train)
    assert 0.0 <= ga.subgroup_gap(model, X_val, y_val, group_val) <= 1.0


def test_fitness_returns_finite_float(data):
    score = ga.fitness([10, 3, 2], data)
    assert isinstance(score, float) and np.isfinite(score)


def test_run_ga_is_deterministic_and_never_regresses(data):
    a = ga.run_ga(data, pop_size=4, generations=3)
    b = ga.run_ga(data, pop_size=4, generations=3)
    assert a[0] == b[0] and a[1] == pytest.approx(b[1])
    history = a[2]
    assert all(later >= earlier - 1e-12 for earlier, later in zip(history, history[1:]))
