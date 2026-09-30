# Drift-Robust, Fair, Lightweight Classifier via Evolutionary Hyperparameter Search

Team allies (OPT-26-9838) | OptiForge 2026 | Track 04: Machine Learning & AI

## Problem
Deployed classifiers lose accuracy when data drifts, can behave unevenly across
subgroups, and are often larger and slower than needed. This project uses a genetic
algorithm (GA) to tune a RandomForest against several goals at once instead of
accuracy alone.

## SDG alignment
**SDG 9: Industry, Innovation and Infrastructure.** The search penalizes model size
and rewards stability, so it favors resilient, resource-efficient models that can run
on modest hardware.

## Approach
- **Representation:** an individual is `[n_estimators, max_depth, min_samples_split]`,
  each an integer within fixed ranges (`GENE_RANGES` in `ga_optimizer.py`).
- **Operators:** tournament selection (k=3), uniform crossover, per-gene mutation
  (rate 0.2), elitism (top 2 kept unchanged, so the best score never drops).
- **Fitness (higher is better):**
  `0.4*clean accuracy + 0.4*drift accuracy - 1.0*CV std - 0.3*subgroup gap - 0.1*size penalty`
  - drift accuracy: validation accuracy on a copy shifted by +0.5 with added noise
  - CV std: standard deviation of 3-fold cross-validation accuracy (stability)
  - subgroup gap: accuracy difference between two validation subgroups (fairness proxy)
  - size penalty: `n_estimators * max_depth`, normalised
- **Determinism:** fixed seed (42) for data, drift and the GA loop.
- **Config:** population 20, generations 15 (top of `ga_optimizer.py`).

## Run
```
pip install -r requirements.txt
python ga_optimizer.py     # runs the GA, prints best fitness per generation
pytest -q                  # unit tests
python benchmark.py        # inference latency and memory
```

## Inference benchmark
Measured with `benchmark.py` on individual `[36, 13, 9]` (Google Colab CPU):

| batch | mean ms | ms/sample | peak KB |
|------:|--------:|----------:|--------:|
| 1     | 12.60   | 12.595    | 51.5    |
| 10    | 12.49   | 1.249     | 52.5    |
| 100   | 12.47   | 0.125     | 66.0    |
| 300   | 12.66   | 0.042     | 88.1    |

Latency is nearly flat across batch sizes, which suggests fixed per-call overhead
dominates. Batching lowers the cost per sample.

## Limitations (honest notes)
- Data is synthetic (`make_classification`); results are not from a real domain.
- Drift is simulated with a mean shift and noise, not real distribution change.
- The two "subgroups" come from splitting on one feature, so the fairness check is
  a proxy, not a demographic analysis.
- Fitness weights are design choices, not tuned or proven optimal.

## Attempt log
- Attempt 1: baseline GA (accuracy, drift accuracy, size penalty). Score 35.42/100.
  Gaps found: no tests, no benchmark, declared features not in the repo.
- Attempt 2: added stability and fairness terms in `ga_optimizer.py`, pytest suite,
  `benchmark.py`, type hints, and this README.

## Files
- `ga_optimizer.py` - GA, fitness, data loading
- `tests/test_ga.py` - pytest suite
- `benchmark.py` - latency and memory benchmark
- `requirements.txt`, `.env.example` - dependencies and (empty) config template
