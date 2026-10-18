# ViewFence

Implementation of **Trace-closed disclosure and action synthesis for stateful tool agents**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data

Worlds have priors and secret labels. Actions contain per-world observation distributions, public cost and reward. Complete events contain body, tool, refusal, status, byte/precision and timing classes.

The bundled generator writes synthetic finite-model scenarios as JSONL. Fix the seed and keep the generated input for paired comparisons.

```bash
python -m viewfence prepare --count 200 --seed 17 --output data/generated.jsonl
```

## Execution and evaluation

```bash
python -m viewfence run --data data/generated.jsonl --output runs/main
```

## Comparisons and ablations

```bash
python -m viewfence suite --data data/generated.jsonl --output runs/comparison
```

Use `--methods` to select: `viewfence`, `one_step`, `no_defense`, `body_only`, `actual_branch`.

## Outputs

`runs/main/viewfence/records.jsonl` stores per-case results; `metrics.json` stores aggregate measurements. Comparison outputs use one directory per method.

```bash
python -m viewfence plot --inputs runs/comparison/viewfence --output figures
```

Plotting writes `figures/summary.csv` and metric PNGs. Statistical intervals can be computed from a retained record file:

```bash
python -m viewfence.statistics --records runs/main/viewfence/records.jsonl --output results/intervals.json
```
