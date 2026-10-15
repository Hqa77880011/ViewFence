# ViewFence

Implementation of **Trace-closed disclosure and action synthesis for stateful tool agents**.

Uses rational posterior masses, complete serialized observations and finite-horizon AND-OR planning. Every positive-probability observation branch is checked before selecting a belief-uniform schema.

## Install

Use Python 3.11 or newer.

```bash
git clone https://github.com/Hqa77880011/ViewFence.git
cd ViewFence
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install.

```bash
python -m pip install -e .
```

## Data preparation

Worlds have priors and secret labels. Actions contain per-world observation distributions, public cost and reward. Complete events contain body, tool, refusal, status, byte/precision and timing classes.

```bash
python -m viewfence prepare --count 200 --seed 17 --output data/workload.jsonl
```

The generator creates finite workloads from the formal model. Each JSONL row is one scenario. `--count` controls corpus size and `--seed` controls generation. Imported cases use the same schema. Generate calibration and test scenarios independently.

## Execute and compare

```bash
python -m viewfence run --data data/workload.jsonl --output runs/main
python -m viewfence suite --data data/workload.jsonl --output runs/comparison
```

Select methods with `--methods`. viewfence uses the declared horizon; one_step uses H=1; no_defense removes posterior constraints; body_only drops side fields; actual_branch checks only the realized branch.

## Evaluation and visualization

Maximum secret posterior over the complete trace, policy violation, completion, safe completion, tool calls, reward and planning cost.

Each method writes `records.jsonl` and `metrics.json` with means, sample standard deviations, usable counts and P50/P95/P99. Rates are fractions. Undefined conditional metrics remain unavailable. Method timers exclude reference scoring. RSS includes the Python runtime; serialized sizes measure JSON bytes.

```bash
python -m viewfence plot --inputs runs/comparison/viewfence --output figures
```

Pass multiple method directories to compare them. Plotting writes `summary.csv` and metric PNGs.

## Implementation conventions

The bounded library consumes each action once. Initial belief must satisfy the disclosure cap. Execution samples the declared transitions while planning retains every positive-probability branch. strict_completion optionally requires all branches to reach a goal within the horizon. Imported tools must expose every visible outcome through this schema.


## Uncertainty and paired comparisons

```bash
python -m viewfence.statistics --records runs/main/viewfence/records.jsonl --output results/intervals.json
```

Supply --reference with a matched record file for paired differences. Resampling uses independent case IDs, or explicit cluster IDs when provided. Repeated observations from one source unit must share cluster. Zero-event bounds refer to an event in an independent cluster.
