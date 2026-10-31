# ViewFence

An implementation of **ViewFence: Trace-Closed Disclosure and Action Synthesis for Stateful Tool Agents**. It enumerates complete finite observation branches, keeps exact rational beliefs over immutable worlds and tool states, and plans safe continuations against a public contract.

## Installation

Run the commands from the repository root in an activated environment.

~~~bash
python -m pip install -e .
~~~

Python 3.11+ is required. The planner uses Fraction and SQLite; declared scientific dependencies provide shared evaluation/plots. No language model is needed for finite-library enumeration.

## Data

~~~bash
python -m viewfence prepare --count 4 --seed 17 --output data/demo.jsonl
~~~

A JSONL case has id, immutable world secrets, rational/decimal prior, initial_states, scorer-only actual_world, actions, goal_states, alpha, planning horizon and separate call_budget. Actions declare a public schema name, cost, support-wide preconditions, and per-world/state rows with exact probability, complete visible event and next_state. Probabilities sum exactly to one; rational strings such as "1/3" are accepted.

Events must encode every modeled recipient, invocation, argument/precision, status, refusal, byte and public timing class the observer distinguishes. Test histories must not change this serializer. configs/small.json fixes alpha 0.60, horizon three, six calls, cost weight 0.05 and a two-second search cap. Optional support_k selects a distinct minimum-secret-support policy instead of a posterior ceiling. progress_values and leaf_weight are public contract data.

Split task/context families before choosing costs or horizons. PrivacyLens and CIMemories provide contextual scenario resources in the paper, but their licensed finite adaptations, inventories and service backends are not bundled. Imported scenarios must compile explicitly to this schema; source dataset counts are not executable-task counts.

## Planning and linked execution

~~~bash
python -m viewfence run --data data/demo.jsonl --config configs/small.json --output runs/main
python -m viewfence.session --data data/demo.jsonl --state-db runs/state.sqlite --output runs/sessions.jsonl
~~~

Every positive-probability complete branch must be safe before authorization. Selected schemas depend on public belief, not actual world. Bayesian updates retain counterfactual tool state and world mass. Terminal contract reward is counted once; boundary value is public progress. Planning depth differs from total calls. strict_completion instead requires all-branch viability and must not be confused with expected completion.

Linked tasks use a common session_id, unique request IDs and consistent world identity/policy. SQLite retains exact posterior fractions and actual sandbox state, reserves before each effect and commits the complete observation. Completed retries return stored results. Interrupted uncertain/busy sessions disable further service instead of resetting the prior. This wrapper executes finite local kernels, not real email/calendar/booking calls.

## Evaluation and paper correspondence

| Paper module or metric | Code entry | Configuration or command |
| --- | --- | --- |
| Complete serializer and exact Bayesian successors | core.key; core.branches; core.safe | complete event and exact kernel |
| Uniform all-branch censor and continuation/strict viability | core.choose | alpha/support-k; horizon/call budget |
| Single completion reward, posterior maximum, violations/support | core.run_case; core.score; core.aggregate | run; plot |
| Oracle-safe completion and unnecessary refusal | core.score; cached full-contract oracle | same finite call budget |
| Exact binary inference AUC and calls/completed task | core.aggregate | labeled cases and complete posteriors |
| Durable complete-history replay/recovery | session.SessionStore | linked execution and recovery test |

The full-contract oracle maximizes safe completion probability at zero call penalty under the same call budget, without actual-world planning access. A positive safe-completion probability defines this implementation's public oracle-feasible set. Its result is cached across methods. A timed-out oracle is unknown, not infeasible, with null output retained and excluded from that denominator. This is not the manuscript's external 800-task inventory.

Policy failures and incomplete tasks stay visible. Calls/completed task divides all attempted calls by completed tasks. maximum_posterior uses the worst observed task/prefix, not a mean. Exact binary AUC requires both secret classes; otherwise it is omitted. Physical solver timing is measured but is outside the logical observation theorem. Compiler/model errors and initial infeasibility are not repaired by selecting favorable branches.

## Baselines and ablations

~~~bash
python -m viewfence suite --data data/demo.jsonl --config configs/small.json --output runs/comparison
~~~

Methods: viewfence, one_step (same complete censor at horizon one), no_defense, body_only, actual_branch. Body-only and actual-branch are unsafe diagnostics, scored against the complete observer. Vary horizon, alpha/support-k, cost and public progress using explicit configs while keeping the action library fixed.

ARIEL, CaMeL, Tattle-Tale, language-model proposal training, the matched public-distance heuristic and real service integrations are not implemented. Exact local checks do not reproduce those baseline rows or the original task inventory. Whole-process RSS and local solver time are not language-model or remote-service measurements.

## Implementation provenance

The inherited core, CLI and statistics modules are finite-model implementation code. The original server experiment backend and raw records were not available in this checkout. Version 0.2 extends and checks this implementation. Generated fixtures and the checks below are functional validation, not evidence for the manuscript's reported performance. No manuscript result table is presented as an implementation benchmark.

## Outputs and verification

Each method writes records.jsonl, metrics.json and run_config.json. Records retain failed cases, seeds and clusters; the manifest retains the command and dependency versions. Evaluator records may contain private truth or local state; they are not a privacy-filtered service API. Runtime data, weights, SQLite databases and figures are excluded from Git.

~~~bash
python -m viewfence.statistics --records runs/main/viewfence/records.jsonl --resamples 2000 --seed 17 --output runs/main/intervals.json
python -m viewfence plot --inputs runs/main/viewfence --metrics maximum_posterior safe_completion --output runs/figures
python -m unittest discover -s tests -v
~~~

Use --reference OTHER/records.jsonl for paired cluster comparisons with identical case IDs. Clusters use the supplied cluster field, otherwise the case ID. Generic cluster intervals do not replace a manuscript-specific multi-seed analysis. Plots and summary.csv use the same aggregate values as metrics.json. Choose fresh output paths on reruns.

Validated on Windows with Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0 and psutil 7.1.0.

The existing MIT code license is retained; datasets and models retain their own licenses.

Executed checks: six regression tests, including rational-string kernel execution and JSON round-trip equality; four-case suites for all five methods; exact stochastic posterior with 1/4 and 3/4 likelihoods; a status-channel counterexample; continuation rescue from a greedy dead end; single completion reward; durable replay and uncertain-session rejection; statistics and plots. Full adapted benchmarks and real service effects were not run.
