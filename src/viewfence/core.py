import functools
import json
import random
from fractions import Fraction
import time


METHODS = ["viewfence", "one_step", "no_defense", "body_only", "actual_branch"]


class SolverTimeout(RuntimeError):
    """The public search budget expired before an action was authorized."""


def normalize(entries):
    total = sum(entries.values())
    if total <= 0:
        raise ValueError("Empty posterior support")
    return tuple(
        (world, state, mass / total) for (world, state), mass in sorted(entries.items()) if mass > 0
    )


def key(event, body_only=False):
    return json.dumps(event.get("body") if body_only else event, sort_keys=True)


def transitions(action, world, state):
    table = action["worlds"][world]
    rows = table.get(state, []) if isinstance(table, dict) else table
    if not rows:
        raise ValueError("Action has no declared transition for a reachable world/state")
    probabilities = [Fraction(str(row["probability"])) for row in rows]
    if any(p < 0 for p in probabilities) or sum(probabilities) != 1:
        raise ValueError("Transition probabilities must form an exact distribution")
    return rows


def permitted(action, belief):
    requirements = action.get("preconditions")
    return requirements is None or all(
        state in requirements[world] for world, state, mass in belief if mass
    )


def branches(belief, action, body_only=False):
    grouped = {}
    for world, state, mass in belief:
        for row in transitions(action, world, state):
            event_key = key(row["event"], body_only)
            group = grouped.setdefault(event_key, {})
            destination = (world, str(row.get("next_state", state)))
            group[destination] = group.get(destination, Fraction(0)) + mass * Fraction(
                str(row["probability"])
            )
    return [
        (event_key, sum(values.values()), normalize(values))
        for event_key, values in sorted(grouped.items())
        if sum(values.values()) > 0
    ]


def safe(belief, secrets, alpha, support_k=None):
    """Evaluate a posterior ceiling or a distinct minimum-secret-support policy."""
    totals = {}
    for world, _, mass in belief:
        secret = secrets[world]
        totals[secret] = totals.get(secret, Fraction(0)) + mass
    return (len(totals) >= support_k if support_k is not None
            else max(totals.values(), default=Fraction(1)) <= alpha)


def completed(case, belief):
    goals = case.get("goal_states", [])
    return bool(goals) and all(state in goals for _, state, mass in belief if mass)


def choose(case, belief, remaining, horizon, method):
    alpha = Fraction(str(case["alpha"]))
    body_only = method == "body_only"
    deadline = time.perf_counter() + case.get("solver_seconds", 2.0)

    @functools.lru_cache(None)
    def search(current, available, depth):
        if time.perf_counter() > deadline:
            raise SolverTimeout("Public solver budget exhausted")
        if completed(case, current):
            return case.get("completion_reward", 1.0), None, True
        if depth == 0 or not available:
            progress = case.get("progress_values", {})
            leaf = sum(float(mass) * progress.get(state, 0) for _, state, mass in current)
            return case.get("leaf_weight", 0.25) * leaf, None, False
        best_value, best_action, viable = 0.0, None, False
        for index in available:
            action = case["actions"][index]
            if not permitted(action, current):
                continue
            outcomes = branches(current, action, body_only)
            checked = outcomes
            if method == "actual_branch":
                checked = [
                    outcome
                    for outcome in outcomes
                    if any(w == case["actual_world"] for w, _, _ in outcome[2])
                ]
            if method != "no_defense" and any(
                not safe(post, case["secrets"], alpha, case.get("support_k")) for _, _, post in checked
            ):
                continue
            after = (
                available
                if action.get("repeatable", False)
                else tuple(i for i in available if i != index)
            )
            future = [search(post, after, depth - 1) for _, _, post in outcomes]
            all_viable = all(entry[2] for entry in future)
            if case.get("strict_completion", False) and not all_viable:
                continue
            reward = action.get("progress_reward", 0.0) - case.get("cost_weight", 0.05) * action.get(
                "cost", 1.0
            )
            value = reward + sum(
                float(probability) * entry[0]
                for (_, probability, _), entry in zip(outcomes, future)
            )
            if value > best_value or (case.get("strict_completion", False) and best_action is None and all_viable):
                best_value, best_action, viable = value, index, all_viable
        return best_value, best_action, viable

    return search(belief, remaining, horizon)


def generate(rng, count):
    for index in range(count):
        actions = []
        for name, reward, stage, next_state in [
            ("inspect", 0.05, ["0"], "inspected"),
            ("finish", 1.0, ["inspected"], "done"),
            ("exact_reply", 1.0, ["0"], "done"),
        ]:
            worlds = []
            for world in range(4):
                body = (
                    str(world)
                    if name == "exact_reply"
                    else str(world % 2)
                    if name == "inspect"
                    else "complete"
                )
                event = {
                    "tool": name,
                    "body": body,
                    "status": "ok",
                    "bytes": 64,
                    "timing_class": 1,
                    "refusal": False,
                }
                worlds.append(
                    {
                        state: [{"event": event, "probability": 1, "next_state": next_state}]
                        for state in stage
                    }
                )
            actions.append(
                {
                    "name": name,
                    "reward": reward,
                    "cost": 1,
                    "worlds": worlds,
                    "preconditions": [stage for _ in range(4)],
                }
            )
        yield {
            "id": str(index),
            "family": f"stateful-task-{index}",
            "secrets": [0, 0, 1, 1],
            "prior": [0.25] * 4,
            "actual_world": rng.randrange(4),
            "actions": actions,
            "initial_states": ["0"] * 4,
            "goal_states": ["done"],
            "alpha": 0.6,
            "horizon": 3,
            "call_budget": 6,
            "cost_weight": 0.05,
            "progress_values": {"inspected": 0.5},
            "strict_completion": False,
        }


def run_case(case, method, initial_belief=None, actual_state=None, step_hook=None):
    """Execute a finite local model using exact, complete-history Bayes updates."""
    states = [str(s) for s in case.get("initial_states", ["0"] * len(case["prior"]))]
    if (len(states) != len(case["prior"]) or len(case["secrets"]) != len(states)
            or any(Fraction(str(p)) < 0 for p in case["prior"])):
        raise ValueError("World, prior and secret arrays must agree and have nonnegative masses")
    belief = initial_belief or normalize(
        {
            (world, state): Fraction(str(mass))
            for world, (state, mass) in enumerate(zip(states, case["prior"]))
        }
    )
    observer = belief
    if not safe(belief, case["secrets"], Fraction(str(case["alpha"])), case.get("support_k")):
        raise ValueError("Initial belief violates the disclosure policy")
    remaining = tuple(range(len(case["actions"])))
    world = case["actual_world"]
    possible_states = [(s, float(mass)) for w, s, mass in belief if w == world]
    rng = random.Random(case.get("seed", str(case["id"])))
    state = actual_state if actual_state is not None else rng.choices(
        [s for s, _ in possible_states], weights=[m for _, m in possible_states])[0]
    events, reward = [], 0.0
    maximum = 0.0
    violation = False
    status = "terminated"
    budget = case.get("call_budget", case["horizon"])
    for step in range(budget):
        totals = {}
        for w, _, mass in observer:
            totals[case["secrets"][w]] = totals.get(case["secrets"][w], 0) + mass
        maximum = max(maximum, float(max(totals.values())))
        if completed(case, observer):
            break
        try:
            _, index, _ = choose(
                case, belief, remaining, 1 if method == "one_step" else min(case["horizon"], budget - step), method
            )
        except SolverTimeout:
            status = "solver_timeout"
            break
        if index is None:
            break
        action = case["actions"][index]
        if step_hook is not None:
            step_hook("reserve", observer, state)
        rows = transitions(action, world, state)
        row = rng.choices(rows, weights=[float(item["probability"]) for item in rows], k=1)[0]
        event = row["event"]
        observed_key = key(event)
        observer = next(post for k, _, post in branches(observer, action) if k == observed_key)
        belief_key = key(event, method == "body_only")
        belief = next(
            post
            for k, _, post in branches(belief, action, method == "body_only")
            if k == belief_key
        )
        state = str(row.get("next_state", state))
        if step_hook is not None:
            step_hook("commit", observer, state)
        violation |= not safe(observer, case["secrets"], Fraction(str(case["alpha"])), case.get("support_k"))
        events.append(event)
        reward += action.get("progress_reward", 0)
        if not action.get("repeatable", False):
            remaining = tuple(i for i in remaining if i != index)
    totals = {}
    for w, _, mass in observer:
        totals[case["secrets"][w]] = totals.get(case["secrets"][w], 0) + mass
    maximum = max(maximum, float(max(totals.values())))
    return {
        "events": events,
        "max_secret_posterior": maximum,
        "violation": violation,
        "reward": reward + (case.get("completion_reward", 1.0) if state in case.get("goal_states", []) else 0),
        "completed": state in case.get("goal_states", []),
        "joint_posterior": [[w, s, float(mass)] for w, s, mass in observer],
        "exact_posterior": [[w, s, str(mass)] for w, s, mass in observer],
        "controller_belief": [[w, s, str(mass)] for w, s, mass in belief],
        "remaining_actions": list(remaining),
        "support_count": len(totals),
        "actual_state": state,
        "status": "completed" if state in case.get("goal_states", []) else status,
        "binary_secret": set(case["secrets"]) == {0, 1},
        "attack_label": case["secrets"][case["actual_world"]],
        "attack_positive_posterior": float(totals.get(1, 0)),
    }


def score(case, result):
    oracle_case = {**case, "cost_weight": 0, "leaf_weight": 0, "completion_reward": 1,
                   "actions": [{**action, "progress_reward": 0} for action in case["actions"]]}
    initial_states = case.get("initial_states", ["0"] * len(case["prior"]))
    initial = normalize({(w, str(s)): Fraction(str(p)) for w, (s, p) in enumerate(zip(initial_states, case["prior"]))})
    if "oracle_completion_probability" in case:
        oracle = case["oracle_completion_probability"]
    else:
        try:
            oracle, _, _ = choose(oracle_case, initial, tuple(range(len(case["actions"]))), case.get("call_budget", case["horizon"]), "viewfence")
        except SolverTimeout:
            oracle = None
        case["oracle_completion_probability"] = oracle
    return {
        "maximum_posterior": result["max_secret_posterior"],
        "policy_violation": int(result["violation"]),
        "completion": int(result["completed"]),
        "safe_completion": int(result["completed"] and not result["violation"]),
        "tool_calls": len(result["events"]),
        "reward": result["reward"],
        "minimum_secret_support": result["support_count"],
        "oracle_safe": int(oracle > 0) if oracle is not None else None,
        "oracle_completion_probability": oracle,
        "unnecessary_refusal": int(oracle > 0 and not result["completed"]) if oracle is not None else None,
        "solver_timeout": int(result["status"] == "solver_timeout"),
        "legal_effects": len(result["events"]) if not result["violation"] else 0,
    }


def aggregate(rows):
    """Use one fixed full-contract oracle denominator and count all attempted calls."""
    complete = sum(row["metrics"]["completion"] for row in rows)
    feasible = [row for row in rows if row["metrics"]["oracle_safe"]]
    calls = sum(row["metrics"]["tool_calls"] for row in rows)
    report = {"calls_per_completed_task": dict(count=complete, mean=calls/complete if complete else None),
            "oracle_safe_completion": dict(count=len(feasible), mean=sum(row["metrics"]["safe_completion"] for row in feasible)/len(feasible) if feasible else None),
            "maximum_posterior": dict(count=len(rows), mean=max(row["metrics"]["maximum_posterior"] for row in rows))}
    report["minimum_secret_support"] = dict(count=len(rows), mean=min(row["metrics"]["minimum_secret_support"] for row in rows))
    labels = [row["result"]["attack_label"] for row in rows]
    if all(row["result"]["binary_secret"] for row in rows) and set(labels) == {0, 1}:
        from sklearn.metrics import roc_auc_score
        report["exact_inference_auc"] = dict(count=len(rows), mean=float(roc_auc_score(labels, [row["result"]["attack_positive_posterior"] for row in rows])))
    return report
