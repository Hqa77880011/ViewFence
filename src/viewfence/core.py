import functools
import json
import random
from fractions import Fraction


METHODS = ["viewfence", "one_step", "no_defense", "body_only", "actual_branch"]


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


def safe(belief, secrets, alpha):
    totals = {}
    for world, _, mass in belief:
        secret = secrets[world]
        totals[secret] = totals.get(secret, Fraction(0)) + mass
    return max(totals.values(), default=Fraction(1)) <= alpha


def completed(case, belief):
    goals = case.get("goal_states", [])
    return bool(goals) and all(state in goals for _, state, mass in belief if mass)


def choose(case, belief, remaining, horizon, method):
    alpha = Fraction(str(case["alpha"]))
    body_only = method == "body_only"

    @functools.lru_cache(None)
    def search(current, available, depth):
        if completed(case, current):
            return 0.0, None, True
        if depth == 0 or not available:
            return 0.0, None, False
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
                not safe(post, case["secrets"], alpha) for _, _, post in checked
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
            reward = action.get("reward", 0.0) - case.get("cost_weight", 0.1) * action.get(
                "cost", 1.0
            )
            value = reward + sum(
                float(probability) * entry[0]
                for (_, probability, _), entry in zip(outcomes, future)
            )
            if value > best_value or best_action is None and all_viable:
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
            "strict_completion": False,
        }


def run_case(case, method):
    states = [str(s) for s in case.get("initial_states", ["0"] * len(case["prior"]))]
    belief = normalize(
        {
            (world, state): Fraction(str(mass))
            for world, (state, mass) in enumerate(zip(states, case["prior"]))
        }
    )
    observer = belief
    if not safe(belief, case["secrets"], Fraction(str(case["alpha"]))):
        raise ValueError("Initial belief violates the disclosure policy")
    remaining = tuple(range(len(case["actions"])))
    world, state = case["actual_world"], states[case["actual_world"]]
    rng = random.Random(str(case["id"]))
    events, reward = [], 0.0
    maximum = 0.0
    violation = False
    for step in range(case["horizon"]):
        totals = {}
        for w, _, mass in observer:
            totals[case["secrets"][w]] = totals.get(case["secrets"][w], 0) + mass
        maximum = max(maximum, float(max(totals.values())))
        if completed(case, observer):
            break
        _, index, _ = choose(
            case, belief, remaining, 1 if method == "one_step" else case["horizon"] - step, method
        )
        if index is None:
            break
        action = case["actions"][index]
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
        violation |= not safe(observer, case["secrets"], Fraction(str(case["alpha"])))
        events.append(event)
        reward += action.get("reward", 0)
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
        "reward": reward,
        "completed": state in case.get("goal_states", []),
        "joint_posterior": [[w, s, float(mass)] for w, s, mass in observer],
    }


def score(case, result):
    return {
        "maximum_posterior": result["max_secret_posterior"],
        "policy_violation": int(result["violation"]),
        "completion": int(result["completed"]),
        "safe_completion": int(result["completed"] and not result["violation"]),
        "tool_calls": len(result["events"]),
        "reward": result["reward"],
        "legal_effects": len(result["events"]) if not result["violation"] else 0,
    }
