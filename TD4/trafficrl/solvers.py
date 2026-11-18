"""Exact dynamic-programming planners for the supplied finite TrafficMDP.

No Flask dependency. Iteration callbacks receive actual solver measurements.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from .mdp import TrafficMDP, State


@dataclass
class SolveResult:
    values: dict[State, float]
    policy: dict[State, str]
    history: list[dict]
    converged: bool
    iterations: int
    elapsed_seconds: float


def _validate(gamma: float, epsilon: float, max_iterations: int) -> None:
    if not 0 <= gamma < 1:
        raise ValueError("gamma must be in [0, 1)")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")


def _prepare(mdp: TrafficMDP):
    """Cache transitions once; no repeated Poisson calculations per iteration."""
    return {
        s: {a: mdp.transitions(s, a) for a in mdp.valid_actions(s)} for s in mdp.states
    }


def _q(outcomes, values, gamma):
    # TODO 1: Q(s,a) = sum_{s'} P(s'|s,a) * (R(s,a,s') + gamma * V(s'))
    return sum(p * (r + gamma * values[s2]) for p, s2, r in outcomes)


def action_values(
    mdp: TrafficMDP, state: State, values: dict[State, float], gamma: float
):
    """Q(s,a) for each valid action under a supplied value function."""
    if not 0 <= gamma < 1:
        raise ValueError("gamma must be in [0, 1)")
    return {
        a: _q(mdp.transitions(state, a), values, gamma)
        for a in mdp.valid_actions(state)
    }


def _greedy(actions, values, gamma):
    # Deterministic tie breaking: KEEP is listed first in the MDP.
    # TODO 2: argmax_a Q(s,a). Strict ">" keeps the first action on ties (KEEP).
    best_action, best_q = None, float("-inf")
    for action, outcomes in actions.items():
        q = _q(outcomes, values, gamma)
        if q > best_q:
            best_action, best_q = action, q
    return best_action


def value_iteration(
    mdp: TrafficMDP,
    gamma: float = 0.95,
    epsilon: float = 1e-3,
    max_iterations: int = 1000,
    on_iteration: Callable[[dict], None] | None = None,
) -> SolveResult:
    """Synchronous Bellman optimality updates; epsilon bounds the Bellman residual."""
    _validate(gamma, epsilon, max_iterations)
    start = perf_counter()
    transitions = _prepare(mdp)
    values = dict.fromkeys(mdp.states, 0.0)
    policy = {s: next(iter(transitions[s])) for s in mdp.states}
    history = []
    converged = False
    for iteration in range(1, max_iterations + 1):
        # TODO 3a: synchronous Bellman optimality update (uses previous `values`)
        updated = {
            s: max(_q(outcomes, values, gamma) for outcomes in transitions[s].values())
            for s in mdp.states
        }

        # TODO 3b: Delta_k = max_s |V_{k+1}(s) - V_k(s)|
        delta = max(abs(updated[s] - values[s]) for s in mdp.states)

        # TODO 3c: greedy policy w.r.t. the updated values
        new_policy = {s: _greedy(transitions[s], updated, gamma) for s in mdp.states}
        changes = sum(new_policy[s] != policy[s] for s in mdp.states)
        values, policy = updated, new_policy
        record = {
            "iteration": iteration,
            "delta": delta,
            "policy_changes": changes,
            "elapsed_seconds": perf_counter() - start,
        }
        history.append(record)
        if on_iteration:
            on_iteration(record.copy())
        if delta < epsilon:
            converged = True
            break
    return SolveResult(
        values, policy, history, converged, len(history), perf_counter() - start
    )


def policy_iteration(
    mdp: TrafficMDP,
    gamma: float = 0.95,
    epsilon: float = 1e-3,
    max_iterations: int = 1000,
    evaluation_epsilon: float = 1e-5,
    max_evaluation_iterations: int = 10000,
    on_iteration: Callable[[dict], None] | None = None,
) -> SolveResult:
    """Iterative policy evaluation followed by greedy policy improvement.

    evaluation_epsilon controls evaluation precision; max_iterations limits
    outer policy-improvement passes. An unconverged evaluation never counts
    as a converged solution.
    """
    _validate(gamma, epsilon, max_iterations)
    if evaluation_epsilon <= 0 or max_evaluation_iterations < 1:
        raise ValueError(
            "evaluation_epsilon and max_evaluation_iterations must be positive"
        )
    start = perf_counter()
    transitions = _prepare(mdp)
    values = dict.fromkeys(mdp.states, 0.0)
    policy = {s: next(iter(actions)) for s, actions in transitions.items()}
    history = []
    converged = False
    for iteration in range(1, max_iterations + 1):
        evaluation_converged = False
        for evaluation_steps in range(1, max_evaluation_iterations + 1):
            # TODO 4a: synchronous Bellman expectation update for pi_k
            updated = {
                s: _q(transitions[s][policy[s]], values, gamma) for s in mdp.states
            }
            delta = max(abs(updated[s] - values[s]) for s in mdp.states)
            values = updated
            if delta < evaluation_epsilon:
                evaluation_converged = True
                break
        # TODO 4b: greedy policy improvement w.r.t. V^{pi_k}
        improved = {s: _greedy(transitions[s], values, gamma) for s in mdp.states}
        changes = sum(improved[s] != policy[s] for s in mdp.states)
        policy = improved
        record = {
            "iteration": iteration,
            "delta": delta,
            "policy_changes": changes,
            "evaluation_iterations": evaluation_steps,
            "evaluation_converged": evaluation_converged,
            "elapsed_seconds": perf_counter() - start,
        }
        history.append(record)
        if on_iteration:
            on_iteration(record.copy())
        if changes == 0 and evaluation_converged:
            converged = True
            break
        if not evaluation_converged:
            break
    return SolveResult(
        values, policy, history, converged, len(history), perf_counter() - start
    )