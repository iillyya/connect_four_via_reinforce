import json
import math
from argparse import ArgumentParser
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import torch

from environment import Environment
from opponents import heuristic_opponent_action, random_opponent_action
from train import encode_state, masked_action_distribution, set_global_seed, load_policy


SUPPORTED_OPPONENTS = {"random", "heuristic", "self"}
# PROTOCOL_VERSION = "1.1"


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value from '{value}'")


def parse_opponents(value: str) -> List[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("At least one opponent must be provided")
    unknown = [name for name in names if name not in SUPPORTED_OPPONENTS]
    if unknown:
        raise ValueError(f"Unsupported opponents: {unknown}; supported: {sorted(SUPPORTED_OPPONENTS)}")
    return names


def compute_score(wins: int, draws: int, num_games: int) -> float:
    if num_games == 0:
        return 0.0
    return (wins + 0.5 * draws) / num_games


def wilson_interval(successes: float, num_games: int, z: float = 1.96) -> List[float]:
    if num_games == 0:
        return [0.0, 0.0]
    p_hat = successes / num_games
    z2 = z * z
    denominator = 1.0 + z2 / num_games
    center = (p_hat + z2 / (2.0 * num_games)) / denominator
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * num_games)) / num_games) / denominator
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return [low, high]


def build_start_order(num_games: int, swap_sides: bool, rng: np.random.Generator) -> List[bool]:
    if not swap_sides:
        return [True] * num_games
    if num_games % 2 != 0:
        raise ValueError("num_games must be even when swap_sides=true")
    start_order = [True] * (num_games // 2) + [False] * (num_games // 2)
    rng.shuffle(start_order)
    return start_order


def _policy_action(policy: torch.nn.Module, board: np.ndarray, valid_actions: List[int], greedy: bool = True) -> int:
    state_proc = encode_state(board)
    with torch.no_grad():
        logits = policy(state_proc.unsqueeze(0)).squeeze(0)
        dist = masked_action_distribution(logits, valid_actions)
        if greedy:
            action = int(torch.argmax(dist.probs).item())
        else:
            action = int(dist.sample().item())
    return action


def _opponent_action(
    env: Environment,
    board: np.ndarray,
    opponent_kind: str,
    policy: torch.nn.Module,
    rng: np.random.Generator,
) -> int:
    if opponent_kind == "random":
        return random_opponent_action(env, rng)
    if opponent_kind == "heuristic":
        return heuristic_opponent_action(env, player="opponent", rng=rng)
    if opponent_kind == "self":
        return _policy_action(policy, board, env.available_actions())
    raise ValueError(f"Unknown opponent kind: {opponent_kind}")


def play_single_game(
    policy: torch.nn.Module,
    opponent_kind: str,
    agent_first: bool,
    rng: np.random.Generator,
    evaluation_mode: bool = True,
) -> Dict[str, float]:
    if agent_first:
        env = Environment(player1="agent", player2="opponent", who_starts="agent", agent="agent")
    else:
        env = Environment(player1="opponent", player2="agent", who_starts="opponent", agent="agent")

    state = env.reset()
    done = False
    reward = 0.0
    total_moves = 0
    illegal_moves = 0
    agent_moves = 0

    while not done:
        valid_actions = env.available_actions()
        if env.current_player == "agent":
            action = _policy_action(policy, state, valid_actions, greedy=evaluation_mode)
            agent_moves += 1
            if action not in valid_actions:
                illegal_moves += 1
                action = int(rng.choice(valid_actions))
        else:
            action = _opponent_action(env, state, opponent_kind, policy, rng)
            if action not in valid_actions:
                action = int(rng.choice(valid_actions))

        state, reward, done, _ = env.step(action)
        total_moves += 1

    return {
        "reward": reward,
        "total_moves": total_moves,
        "illegal_moves": illegal_moves,
        "agent_moves": agent_moves,
    }


def _new_aggregate() -> Dict[str, float]:
    return {
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "total_moves": 0,
        "illegal_moves": 0,
        "agent_moves": 0,
        "games": 0,
    }


def _update_aggregate(agg: Dict[str, float], game: Dict[str, float]):
    reward = game["reward"]
    agg["games"] += 1
    agg["total_moves"] += int(game["total_moves"])
    agg["illegal_moves"] += int(game["illegal_moves"])
    agg["agent_moves"] += int(game["agent_moves"])

    if reward > 0:
        agg["wins"] += 1
    elif reward < 0:
        agg["losses"] += 1
    else:
        agg["draws"] += 1


def _finalize_aggregate(agg: Dict[str, float]) -> Dict[str, float]:
    games = int(agg["games"])
    wins = int(agg["wins"])
    draws = int(agg["draws"])
    losses = int(agg["losses"])

    score = compute_score(wins, draws, games)
    ci_low, ci_high = wilson_interval(wins + 0.5 * draws, games)

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "score_ci95_low": ci_low,
        "score_ci95_high": ci_high,
        "win_rate": wins / games if games else 0.0,
        "draw_rate": draws / games if games else 0.0,
        "loss_rate": losses / games if games else 0.0,
        "avg_game_len": agg["total_moves"] / games if games else 0.0,
        "illegal_move_rate": agg["illegal_moves"] / agg["agent_moves"] if agg["agent_moves"] else 0.0,
        "num_games": games,
    }


def evaluate_policy(
    policy: torch.nn.Module,
    num_games: int,
    opponent_kind: str,
    swap_sides: bool = True,
    seed: int = 123,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    start_order = build_start_order(num_games, swap_sides, rng)

    overall = _new_aggregate()
    first = _new_aggregate()
    second = _new_aggregate()

    for agent_first in start_order:
        game = play_single_game(policy, opponent_kind, agent_first, rng, evaluation_mode=False)
        _update_aggregate(overall, game)
        if agent_first:
            _update_aggregate(first, game)
        else:
            _update_aggregate(second, game)

    overall_metrics = _finalize_aggregate(overall)
    first_metrics = _finalize_aggregate(first)
    second_metrics = _finalize_aggregate(second)

    output = dict(overall_metrics)
    output["agent_first"] = first_metrics
    output["agent_second"] = second_metrics
    return output


def run_evaluation(
    model_path: str,
    num_games: int,
    opponents: List[str],
    swap_sides: bool,
    seed: int,
) -> Dict[str, object]:
    set_global_seed(seed)
    policy = load_policy(path=model_path)
    results = {}
    for idx, opponent_kind in enumerate(opponents):
        results[opponent_kind] = evaluate_policy(
            policy=policy,
            num_games=num_games,
            opponent_kind=opponent_kind,
            swap_sides=swap_sides,
            seed=seed + idx,
        )

    return {
        "model_path": model_path,
        "num_games": num_games,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # "protocol_version": PROTOCOL_VERSION,
        "results": results,
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="connect4_policy.pt", help="Path to saved model checkpoint")
    parser.add_argument("--num_games", type=int, default=400, help="Number of games per opponent")
    parser.add_argument(
        "--opponents",
        type=str,
        default="random,heuristic,self",
        help="Comma-separated opponents from: random,heuristic,self",
    )
    parser.add_argument(
        "--swap_sides",
        type=str2bool,
        default=True,
        help="If true, agent plays 50% games first and 50% second",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--output_json", type=str, default="eval_report.json", help="Where to write evaluation report")
    args = parser.parse_args()

    opponents = parse_opponents(args.opponents)
    report = run_evaluation(
        model_path=args.model_path,
        num_games=args.num_games,
        opponents=opponents,
        swap_sides=args.swap_sides,
        seed=args.seed,
    )

    with open(args.output_json, "w", encoding="utf-8") as fobj:
        json.dump(report, fobj, indent=2)

    print(f"Evaluation report saved to {args.output_json}")
    for name, metrics in report["results"].items():
        first = metrics["agent_first"]
        second = metrics["agent_second"]
        print(
            f"[{name}] score={metrics['score']:.3f} "
            f"CI95=({metrics['score_ci95_low']:.3f},{metrics['score_ci95_high']:.3f}) "
            f"W/D/L={metrics['wins']}/{metrics['draws']}/{metrics['losses']} "
            f"f/s={first['score']:.3f}/{second['score']:.3f} "
            f"illegal_move_rate={metrics['illegal_move_rate']:.6f}"
        )


if __name__ == "__main__":
    main()
