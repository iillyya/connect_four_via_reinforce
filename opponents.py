from typing import List, Optional

import numpy as np

from environment import Environment


def _would_player_win_after_action(env: Environment, action: int, player: str) -> bool:
    if action not in env.available_actions():
        return False

    marker = 1 if player == env.player1 else -1
    row = env.filled_positions[action]
    board = env.board.copy()
    board[row, action] = marker

    sim_env = Environment(
        rows=env.rows,
        cols=env.cols,
        win_length=env.win_length,
        player1=env.player1,
        player2=env.player2,
        who_starts=env.current_player,
        agent=env.agent,
    )
    sim_env.board = board
    sim_env.filled_positions = env.filled_positions.copy()
    sim_env.filled_positions[action] += 1
    sim_env._winner_cached = None
    return sim_env.winner_name() == player


def random_opponent_action(env: Environment, rng: np.random.Generator) -> int:
    valid_actions = env.available_actions()
    return int(rng.choice(valid_actions))


def heuristic_opponent_action(
    env: Environment,
    player: str,
    rng: Optional[np.random.Generator] = None,
) -> int:
    del rng  # deterministic choice by default

    valid_actions: List[int] = env.available_actions()
    if not valid_actions:
        raise ValueError("No valid actions available")

    for action in valid_actions:
        if _would_player_win_after_action(env, action, player):
            return action

    opponent = env.player1 if player == env.player2 else env.player2
    for action in valid_actions:
        if _would_player_win_after_action(env, action, opponent):
            return action

    center = env.cols // 2
    ranked = sorted(valid_actions, key=lambda col: (abs(col - center), col))
    return ranked[0]
