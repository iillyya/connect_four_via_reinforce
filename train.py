import copy
import random
from argparse import ArgumentParser
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from environment import Environment
from opponents import heuristic_opponent_action, random_opponent_action

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH_DEFAULT = "connect4_policy.pt"
SUPPORTED_OPPONENT_KINDS = ("lagged", "random", "heuristic")


class PolicyNet(nn.Module):
    def __init__(self, rows=6, cols=7):
        super().__init__()
        self.rows = rows
        self.cols = cols

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * rows * cols, 128),
            nn.ReLU(),
            nn.Linear(128, cols),
        )

    def forward(self, x):
        # x: (batch, rows, cols)
        x = x.unsqueeze(1)  # → (batch, 1, rows, cols)
        x = self.conv(x)
        x = self.head(x)
        return x


class ValueNet(nn.Module):
    def __init__(self, rows=6, cols=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(rows * cols, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# -------------------------
# Helpers
# -------------------------
def encode_state(board: np.ndarray) -> torch.Tensor:
    return torch.tensor(board, dtype=torch.float32, device=device)


def masked_action_distribution(logits: torch.Tensor, valid_actions: List[int]) -> Categorical:
    mask = torch.full_like(logits, float("-inf"))
    if len(valid_actions) > 0:
        mask[valid_actions] = 0.0
    masked_logits = logits + mask
    return Categorical(logits=masked_logits)


def compute_returns(rewards: List[float], gamma=0.99) -> torch.Tensor:
    g_value = 0.0
    returns = []
    for reward in reversed(rewards):
        g_value = reward + gamma * g_value
        returns.insert(0, g_value)
    return torch.tensor(returns, dtype=torch.float32, device=device)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_start_player(start_mode: str, episode: int, rng: np.random.Generator) -> str:
    if start_mode == "agent":
        return "agent"
    if start_mode == "opponent":
        return "opponent"
    if start_mode == "alternate":
        return "agent" if episode % 2 == 1 else "opponent"
    if start_mode == "random":
        return "agent" if rng.integers(0, 2) == 0 else "opponent"
    raise ValueError("start_mode must be one of: agent, opponent, alternate, random")


def build_episode_checkpoint_path(base_path: str, episode: int) -> str:
    path = Path(base_path)
    suffix = path.suffix or ".pt"
    return str(path.with_name(f"{path.stem}_ep{episode}{suffix}"))


def build_best_checkpoint_path(base_path: str) -> str:
    path = Path(base_path)
    suffix = path.suffix or ".pt"
    return str(path.with_name(f"{path.stem}_best{suffix}"))


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value from '{value}'")


def parse_opponent_mix(mix_spec: str) -> Dict[str, float]:
    mix: Dict[str, float] = {}
    for item in mix_spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "Invalid opponent_mix format. Use 'lagged:0.6,random:0.2,heuristic:0.2'"
            )
        name, weight_str = item.split(":", 1)
        name = name.strip().lower()
        if name not in SUPPORTED_OPPONENT_KINDS:
            raise ValueError(f"Unsupported opponent '{name}'. Supported: {SUPPORTED_OPPONENT_KINDS}")
        weight = float(weight_str)
        if weight < 0:
            raise ValueError("Opponent mix weights must be non-negative")
        mix[name] = mix.get(name, 0.0) + weight

    if not mix:
        raise ValueError("opponent_mix must contain at least one opponent kind")

    total = sum(mix.values())
    if total <= 0:
        raise ValueError("opponent_mix weights must sum to a positive value")

    return {key: value / total for key, value in mix.items()}


def choose_opponent_kind(mix: Dict[str, float], rng: np.random.Generator) -> str:
    names = list(mix.keys())
    probs = np.array([mix[name] for name in names], dtype=np.float64)
    return str(rng.choice(names, p=probs))


def _policy_argmax_action(policy: PolicyNet, env: Environment) -> int:
    valid_actions = env.available_actions()
    perspective = 1.0 if env.current_player == env.player1 else -1.0
    state_proc = encode_state(env.get_observation()) * perspective
    with torch.no_grad():
        logits = policy(state_proc.unsqueeze(0)).squeeze(0)
        dist = masked_action_distribution(logits, valid_actions)
        return int(torch.argmax(dist.probs).item())


def _play_eval_game(policy: PolicyNet, opponent_kind: str, agent_first: bool, rng: np.random.Generator) -> float:
    if agent_first:
        env = Environment(player1="agent", player2="opponent", who_starts="agent", agent="agent")
    else:
        env = Environment(player1="opponent", player2="agent", who_starts="opponent", agent="agent")

    state = env.reset()
    done = False
    reward = 0.0

    while not done:
        if env.current_player == "agent":
            action = _policy_argmax_action(policy, env)
        else:
            if opponent_kind == "random":
                action = random_opponent_action(env, rng)
            elif opponent_kind == "heuristic":
                action = heuristic_opponent_action(env, player=env.current_player, rng=rng)
            else:
                raise ValueError(f"Unknown evaluation opponent kind: {opponent_kind}")

        state, reward, done, _ = env.step(action)

    del state
    return reward


def _score_from_counts(wins: int, draws: int, total: int) -> float:
    if total == 0:
        return 0.0
    return (wins + 0.5 * draws) / total


def evaluate_for_selection(policy: PolicyNet, num_games: int, seed: int) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], float]:
    if num_games % 2 != 0:
        raise ValueError("eval_games must be even to split agent-first and agent-second equally")

    rng = np.random.default_rng(seed)
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for opponent_kind in ("random", "heuristic"):
        counts = {
            "overall": {"wins": 0, "draws": 0, "losses": 0, "games": 0},
            "agent_first": {"wins": 0, "draws": 0, "losses": 0, "games": 0},
            "agent_second": {"wins": 0, "draws": 0, "losses": 0, "games": 0},
        }

        for game_idx in range(num_games):
            agent_first = game_idx < (num_games // 2)
            split = "agent_first" if agent_first else "agent_second"
            reward = _play_eval_game(policy, opponent_kind, agent_first=agent_first, rng=rng)

            counts["overall"]["games"] += 1
            counts[split]["games"] += 1

            if reward > 0:
                counts["overall"]["wins"] += 1
                counts[split]["wins"] += 1
            elif reward < 0:
                counts["overall"]["losses"] += 1
                counts[split]["losses"] += 1
            else:
                counts["overall"]["draws"] += 1
                counts[split]["draws"] += 1

        results[opponent_kind] = {}
        for split_name, split_counts in counts.items():
            wins = split_counts["wins"]
            draws = split_counts["draws"]
            losses = split_counts["losses"]
            games = split_counts["games"]
            results[opponent_kind][split_name] = {
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "games": games,
                "score": _score_from_counts(wins, draws, games),
            }

    selection_score = 0.5 * results["random"]["overall"]["score"] + 0.5 * results["heuristic"]["overall"]["score"]
    return results, selection_score


def choose_training_opponent_action(
    opponent_kind: str,
    env: Environment,
    opponent_policy: PolicyNet,
    state: np.ndarray,
    rng: np.random.Generator,
) -> int:
    if opponent_kind == "lagged":
        state_proc = encode_state(state) * -1.0
        with torch.no_grad():
            logits_opp = opponent_policy(state_proc.unsqueeze(0)).squeeze(0)
            valid_actions = env.available_actions()
            dist_opp = masked_action_distribution(logits_opp, valid_actions)
            return int(dist_opp.sample().item())

    if opponent_kind == "random":
        return random_opponent_action(env, rng)

    if opponent_kind == "heuristic":
        return heuristic_opponent_action(env, player=env.current_player, rng=rng)

    raise ValueError(f"Unknown training opponent kind: {opponent_kind}")


# -------------------------
# Save / Load
# -------------------------
def save_checkpoint(policy: PolicyNet, value: ValueNet, path=MODEL_PATH_DEFAULT):
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "value_state_dict": value.state_dict(),
            "rows": policy.rows,
            "cols": policy.cols,
        },
        path,
    )
    print(f"Checkpoint saved to {path}")


def load_policy_value(path=MODEL_PATH_DEFAULT):
    checkpoint = torch.load(path, map_location=device)
    policy = PolicyNet(rows=checkpoint["rows"], cols=checkpoint["cols"]).to(device)
    value = ValueNet(rows=checkpoint["rows"], cols=checkpoint["cols"]).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    value.load_state_dict(checkpoint["value_state_dict"])
    policy.eval()
    value.eval()
    print(f"Loaded checkpoint from {path}")
    return policy, value


# -------------------------
# Training
# -------------------------
def train(
    num_episodes: int = 5000,
    gamma: float = 0.99,
    lr: float = 3e-4,
    model_path: str = MODEL_PATH_DEFAULT,
    lag_k: int = 100,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    beta: float = 0.02,
    beta_decay: float = 0.999,
    beta_min: float = 0.001,
    seed: int = 42,
    checkpoint_every: int = 1000,
    start_mode: str = "alternate",
    update_every: int = 8,
    opponent_mix: str = "lagged:0.6,random:0.2,heuristic:0.2",
    eval_every: int = 1000,
    eval_games: int = 200,
    early_stop_patience: int = 5,
    selection_min_delta: float = 1e-4,
    log_window: int = 200,
    verbose: bool = True,
):
    if checkpoint_every is not None and checkpoint_every < 1:
        raise ValueError("checkpoint_every must be >= 1 or None")
    if update_every < 1:
        raise ValueError("update_every must be >= 1")
    if beta_decay <= 0:
        raise ValueError("beta_decay must be > 0")
    if eval_every is not None and eval_every < 1:
        raise ValueError("eval_every must be >= 1 or None")
    if eval_games < 2 or eval_games % 2 != 0:
        raise ValueError("eval_games must be even and >= 2")

    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    mix = parse_opponent_mix(opponent_mix)

    env = Environment(
        rows=6,
        cols=7,
        win_length=4,
        player1="agent",
        player2="opponent",
        who_starts="agent",
        agent="agent",
    )

    policy = PolicyNet().to(device)
    value_net = ValueNet().to(device)

    opponent_policy = PolicyNet().to(device)
    opponent_policy.load_state_dict(policy.state_dict())
    opponent_policy.eval()

    optimizer = optim.Adam(list(policy.parameters()) + list(value_net.parameters()), lr=lr)

    outcome_window = deque(maxlen=log_window)
    return_window = deque(maxlen=log_window)

    best_eval_score = float("-inf")
    best_policy_state = copy.deepcopy(policy.state_dict())
    best_value_state = copy.deepcopy(value_net.state_dict())
    best_episode = 0
    no_improve_evals = 0
    best_checkpoint_path = build_best_checkpoint_path(model_path)

    batch_log_probs: List[torch.Tensor] = []
    batch_values: List[torch.Tensor] = []
    batch_returns: List[torch.Tensor] = []
    batch_entropies: List[torch.Tensor] = []
    episodes_in_batch = 0
    update_step = 0

    if verbose:
        print(f"Using opponent mix: {mix}")

    for episode in range(1, num_episodes + 1):
        env.current_player = choose_start_player(start_mode, episode, rng)
        opponent_kind = choose_opponent_kind(mix, rng)
        state = env.reset()

        log_probs = []
        values = []
        rewards = []
        entropies = []
        terminal_reward = 0.0

        done = False

        while not done:
            if env.current_player == env.player1:
                state_proc = encode_state(state)

                logits = policy(state_proc.unsqueeze(0)).squeeze(0)
                valid_actions = env.available_actions()
                dist = masked_action_distribution(logits, valid_actions)

                action = dist.sample()
                log_prob = dist.log_prob(action)
                value_pred = value_net(state_proc.unsqueeze(0)).squeeze(0)
                entropy = dist.entropy()

                next_state, reward, done, _ = env.step(action.item())

                log_probs.append(log_prob)
                values.append(value_pred)
                rewards.append(reward)
                entropies.append(entropy)
                state = next_state

                if done:
                    terminal_reward = reward
                    break
            else:
                opp_action = choose_training_opponent_action(
                    opponent_kind=opponent_kind,
                    env=env,
                    opponent_policy=opponent_policy,
                    state=state,
                    rng=rng,
                )
                state, reward, done, _ = env.step(opp_action)

                if done:
                    if rewards:
                        rewards[-1] += reward
                    terminal_reward = reward
                    break

        if len(log_probs) == 0:
            continue

        returns = compute_returns(rewards, gamma)
        batch_log_probs.append(torch.stack(log_probs))
        batch_values.append(torch.stack(values).squeeze())
        batch_returns.append(returns)
        batch_entropies.append(torch.stack(entropies))
        episodes_in_batch += 1

        outcome_window.append(terminal_reward)
        return_window.append(float(returns[0].item()))

        should_update = (episodes_in_batch >= update_every) or (episode == num_episodes)
        if should_update:
            update_step += 1
            log_probs_tensor = torch.cat(batch_log_probs)
            values_tensor = torch.cat(batch_values)
            returns_tensor = torch.cat(batch_returns)
            entropies_tensor = torch.cat(batch_entropies)

            advantages = returns_tensor - values_tensor
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            policy_loss = -(log_probs_tensor * advantages.detach()).mean()
            value_loss = nn.functional.mse_loss(values_tensor, returns_tensor)
            entropy_bonus = entropies_tensor.mean()

            current_beta = max(beta_min, beta * (beta_decay ** (update_step - 1)))
            loss = policy_loss + value_coef * value_loss - current_beta * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(value_net.parameters()),
                    max_grad_norm,
                )
            optimizer.step()

            batch_log_probs.clear()
            batch_values.clear()
            batch_returns.clear()
            batch_entropies.clear()
            episodes_in_batch = 0

        if episode % lag_k == 0:
            opponent_policy.load_state_dict(policy.state_dict())
            opponent_policy.eval()
            if verbose:
                print(f"[Episode {episode}] Updated lagged opponent (lag {lag_k}).")

        if checkpoint_every and episode % checkpoint_every == 0:
            save_checkpoint(policy, value_net, path=build_episode_checkpoint_path(model_path, episode))

        if episode % 100 == 0 and verbose:
            win_rate = float(np.mean([1.0 if x > 0 else 0.0 for x in outcome_window])) if outcome_window else 0.0
            draw_rate = float(np.mean([1.0 if x == 0 else 0.0 for x in outcome_window])) if outcome_window else 0.0
            mean_return = float(np.mean(return_window)) if return_window else 0.0
            print(
                f"Episode {episode:6d} | Opp: {opponent_kind:9s} | "
                f"Win rate ({len(outcome_window)}): {win_rate:.3f} | Draw rate: {draw_rate:.3f} | "
                f"Mean return: {mean_return:.4f}"
            )

        if eval_every and episode % eval_every == 0:
            eval_results, selection_score = evaluate_for_selection(
                policy=policy,
                num_games=eval_games,
                seed=seed + 17 * episode,
            )

            rand_all = eval_results["random"]["overall"]["score"]
            heur_all = eval_results["heuristic"]["overall"]["score"]
            rand_first = eval_results["random"]["agent_first"]["score"]
            rand_second = eval_results["random"]["agent_second"]["score"]
            heur_first = eval_results["heuristic"]["agent_first"]["score"]
            heur_second = eval_results["heuristic"]["agent_second"]["score"]

            if verbose:
                print(
                    f"[Eval @ {episode}] sel={selection_score:.4f} | "
                    f"random all/f/s={rand_all:.3f}/{rand_first:.3f}/{rand_second:.3f} | "
                    f"heuristic all/f/s={heur_all:.3f}/{heur_first:.3f}/{heur_second:.3f}"
                )

            if selection_score > best_eval_score + selection_min_delta:
                best_eval_score = selection_score
                best_policy_state = copy.deepcopy(policy.state_dict())
                best_value_state = copy.deepcopy(value_net.state_dict())
                best_episode = episode
                no_improve_evals = 0
                save_checkpoint(policy, value_net, path=best_checkpoint_path)
            else:
                no_improve_evals += 1

            if early_stop_patience > 0 and no_improve_evals >= early_stop_patience:
                if verbose:
                    print(
                        f"Early stopping at episode {episode}: no evaluation improvement in "
                        f"{no_improve_evals} eval rounds. Best episode: {best_episode}, "
                        f"best score: {best_eval_score:.4f}."
                    )
                break

    if best_policy_state is not None:
        policy.load_state_dict(best_policy_state)
        value_net.load_state_dict(best_value_state)
        if verbose and best_episode > 0:
            print(
                f"Loaded best checkpoint from episode {best_episode} "
                f"with selection score {best_eval_score:.4f}"
            )

    save_checkpoint(policy, value_net, path=model_path)
    return policy, value_net


# -------------------------
# Simple play function to test trained policy vs opponent
# -------------------------
def play_against_policy(policy: PolicyNet, opponent: PolicyNet = None, render: bool = False):
    del render

    env = Environment(player1="agent", player2="opponent", agent="agent")
    state = env.reset()
    done = False

    while not done:
        state_proc = encode_state(state)
        with torch.no_grad():
            logits = policy(state_proc.unsqueeze(0)).squeeze(0)
            valid_actions = env.available_actions()
            mask = torch.full_like(logits, float("-inf"))
            mask[valid_actions] = 0.0
            probs = torch.softmax(logits + mask, dim=-1)
            action = torch.argmax(probs).item()

        state, reward, done, _ = env.step(action)
        env.print_board()
        print()

        if done:
            print("Final reward:", reward)
            break

        valid = env.available_actions()
        if opponent is None:
            opp_action = np.random.choice(valid)
        else:
            state_proc_opp = encode_state(state) * -1.0
            with torch.no_grad():
                logits_opp = opponent(state_proc_opp.unsqueeze(0)).squeeze(0)
                mask = torch.full_like(logits_opp, float("-inf"))
                mask[valid] = 0.0
                probs_opp = torch.softmax(logits_opp + mask, dim=-1)
                opp_action = torch.argmax(probs_opp).item()

        state, reward, done, _ = env.step(opp_action)


def main():
    parser = ArgumentParser()
    parser.add_argument("--MODEL_PATH", type=str, default=MODEL_PATH_DEFAULT, help="Path to save/load the model")
    parser.add_argument("--episodes", type=int, default=20000, help="Number of training episodes")
    parser.add_argument("--lag_k", type=int, default=100, help="Episodes between lagged-opponent refreshes")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--value_coef", type=float, default=0.5, help="Weight for value loss")
    parser.add_argument("--beta", type=float, default=0.02, help="Initial entropy regularization coefficient")
    parser.add_argument("--beta_decay", type=float, default=0.999, help="Multiplicative decay applied each update step")
    parser.add_argument("--beta_min", type=float, default=0.001, help="Lower bound for beta after decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_every", type=int, default=1000, help="Save intermediate checkpoint every N episodes")
    parser.add_argument(
        "--start_mode",
        type=str,
        default="alternate",
        choices=["agent", "opponent", "alternate", "random"],
        help="Who starts each episode",
    )
    parser.add_argument("--update_every", type=int, default=8, help="Number of episodes per optimizer update")
    parser.add_argument(
        "--opponent_mix",
        type=str,
        default="lagged:0.6,random:0.2,heuristic:0.2",
        help="Training opponent distribution",
    )
    parser.add_argument("--eval_every", type=int, default=1000, help="Run model-selection evaluation every N episodes")
    parser.add_argument("--eval_games", type=int, default=200, help="Evaluation games per opponent (must be even)")
    parser.add_argument("--early_stop_patience", type=int, default=5, help="Stop after this many eval rounds without improvement")
    parser.add_argument("--selection_min_delta", type=float, default=1e-4, help="Minimum score improvement to reset patience")
    parser.add_argument("--verbose", type=str2bool, default=True, help="Whether to print training progress")
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        model_path=args.MODEL_PATH,
        lag_k=args.lag_k,
        value_coef=args.value_coef,
        beta=args.beta,
        beta_decay=args.beta_decay,
        beta_min=args.beta_min,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        start_mode=args.start_mode,
        update_every=args.update_every,
        opponent_mix=args.opponent_mix,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        early_stop_patience=args.early_stop_patience,
        selection_min_delta=args.selection_min_delta,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
