import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from argparse import ArgumentParser
from typing import List

from environment import Environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH_DEFAULT = "connect4_policy.pt"


# -------------------------
# Networks
# -------------------------
class PolicyNet(nn.Module):
    def __init__(self, rows=6, cols=7):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(rows * cols, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, cols)
        )

    def forward(self, x):
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self, rows=6, cols=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(rows * cols, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # returns (batch,) or scalar


# -------------------------
# Helpers
# -------------------------
def encode_state(board: np.ndarray) -> torch.Tensor:
    """
    Convert board (rows x cols, values -1/0/1) to float tensor on device.
    """
    return torch.tensor(board, dtype=torch.float32, device=device)


def masked_action_distribution(logits: torch.Tensor, valid_actions: List[int]) -> Categorical:
    """
    Mask invalid actions by setting logits to -inf for them.
    logits: 1D tensor (cols,)
    valid_actions: list of column indices that are allowed
    """
    # Create mask of -inf, then set valid indices to 0 so logits + mask keeps only valid logits
    mask = torch.full_like(logits, float("-inf"))
    if len(valid_actions) > 0:
        mask[valid_actions] = 0.0
    masked_logits = logits + mask
    return Categorical(logits=masked_logits)


def compute_returns(rewards: List[float], gamma=0.99) -> torch.Tensor:
    """
    Compute Monte-Carlo returns G_t for a sequence of rewards (list).
    Returns tensor on device with same length as rewards.
    """
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32, device=device)


# -------------------------
# Save / Load
# -------------------------
def save_checkpoint(policy: PolicyNet, value: ValueNet, path=MODEL_PATH_DEFAULT):
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "value_state_dict": value.state_dict(),
        "rows": policy.rows,
        "cols": policy.cols,
    }, path)
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
# Training with lagged opponent + value baseline
# -------------------------
def train(
    num_episodes: int = 5000,
    gamma: float = 0.99,
    lr: float = 1e-3,
    model_path: str = MODEL_PATH_DEFAULT,
    lag_k: int = 100,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    beta: float = 0.01,  # entropy regularization coefficient
    verbose: bool = True,
):
    """
    Train with:
      - trainable policy (plays as player1)
      - lagged opponent policy (frozen copy of trainable, updated every lag_k episodes)
      - value network as baseline (trained jointly)
    """

    env = Environment(
        rows=6,
        cols=7,
        win_length=4,
        player1="agent",
        player2="opponent",
        who_starts="agent",
        agent="agent",
    )

    # trainable networks
    policy = PolicyNet().to(device)
    value_net = ValueNet().to(device)

    # opponent is a frozen copy of policy (starts identical)
    opponent_policy = PolicyNet().to(device)
    opponent_policy.load_state_dict(policy.state_dict())
    opponent_policy.eval()

    optimizer = optim.Adam(list(policy.parameters()) + list(value_net.parameters()), lr=lr)

    for episode in range(1, num_episodes + 1):
        state = env.reset()

        # store values for agent moves only (player1 = "agent")
        log_probs = []
        values = []
        rewards = []  # rewards from perspective of agent, appended after agent move (and possibly after opponent ends)

        done = False

        while not done:
            # whose turn?
            if env.current_player == env.player1:
                # Agent (trainable) acts
                # Represent state from current player's perspective.
                # current player is agent (player1), marker = 1 -> board stays as-is.
                state_proc = encode_state(state) * 1.0  # explicit

                logits = policy(state_proc.unsqueeze(0)).squeeze(0)
                valid_actions = env.available_actions()
                dist = masked_action_distribution(logits, valid_actions)

                action = dist.sample()
                log_prob = dist.log_prob(action)
                value_pred = value_net(state_proc.unsqueeze(0)).squeeze(0)

                # step environment
                next_state, reward, done, _ = env.step(action.item())

                # store (for agent update)
                log_probs.append(log_prob)
                values.append(value_pred)
                rewards.append(reward)  # reward might be 0 unless terminal

                state = next_state

                if done:
                    break

            else:
                # Opponent (frozen) acts
                # represent from opponent perspective: multiply board by -1
                state_proc = encode_state(state) * -1.0

                with torch.no_grad():
                    logits_opp = opponent_policy(state_proc.unsqueeze(0)).squeeze(0)
                    valid_actions = env.available_actions()
                    dist_opp = masked_action_distribution(logits_opp, valid_actions)
                    opp_action = dist_opp.sample().item()

                state, reward, done, _ = env.step(opp_action)

                # If opponent ended the game here, the environment returns reward from agent perspective
                # We need to append that reward to the last agent step sequence so lengths match.
                if done:
                    # This reward corresponds to the terminal outcome and should be appended
                    # to the rewards list (so returns account for it).
                    rewards[-1] += reward
                    break

        # If there were no agent moves (very unlikely), skip update
        if len(log_probs) == 0:
            continue

        # compute returns (one per agent move)
        returns = compute_returns(rewards, gamma)

        # convert lists to tensors
        log_probs_tensor = torch.stack(log_probs)                  # shape (T_agent,)
        values_tensor = torch.stack(values).squeeze()             # shape (T_agent,)
        returns_tensor = returns                                  # shape (T_agent,)

        # advantage
        advantages = returns_tensor - values_tensor

        # optionally normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # policy loss (actor): - sum log_prob * advantage
        policy_loss = -(log_probs_tensor * advantages.detach()).sum()

        # value loss (critic): MSE between returns and value predictions
        value_loss = nn.functional.mse_loss(values_tensor, returns_tensor)

        entropy_loss = -beta * dist.entropy().sum() # entropy regularization

        loss = policy_loss + value_coef * value_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        # optional grad clipping
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value_net.parameters()), max_grad_norm)
        optimizer.step()

        # every lag_k episodes update opponent to current policy (make a new frozen copy)
        if episode % lag_k == 0:
            opponent_policy.load_state_dict(policy.state_dict())
            opponent_policy.eval()
            if verbose:
                print(f"[Episode {episode}] Updated opponent policy (lag {lag_k}).")

        # logging
        if episode % 100 == 0:
            total_reward = returns_tensor.sum().item()  # sum of returns (not perfect metric but ok)
            if verbose:
                print(f"Episode {episode:6d} | Loss: {loss.item():.4f} | Policy loss: {policy_loss.item():.4f} | "
                      f"Value loss: {value_loss.item():.4f} | Sum returns: {total_reward:.4f}")

    # save final models
    save_checkpoint(policy, value_net, path=model_path)
    return policy, value_net


# -------------------------
# Simple play function to test trained policy vs opponent (random or snapshot)
# -------------------------
def play_against_policy(policy: PolicyNet, opponent: PolicyNet = None, render: bool = False):
    env = Environment(player1="agent", player2="opponent", agent="agent")
    state = env.reset()
    done = False

    while not done:
        # agent move (policy)
        state_proc = encode_state(state) * 1.0
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

        # opponent move
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

    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--MODEL_PATH", type=str, default=MODEL_PATH_DEFAULT, help="Path to save/load the model")
    parser.add_argument("--episodes", type=int, default=20000, help="Number of training episodes")
    parser.add_argument("--lag_k", type=int, default=100, help="Number of episodes between opponent updates")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--value_coef", type=float, default=0.5, help="Weight for value loss")
    parser.add_argument("--beta", type=float, default=0.01, help="Entropy regularization coefficient")
    parser.add_argument("--verbose", type=bool, default=True, help="Whether to print training progress")
    args = parser.parse_args()

    policy_trained, value_trained = train(
        num_episodes=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        model_path=args.MODEL_PATH,
        lag_k=args.lag_k,
        value_coef=args.value_coef,
    )

    # load checkpoint and play one game (opponent is final lagged copy)
    loaded_policy, loaded_value = load_policy_value(path=args.MODEL_PATH)
    # opponent is the same as the loaded policy for a quick check:
    play_against_policy(loaded_policy, opponent=loaded_policy)