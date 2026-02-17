import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from argparse import ArgumentParser

from environment import Environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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


def encode_state(board: np.ndarray) -> torch.Tensor:
    return torch.tensor(board, dtype=torch.float32, device=device)


def masked_action_distribution(logits, valid_actions):
    mask = torch.full_like(logits, float("-inf"))
    mask[valid_actions] = 0.0
    return Categorical(logits=logits + mask)


def compute_returns(rewards, gamma=0.99):
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32, device=device)



def save_policy(policy, path=MODEL_PATH):
    torch.save({
        "model_state_dict": policy.state_dict(),
        "rows": policy.rows,
        "cols": policy.cols,
    }, path)
    print(f"Model saved to {path}")


def load_policy(path=MODEL_PATH):
    checkpoint = torch.load(path, map_location=device)

    policy = PolicyNet(
        rows=checkpoint["rows"],
        cols=checkpoint["cols"]
    ).to(device)

    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    print(f"Model loaded from {path}")
    return policy

def train(num_episodes=5000, gamma=0.99, lr=1e-3, MODEL_PATH=MODEL_PATH):

    env = Environment(
        rows=6,
        cols=7,
        win_length=4,
        player1="agent",
        player2="random",
        who_starts="agent",
        agent="agent",
    )

    policy = PolicyNet().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for episode in range(num_episodes):

        state = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:

            # Agent move
            state_tensor = encode_state(state).unsqueeze(0)
            logits = policy(state_tensor).squeeze(0)

            valid_actions = env.available_actions()
            dist = masked_action_distribution(logits, valid_actions)

            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done, _ = env.step(action.item())

            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state

            if done:
                break

            # Random opponent
            opponent_action = np.random.choice(env.available_actions())
            state, reward, done, _ = env.step(opponent_action)

            if done:
                rewards.append(reward)

        # REINFORCE update
        returns = compute_returns(rewards, gamma)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = torch.stack([-lp * G for lp, G in zip(log_probs, returns)]).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 500 == 0:
            print(f"Episode {episode+1}, total reward: {sum(rewards):.2f}")

    save_policy(policy)
    return policy


def play_against_random(policy):

    env = Environment(player1="agent", player2="random", agent="agent")
    state = env.reset()
    done = False

    while not done:

        state_tensor = encode_state(state).unsqueeze(0)
        with torch.no_grad():
            logits = policy(state_tensor).squeeze(0)

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

        opponent_action = np.random.choice(env.available_actions())
        state, reward, done, _ = env.step(opponent_action)



if __name__ == "__main__":


    parser = ArgumentParser()
    parser.add_argument("--MODEL_PATH", type=str, default="connect4_policy.pt", help="Path to save/load the model")
    args = parser.parse_args()

    # Train and save
    train(MODEL_PATH=args.MODEL_PATH)

    # Load for inference
    policy = load_policy(path=args.MODEL_PATH)

    # Play one game
    play_against_random(policy)