# Connect Four via REINFORCE

This project implements a **Connect Four** agent trained using the **REINFORCE (Monte Carlo Policy Gradient)** algorithm in PyTorch.

The environment is implemented from scratch using **NumPy** (no Gym/Gymnasium), and visualization is handled optionally via **Pygame**.

---

## Project Goals

- Implement Connect Four environment without external RL libraries
- Train an agent using vanilla REINFORCE
- Explore instability and variance issues in Monte Carlo policy gradients
- Experiment with improvements (entropy regularization, stronger opponents, self-play, actor-critic)

---


## Installation

- Install uv (https://docs.astral.sh/uv/getting-started/installation/#pypi)

- Clone the repository

```bash
git clone https://github.com/iillyya/connect_four_via_reinforce.git
```

- Sync dependencies

```bash
uv sync
```

- Download the model

``` bash
uv run download_model.py "https://drive.google.com/uc?id=18ckHsBTUuwq5ny8UpS8DtsUgxkSB3IlX"
```

- Start the game

```bash
uv run renderer.py
```
