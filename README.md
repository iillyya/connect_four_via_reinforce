# Connect Four via REINFORCE

Compact RL project: a Connect-4 agent trained with REINFORCE

## 1. Problem formulation (MDP)

We model Connect-4 as an MDP:

```math
(\mathcal{S}, \mathcal{A}, P, R, \gamma)
```

- State:
  ```math
  s_t = (B_t, p_t)
  ```
  where
  ```math
  B_t \in \{-1,0,+1\}^{6 \times 7}
  ```
  and `p_t` is the current player.
- Action:
  ```math
  a_t \in \mathcal{A}(s_t)
  ```
  (selected column).
- Valid action set:
  ```math
  \mathcal{A}(s_t)=\{c \in \{0,\dots,6\}\ |\ \text{column }c\text{ is not full}\}.
  ```
- Transition (deterministic game physics):
  ```math
  s_{t+1}=T(s_t,a_t)
  ```
- Reward (agent perspective):
  ```math
  r_t=
  \begin{cases}
  +1,& \text{agent wins}\\
  -1,& \text{agent loses}\\
  0,& \text{otherwise (including draw)}
  \end{cases}
  ```
- Objective:
  ```math
  J(\theta)=\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T}\gamma^t r_t\right].
  ```

## 2. Method (REINFORCE + CNN)

- Policy network (`PolicyNet`): `Conv2d(1,32,3) -> Conv2d(32,64,3) -> Linear(64*6*7,128) -> Linear(128,7)`.
- Invalid moves are masked before sampling/action selection (`masked_action_distribution`).
- Returns:
  ```math
  G_t=\sum_{k=t}^{T}\gamma^{k-t}r_k
  ```
- Loss with entropy regularization:
  ```math
  \mathcal{L}(\theta)= -\mathbb{E}\left[\sum_t \log \pi_\theta(a_t|s_t)\,\hat{G}_t\right]
  -\beta\,\mathbb{E}\left[\sum_t \mathcal{H}\left(\pi_\theta(\cdot|s_t)\right)\right]
  ```
  where
  ```math
  \hat{G}_t
  ```
  are normalized returns.
- Training opponent is sampled from a mixture of `lagged`, `random`, and `heuristic` opponents.

Algorithm essentials:

```text
Initialize policy πθ and lagged opponent π¯
for each episode:
  reset env, sample starter and opponent type
  optionally do random opening moves
  roll out episode:
    if agent turn: sample masked action from πθ, store log_prob and entropy
    else: play opponent move (lagged/random/heuristic)
  compute discounted returns Gt
  every K episodes:
    update θ with REINFORCE + entropy bonus
  every lag_k episodes: copy πθ -> π¯
  periodically evaluate; keep best checkpoint; early-stop on no improvement
```

## 3. Results and interpretation

From the project presentation:

- Training only against a random opponent led to degenerate exploitative strategies.
- Pure self-play showed a symmetry/mirror pathology: very strong as first player, weak as second player in deterministic evaluation.
- Adding lagged self-play, entropy bonus, opponent mixture, and random openings (0-4 opening moves) improved robustness.
- CNN policy outperformed the earlier MLP by better capturing local board patterns.

Quantitative demo (pretrained model, 400 games/opponent, `eval_report.json`, run on February 18, 2026):

| Opponent | Score | W/D/L | Agent first | Agent second |
|---|---:|---:|---:|---:|
| random | 0.920 | 368/0/32 | 0.955 | 0.885 |
| heuristic | 0.839 | 333/5/62 | 0.708 | 0.970 |
| self | 0.473 | 189/0/211 | 0.905 | 0.040 |

Interpretation: the side-dependent asymmetry is still visible (especially in self-play), which is consistent with the slide discussion about brittle deterministic policies and evaluation challenges.

`evaluate.py` reports:

- score:
  ```math
  (\text{wins} + 0.5 \cdot \text{draws})/\text{games}
  ```
- 95% Wilson CI for score,
- split metrics for agent-first vs agent-second,
- illegal move rate.

## 4. Installation and quick start

Requirements:

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

Setup:

```bash
git clone https://github.com/iillyya/connect_four_via_reinforce.git
cd connect_four_via_reinforce
uv sync
```

Download pretrained model:

```bash
uv run python download_model.py "https://drive.google.com/uc?id=18ckHsBTUuwq5ny8UpS8DtsUgxkSB3IlX" --output connect_4_bot.pt
```

Play against model (Pygame):

```bash
uv run python renderer.py --model_path connect_4_bot.pt --who_starts alternate
```

Evaluate model:

```bash
uv run python evaluate.py --model_path connect_4_bot.pt --num_games 400 --opponents random,heuristic,self --swap_sides true --output_json eval_report.json
```

## 5. Essential code highlights

- `environment.py`
  - `Environment`: game state, move logic, terminal checks, reward semantics.
  - Key methods: `step`, `available_actions`, `get_winner`, `reset`.
- `train.py`
  - `PolicyNet`: CNN policy.
  - Key methods: `train`, `compute_returns`, `masked_action_distribution`, `evaluate_for_selection`.
  - Includes checkpointing, entropy regularization, lagged-opponent updates, early stopping.
- `opponents.py`
  - `random_opponent_action`, `heuristic_opponent_action` (one-step win/block + center preference).
- `evaluate.py`
  - `evaluate_policy`, `run_evaluation`: benchmarking with confidence intervals and side splits.
- `renderer.py`
  - `PygameRenderer`: human-vs-model local gameplay UI.
