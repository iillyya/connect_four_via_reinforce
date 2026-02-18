import optuna
import json
import pandas as pd
import os

from train import train, MODEL_PATH_DEFAULT, set_global_seed
from evaluate import run_evaluation

EPISODES_PER_TRIAL = 10000
EVAL_GAMES = 500
BASE_SEED = 42


# Objective Function
def objective(trial):
    # Hyperparameter search space
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    gamma = 1.0
    beta = trial.suggest_float("beta", 0.001, 0.05)
    beta_decay = trial.suggest_float("beta_decay", 0.995, 1.0)
    update_every = trial.suggest_categorical("update_every", [4, 8, 16, 32, 64])
    lag_k = trial.suggest_categorical("lag_k", [500, 1000, 2000, 5000])
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 2.0)

    seed = BASE_SEED + trial.number * 10
    set_global_seed(seed)

    
    # Train 
    model_path = f"trial_{trial.number}.pt"
    train(
        num_episodes=EPISODES_PER_TRIAL,
        lr=lr,
        gamma=gamma,
        beta=beta,
        beta_decay=beta_decay,
        update_every=update_every,
        lag_k=lag_k,
        max_grad_norm=max_grad_norm,
        checkpoint_every=None,
        verbose=False,
        model_path=model_path,
        opponent_mix="lagged:0.4,random:0.2,heuristic:0.4",
        start_mode="random",
        max_num_random_openings=4,
    )

    # Evaluate 
    eval_report = run_evaluation(
        model_path=model_path,
        num_games=EVAL_GAMES,
        opponents=["random", "heuristic"],
        swap_sides=True,
        seed=seed,
    )

    random_score = eval_report["results"]["random"]["score"]
    heuristic_score = eval_report["results"]["heuristic"]["score"]

    heur_min = min(eval_report["results"]["heuristic"]["agent_first"]["score"],
                    eval_report["results"]["heuristic"]["agent_second"]["score"])

    avg_score = 0.5 * (random_score + heur_min)

    # Log additional metrics inside trial
    trial.set_user_attr("random_score", random_score)
    trial.set_user_attr("heuristic_score", heuristic_score)
    trial.set_user_attr("heuristic_min_score", heur_min)
    print(f"Trial {trial.number}: random_score={random_score:.3f}, heuristic_score={heuristic_score:.3f}, heuristic_min_score={heur_min:.3f}, avg_score={avg_score:.3f}")

    return avg_score


if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)

    print("\nBest trial:")
    print("Score:", study.best_value)
    print("Params:", study.best_params)

    with open("optuna_best_params_v3.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    df.to_csv("optuna_all_trials_v3.csv", index=False)

    all_trials = []
    for t in study.trials:
        all_trials.append({
            "trial_number": t.number,
            "score": t.value,
            "params": t.params,
            "random_score": t.user_attrs.get("random_score"),
            "heuristic_score": t.user_attrs.get("heuristic_score"),
        })

    with open("optuna_all_trials_v3.json", "w") as f:
        json.dump(all_trials, f, indent=2)

    print("\nSaved:")
    print(" - optuna_best_params.json")
    print(" - optuna_all_trials.csv")
    print(" - optuna_all_trials.json")