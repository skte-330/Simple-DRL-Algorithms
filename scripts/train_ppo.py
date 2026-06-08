import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import argparse
import json
import math
import matplotlib.pyplot as plt

from algorithms.ppo import PPOAgent, PPOConfig


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--total-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--target-kl", type=float, default=None)

    parser.add_argument("--actor-lr", type=float, default=2.5e-4)
    parser.add_argument("--critic-lr", type=float, default=2.5e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--normalize-advantages", action="store_true", default=True)
    parser.add_argument("--no-normalize-advantages", dest="normalize_advantages", action="store_false")

    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-final", action="store_true")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false")

    return parser.parse_args()


def _finite_or_none(x: float):
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


def save_metrics(metrics: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [
        {key: _finite_or_none(value) for key, value in row.items()}
        for row in metrics
    ]
    with path.open("w") as f:
        json.dump(serializable, f, indent=2)


def plot_returns(metrics: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [m["epoch"] for m in metrics]
    last_returns = [m["last_episode_return"] for m in metrics]
    mean_returns = [m["mean_return_20"] for m in metrics]

    plt.figure()
    plt.plot(epochs, last_returns, label="last episode return")
    plt.plot(epochs, mean_returns, label="mean return 20")
    plt.xlabel("Epoch")
    plt.ylabel("Return")
    plt.title("PPO Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    args = parse_args()

    # Building run name
    run_name = args.run_name
    if run_name is None:
        run_name = "_".join(["ppo", args.env_id, f"seed{args.seed}"])

    print(f"Running {run_name} training.")

    # Setting directory
    output_dir = Path(args.output_dir) / run_name
    checkpoint_dir = Path(args.checkpoint_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_final:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Creating config
    cfg = PPOConfig(
        env_id=args.env_id,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip=args.clip,
        entropy_coef=args.entropy_coef,
        normalize_advantages=args.normalize_advantages,
        target_kl=args.target_kl,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        total_epochs=args.total_epochs,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        seed=args.seed,
        device=args.device,
    )

    # Saving training config
    with (output_dir / "config.json").open("w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    # Run PPO training
    agent = PPOAgent(cfg)
    metrics = agent.train()

    # Save training metrics
    metrics_path = output_dir / "metrics.json"
    save_metrics(metrics, metrics_path)
    print(f"Metrics saved to: {metrics_path}")

    # Plot training returns
    if args.plot:
        curve_path = output_dir / "return_curve.png"
        plot_returns(metrics, curve_path)
        print(f"Return curve saved to: {curve_path}")

    # Save final models
    if args.save_final:
        final_checkpoint_path = checkpoint_dir / "final.pt"
        agent.save(final_checkpoint_path)
        print(f"Checkpoint saved to: {final_checkpoint_path}")

    # Print info and quit
    final_mean_return = metrics[-1]["mean_return_20"] if metrics else float("nan")
    final_global_step = metrics[-1]["global_step"] if metrics else 0
    final_episode = metrics[-1]["episode"] if metrics else 0

    print("Training finished.")
    print(f"Run name: {run_name}")
    print(f"Final global step: {final_global_step:.0f}")
    print(f"Completed episodes: {final_episode:.0f}")
    print(f"Final mean return over 20 episodes: {final_mean_return:.2f}")


if __name__ == "__main__":
    main()
