import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import argparse
import json
import math
import matplotlib.pyplot as plt

from algorithms.trpo import TRPOAgent, TRPOConfig


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--total-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)

    parser.add_argument("--max-kl", type=float, default=0.01)
    parser.add_argument("--cg-iters", type=int, default=10)
    parser.add_argument("--cg-damping", type=float, default=0.1)
    parser.add_argument("--backtrack-iters", type=int, default=10)
    parser.add_argument("--backtrack-coeff", type=float, default=0.8)

    parser.add_argument("--critic-epochs", type=int, default=20)
    parser.add_argument("--critic-batch-size", type=int, default=64)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--critic-grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--hidden-dim", type=int, default=128)

    parser.add_argument("--no-normalize-advantages", dest="normalize_advantages", action="store_false")
    parser.set_defaults(normalize_advantages=True)

    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-final", action="store_true")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false")

    return parser.parse_args()


def _json_safe(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def plot_returns(metrics: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [m["epoch"] for m in metrics]

    plt.figure()

    last_points = [
        (m["epoch"], m["last_episode_return"])
        for m in metrics
        if math.isfinite(m["last_episode_return"])
    ]
    if last_points:
        x_last, y_last = zip(*last_points)
        plt.plot(x_last, y_last, label="last episode return")

    mean_points = [
        (m["epoch"], m["mean_return_20"])
        for m in metrics
        if math.isfinite(m["mean_return_20"])
    ]
    if mean_points:
        x_mean, y_mean = zip(*mean_points)
        plt.plot(x_mean, y_mean, label="mean return 20")

    plt.xlabel("Epoch")
    plt.ylabel("Return")
    plt.title("TRPO Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    args = parse_args()

    # Building run name
    run_name = args.run_name
    if run_name is None:
        run_name = "_".join(["trpo", args.env_id, f"seed{args.seed}"])

    print(f"Running {run_name} training.")

    # Setting directory
    output_dir = Path(args.output_dir) / run_name
    checkpoint_dir = Path(args.checkpoint_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_final:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    critic_grad_clip_norm = (
        None if args.critic_grad_clip_norm <= 0 else args.critic_grad_clip_norm
    )

    # Creating config
    cfg = TRPOConfig(
        env_id=args.env_id,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        max_kl=args.max_kl,
        cg_iters=args.cg_iters,
        cg_damping=args.cg_damping,
        backtrack_iters=args.backtrack_iters,
        backtrack_coeff=args.backtrack_coeff,
        critic_epochs=args.critic_epochs,
        critic_batch_size=args.critic_batch_size,
        critic_lr=args.critic_lr,
        critic_grad_clip_norm=critic_grad_clip_norm,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        total_epochs=args.total_epochs,
        normalize_advantages=args.normalize_advantages,
        seed=args.seed,
        device=args.device,
    )

    # Saving training config
    with (output_dir / "config.json").open("w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    # Run TRPO training
    agent = TRPOAgent(cfg)
    metrics = agent.train()

    # Saving metrics
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(_json_safe(metrics), f, indent=2)
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
    final_episode_count = metrics[-1]["episode"] if metrics else 0

    print("Training finished.")
    print(f"Run name: {run_name}")
    print(f"Final global step: {final_global_step:.0f}")
    print(f"Final completed episodes: {final_episode_count:.0f}")
    print(f"Final mean return over 20 episodes: {final_mean_return:.2f}")


if __name__ == "__main__":
    main()
