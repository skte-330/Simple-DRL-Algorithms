import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import argparse
import json
import matplotlib.pyplot as plt

from algorithms.dqn import DQNAgent, DQNConfig

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--max-episodes", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)

    parser.add_argument("--buffer-capacity", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--target-update-frequency", type=int, default=50)

    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--final-decay-steps", type=int, default=10000)

    parser.add_argument("--double-dqn", action="store_true")
    parser.add_argument("--dueling-dqn", action="store_true")

    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-final", action="store_true")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false")

    return parser.parse_args()


def plot_returns(metrics: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    episodes = [m["episode"] for m in metrics]
    returns = [m["episode_return"] for m in metrics]
    mean_returns = [m["mean_return_20"] for m in metrics]

    plt.figure()
    plt.plot(episodes, returns, label="episode return")
    plt.plot(episodes, mean_returns, label="mean return 20")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("DQN Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    args = parse_args()

    # Building run name
    run_name = args.run_name
    if run_name is None:
        parts = ["dqn", args.env_id, f"seed{args.seed}"]
        if args.double_dqn:
            parts.append("double")
        if args.dueling_dqn:
            parts.append("dueling")
        run_name = "_".join(parts)

    print(f'Running {run_name} training.')

    # Setting directory
    output_dir = Path(args.output_dir) / run_name
    checkpoint_dir = Path(args.checkpoint_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_final:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Creating config
    cfg = DQNConfig(
        env_id=args.env_id,
        gamma=args.gamma,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        target_update_frequency=args.target_update_frequency,
        train_every=args.train_every,
        max_episodes=args.max_episodes,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        final_decay_steps=args.final_decay_steps,
        warmup_steps=args.warmup_steps,
        double_dqn=args.double_dqn,
        dueling_dqn=args.dueling_dqn,
        seed=args.seed,
        device=args.device,
    )

    # Saving training config
    with (output_dir / "config.json").open("w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    # Run DQN training
    agent = DQNAgent(cfg)
    metrics = agent.train()

    # Save training metrics
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
        
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

    print(f"Training finished.")
    print(f"Run name: {run_name}")
    print(f"Final global step: {final_global_step:.0f}")
    print(f"Final mean return over 20 episodes: {final_mean_return:.2f}")


if __name__ == "__main__":
    main()