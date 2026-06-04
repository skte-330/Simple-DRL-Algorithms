# Deep Reinforcement Learning Algorithms

Clean PyTorch implementations of classical deep reinforcement learning algorithms.

## Implemented Algorithms

| Algorithm | Type | Action Space | Environment | Status |
| --- | --- | --- | --- | --- |
| DQN/Double DQN/ Dueling DQN | Value-based | Discrete | CartPole-v1/Atari | Done |
| DDPG | Off-policy Actor-Critic | Continuous | Pendulum-v1 | Done |
| TRPO | Trust-region Actor-Critic | Discrete | CartPole-v1 | Done |
| PPO | On-policy Actor-Critic | Discrete | CartPole-v1 | Done |
| SAC | Off-policy Actor-Critic | Continuous | Pendulum-v1 | Done |

## Highlights

- Clean PyTorch implementations from scratch
- Unified training entry point, starting with DQN
- Reproducible experiments with fixed random seeds
- Training metrics saved as json for plotting and analysis

## Dependency
- Python 3.10-3.12
- PyTorch
- Openai gymnasium, and ale_py for atari
- numpy, tqdm

## Quick Start

Train vanilla DQN on CartPole:

```bash
python scripts/train_dqn.py 
```

Train Double DQN:

```bash
python scripts/train_dqn.py --double-dqn
```

Each run saves `config.json`, `metrics.json` and `return_curve.png`  under `results/`.

## Project Structure

```text
algorithms/     Algorithm implementations
utils/          Shared training utils
scripts/        Command-line training entry points
results/        Local training outputs
checkpoints/    Model saving directory
```

## TODOS

- [x] Implement all algorithms
- [ ] Add wandb logging
- [ ] Support more environments in TRPO and PPO
- [ ] Add evaluation and visualization from checkpoints
- [ ] Add requirements.txt or uv.lock for environment setting
- [ ] Add technical notes for each algorithm
- [ ] Add references
