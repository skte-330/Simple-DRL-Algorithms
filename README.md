# Deep Reinforcement Learning Algorithms

Clean PyTorch implementations of classical deep reinforcement learning algorithms. The current refactor starts from DQN and gradually migrates the legacy implementations into a unified training interface.

## Implemented Algorithms

| Algorithm | Type | Action Space | Environment | Status |
| --- | --- | --- | --- | --- |
| DQN | Value-based | Discrete | CartPole-v1 | Migrated |
| Double DQN | Value-based | Discrete | CartPole-v1 | Migrated |
| Dueling DQN | Value-based | Discrete | CartPole-v1 | Migrated |
| DDPG | Off-policy Actor-Critic | Continuous | Pendulum-v1 | Legacy |
| PPO | Policy Optimization | Discrete | CartPole-v1 | Legacy |
| SAC | Off-policy Actor-Critic | Continuous | Pendulum-v1 | Legacy |
| TRPO | Trust-region Actor-Critic | Discrete | CartPole-v1 | Legacy |

## Highlights

- Clean PyTorch implementations from scratch
- Unified training entry point, starting with DQN
- Configurable hyperparameters with YAML
- Reproducible experiments with fixed random seeds
- Training metrics saved as CSV for plotting and analysis

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Train vanilla DQN on CartPole:

```bash
python scripts/train.py --algo dqn --config configs/dqn_cartpole.yaml
```

Train Double DQN:

```bash
python scripts/train.py --algo dqn --config configs/dqn_cartpole.yaml --double-dqn
```

Train Dueling DQN:

```bash
python scripts/train.py --algo dqn --config configs/dqn_cartpole.yaml --dueling-dqn
```

Each run saves `config.json`, `metrics.csv`, and `model.pt` under `runs/`.

## Project Structure

```text
algorithms/     Migrated algorithm implementations
utils/          Shared buffers and neural network modules
configs/        Reproducible experiment configs
scripts/        Command-line training entry points
legacy/         Original exploratory implementations
runs/           Local training outputs
```

## Roadmap

- [x] Migrate DQN, Double DQN, and Dueling DQN into the new structure
- [x] Add a basic command-line training entry point
- [ ] Add plotting utilities for saved CSV metrics
- [ ] Migrate PPO, DDPG, SAC, and TRPO
- [ ] Add benchmark tables and learning curves
- [ ] Add technical notes for each algorithm
