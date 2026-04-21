# Multi-Agent RL Applications

This repository compares three multi-agent reinforcement learning algorithms on a shared cooperative control task:

- `MAPPO.py`
- `MADDPG.py`
- `MASAC.py`

The aim is to study how different MARL training paradigms behave under the same environment, action setting, network scale, and evaluation metric.

## Environment

All three implementations are configured for the PettingZoo MPE environment `simple_spread_v3` via `mpe2`, with the same default environment setup:

- Number of agents: `3`
- Action space: continuous
- `local_ratio = 0.5`
- `max_cycles = 25`

The task is cooperative coverage and coordination: agents must spread out and cover landmarks while avoiding poor team behavior. For consistency across algorithms, the training loop uses a shared team reward at each step by averaging the live agents' rewards.

## Algorithms

The repository includes three standard MARL baselines:

- `MAPPO.py`: Multi-Agent Proximal Policy Optimization
- `MADDPG.py`: Multi-Agent Deep Deterministic Policy Gradient
- `MASAC.py`: Multi-Agent Soft Actor-Critic

### Algorithm Properties

| Algorithm | Policy Type | Actor | Critic | Training Regime | Centralized Training | Decentralized Execution | Replay Buffer | Target Networks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MAPPO | On-policy | Shared stochastic Beta policy, conditioned on local observation | Centralized value critic `V(s)` using global state | PPO with GAE and minibatch updates | Yes | Yes | No | No |
| MADDPG | Off-policy | Deterministic per-agent actor, conditioned on local observation | Centralized per-agent critic `Q_i(o_1,\dots,o_n,a_1,\dots,a_n)` | Replay-based actor-critic with soft target updates | Yes | Yes | Yes | Yes |
| MASAC | Off-policy | Stochastic per-agent Gaussian actor, conditioned on local observation | Centralized twin critics with entropy regularization | Replay-based maximum-entropy actor-critic | Yes | Yes | Yes | Yes |

### Centralized vs. Decentralized

- `MAPPO` uses a decentralized actor and a centralized critic.
- `MADDPG` uses decentralized actors and centralized critics.
- `MASAC` uses decentralized actors and centralized twin critics.

In all three cases, the training setup follows the common CTDE pattern: centralized training, decentralized execution.

## Repository Structure

- [`MAPPO.py`](MAPPO.py): cooperative continuous-action MAPPO baseline
- [`MADDPG.py`](MADDPG.py): cooperative continuous-action MADDPG baseline
- [`MASAC.py`](MASAC.py): cooperative continuous-action MASAC baseline
- [`MARL_comparison.png`](MARL_comparison.png): comparison figure for evaluation performance

## Running

Examples:

```bash
python MAPPO.py
python MADDPG.py
python MASAC.py
```

Each script supports training and evaluation modes, checkpoint saving, and periodic evaluation logging through command-line arguments.

## Notes

- All methods are compared on the same environment family and continuous-action setting.
- The shared team reward is used to keep the comparison protocol aligned across algorithms.
- Raw learning behavior can still differ substantially because the algorithms are fundamentally different: PPO-style on-policy updates for MAPPO, deterministic off-policy actor-critic for MADDPG, and entropy-regularized off-policy actor-critic for MASAC.
