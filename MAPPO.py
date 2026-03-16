import argparse
import csv
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from mpe2 import simple_spread_v3


ENV_KWARGS = {
    "N": 3,
    "local_ratio": 0.5,
    "max_cycles": 25,
    "continuous_actions": True,
}

CHECKPOINT_SHAPE_FIELDS = (
    "n_agents",
    "local_ratio",
    "max_cycles",
    "hidden_sizes",
)

ACTION_EPS = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(description="MAPPO v2 for PettingZoo MPE simple_spread_v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-episodes", type=int, default=5000)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--local-ratio", type=float, default=0.5)
    parser.add_argument("--hidden-sizes", type=str, default="128,128")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=6)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="./MAPPO_v2_checkpoints")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--checkpoint-path", type=str, default="./MAPPO_v2_checkpoints")
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def parse_hidden_sizes(hidden_sizes_text):
    sizes = []
    for part in hidden_sizes_text.split(","):
        part = part.strip()
        if not part:
            continue
        sizes.append(int(part))
    if not sizes:
        raise ValueError("hidden_sizes must contain at least one integer")
    return tuple(sizes)


def resolve_device(device_name):
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(n_agents, local_ratio, max_cycles, render_mode=None):
    env_kwargs = dict(ENV_KWARGS)
    env_kwargs["N"] = n_agents
    env_kwargs["local_ratio"] = local_ratio
    env_kwargs["max_cycles"] = max_cycles
    return simple_spread_v3.parallel_env(render_mode=render_mode, **env_kwargs)


def setup_run_dir(base_dir, run_name):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_run_name = run_name or f"mappo_v2_{timestamp}"
    run_dir = Path(base_dir) / final_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logger(run_dir):
    logger = logging.getLogger(f"mappo_v2_{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(run_dir / "train.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def build_full_obs_dict(obs_dict, agent_names, obs_dim):
    full_obs = {
        agent: np.zeros(obs_dim, dtype=np.float32)
        for agent in agent_names
    }
    for agent, obs in obs_dict.items():
        full_obs[agent] = np.asarray(obs, dtype=np.float32)
    return full_obs


def get_central_state(env, obs_dict, agent_names, obs_dim):
    if hasattr(env, "state"):
        try:
            state = env.state()
            if state is not None:
                return np.asarray(state, dtype=np.float32)
        except Exception:
            pass

    full_obs = build_full_obs_dict(obs_dict, agent_names, obs_dim)
    ordered_obs = [full_obs[agent] for agent in agent_names]
    return np.concatenate(ordered_obs, axis=0).astype(np.float32)


def orthogonal_init(module, gain=1.0):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        layers = []
        input_dim = obs_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.alpha_head = nn.Linear(input_dim, act_dim)
        self.beta_head = nn.Linear(input_dim, act_dim)

        self.backbone.apply(lambda module: orthogonal_init(module, gain=np.sqrt(2.0)))
        orthogonal_init(self.alpha_head, gain=0.01)
        orthogonal_init(self.beta_head, gain=0.01)

    def forward(self, obs):
        features = self.backbone(obs)
        alpha = F.softplus(self.alpha_head(features)) + 1.0
        beta = F.softplus(self.beta_head(features)) + 1.0
        return alpha, beta


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_sizes):
        super().__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.value_head = nn.Linear(input_dim, 1)

        self.backbone.apply(lambda module: orthogonal_init(module, gain=np.sqrt(2.0)))
        orthogonal_init(self.value_head, gain=1.0)

    def forward(self, state):
        features = self.backbone(state)
        return self.value_head(features)


def select_actions(actor, obs_dict, live_agents, device, deterministic=False):
    obs_batch = np.stack([np.asarray(obs_dict[agent], dtype=np.float32) for agent in live_agents], axis=0)
    obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)

    with torch.no_grad():
        alpha, beta = actor(obs_tensor)
        dist = Beta(alpha, beta)
        if deterministic:
            actions = alpha / (alpha + beta)
        else:
            actions = dist.sample()
        actions = actions.clamp(ACTION_EPS, 1.0 - ACTION_EPS)
        logprobs = dist.log_prob(actions).sum(dim=-1)

    actions_dict = {
        agent: actions[index].cpu().numpy().astype(np.float32)
        for index, agent in enumerate(live_agents)
    }
    logprob_dict = {
        agent: float(logprobs[index].item())
        for index, agent in enumerate(live_agents)
    }
    return actions_dict, logprob_dict


def compute_gae(rewards, dones, values, next_values, gamma, gae_lambda):
    n_steps = len(rewards)
    advantages = [0.0] * n_steps
    returns = [0.0] * n_steps
    gae = 0.0

    for t in range(n_steps - 1, -1, -1):
        reward_t = rewards[t]
        done_t = float(dones[t])
        value_t = values[t]
        next_value_t = next_values[t]

        not_done_t = 1.0 - done_t
        td_error_t = reward_t + gamma * next_value_t * not_done_t - value_t
        gae = td_error_t + gamma * gae_lambda * not_done_t * gae

        advantages[t] = gae
        returns[t] = gae + value_t

    return returns, advantages


def evaluate_policy(actor, args, device, agent_names, obs_dim):
    eval_env = make_env(
        n_agents=args.n_agents,
        local_ratio=args.local_ratio,
        max_cycles=args.max_cycles,
        render_mode=None,
    )

    episode_rewards = []
    episode_lengths = []

    for episode_idx in range(args.num_eval_episodes):
        obs_dict, infos = eval_env.reset(seed=args.seed + 100_000 + episode_idx)
        team_reward = 0.0
        step_count = 0

        while eval_env.agents:
            live_agents = list(eval_env.agents)
            actions_dict, _ = select_actions(
                actor=actor,
                obs_dict=obs_dict,
                live_agents=live_agents,
                device=device,
                deterministic=True,
            )
            next_obs_dict, rewards, terminations, truncations, infos = eval_env.step(actions_dict)

            step_reward_sum = 0.0
            step_reward_count = 0
            for agent in live_agents:
                step_reward_sum += rewards[agent]
                step_reward_count += 1
            team_reward += step_reward_sum / step_reward_count

            obs_dict = next_obs_dict
            step_count += 1

        episode_rewards.append(team_reward / max(1, step_count))
        episode_lengths.append(step_count)

    eval_env.close()
    return {
        "eval_mean_team_reward": float(np.mean(episode_rewards)),
        "eval_std_team_reward": float(np.std(episode_rewards)),
        "eval_mean_episode_length": float(np.mean(episode_lengths)),
    }


def save_checkpoint(path, actor, critic, actor_optimizer, critic_optimizer, episode, best_eval_reward, args):
    checkpoint = {
        "episode": episode,
        "best_eval_reward": best_eval_reward,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "config": vars(args),
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, actor, critic, actor_optimizer=None, critic_optimizer=None, device=None):
    checkpoint = torch.load(path, map_location=device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])

    if actor_optimizer is not None and "actor_optimizer_state_dict" in checkpoint:
        actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    if critic_optimizer is not None and "critic_optimizer_state_dict" in checkpoint:
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

    return checkpoint


def load_checkpoint_config(path):
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint.get("config", {})


def apply_checkpoint_shape_config(args, checkpoint_config):
    for field in CHECKPOINT_SHAPE_FIELDS:
        if field in checkpoint_config:
            setattr(args, field, checkpoint_config[field])
    return args


def flatten_trajectories(episode_data, agent_names, gamma, gae_lambda, device):
    all_obs = []
    all_states = []
    all_actions = []
    all_old_logprobs = []
    all_returns = []
    all_advantages = []

    for agent in agent_names:
        rewards = episode_data[agent]["rewards"]
        dones = episode_data[agent]["dones"]
        values = episode_data[agent]["values"]
        next_values = episode_data[agent]["next_values"]

        if not rewards:
            continue

        returns, advantages = compute_gae(
            rewards=rewards,
            dones=dones,
            values=values,
            next_values=next_values,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        all_obs.extend(episode_data[agent]["obs"])
        all_states.extend(episode_data[agent]["states"])
        all_actions.extend(episode_data[agent]["actions"])
        all_old_logprobs.extend(episode_data[agent]["logprobs"])
        all_returns.extend(returns)
        all_advantages.extend(advantages)

    if not all_obs:
        return None

    obs_tensor = torch.tensor(np.array(all_obs), dtype=torch.float32, device=device)
    states_tensor = torch.tensor(np.array(all_states), dtype=torch.float32, device=device)
    actions_tensor = torch.tensor(np.array(all_actions), dtype=torch.float32, device=device)
    old_logprobs_tensor = torch.tensor(np.array(all_old_logprobs), dtype=torch.float32, device=device)
    returns_tensor = torch.tensor(np.array(all_returns), dtype=torch.float32, device=device)
    advantages_tensor = torch.tensor(np.array(all_advantages), dtype=torch.float32, device=device)
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

    return {
        "obs": obs_tensor,
        "states": states_tensor,
        "actions": actions_tensor,
        "old_logprobs": old_logprobs_tensor,
        "returns": returns_tensor,
        "advantages": advantages_tensor,
    }


def ppo_update(actor, critic, actor_optimizer, critic_optimizer, batch, args):
    num_samples = batch["obs"].shape[0]
    last_actor_loss = 0.0
    last_critic_loss = 0.0
    last_entropy = 0.0

    minibatch_size = min(args.minibatch_size, num_samples)

    for _ in range(args.update_epochs):
        indices = torch.randperm(num_samples, device=batch["obs"].device)

        for start in range(0, num_samples, minibatch_size):
            mb_idx = indices[start:start + minibatch_size]

            mb_obs = batch["obs"][mb_idx]
            mb_states = batch["states"][mb_idx]
            mb_actions = batch["actions"][mb_idx]
            mb_old_logprobs = batch["old_logprobs"][mb_idx]
            mb_returns = batch["returns"][mb_idx]
            mb_advantages = batch["advantages"][mb_idx]

            alpha, beta = actor(mb_obs)
            dist = Beta(alpha, beta)
            clipped_actions = mb_actions.clamp(ACTION_EPS, 1.0 - ACTION_EPS)
            new_logprobs = dist.log_prob(clipped_actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            ratio = torch.exp(new_logprobs - mb_old_logprobs)
            unclipped_objective = ratio * mb_advantages
            clipped_objective = torch.clamp(
                ratio,
                1.0 - args.clip_epsilon,
                1.0 + args.clip_epsilon,
            ) * mb_advantages

            actor_loss = -torch.min(unclipped_objective, clipped_objective).mean()
            actor_loss = actor_loss - args.entropy_coef * entropy

            values_pred = critic(mb_states).squeeze(-1)
            critic_loss = args.value_coef * F.mse_loss(values_pred, mb_returns)

            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
            critic_optimizer.step()

            last_actor_loss = float(actor_loss.item())
            last_critic_loss = float(critic_loss.item())
            last_entropy = float(entropy.item())

    return last_actor_loss, last_critic_loss, last_entropy


def train(args):
    if args.resume_from:
        checkpoint_config = load_checkpoint_config(args.resume_from)
        args = apply_checkpoint_shape_config(args, checkpoint_config)

    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    device = resolve_device(args.device)

    torch.set_num_threads(args.torch_threads)
    set_seed(args.seed)

    run_dir = setup_run_dir(args.output_dir, args.run_name)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir)
    logger.info("Run directory: %s", run_dir)
    logger.info("Using device: %s", device)

    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(vars(args), config_file, indent=2, sort_keys=True)

    train_env = make_env(
        n_agents=args.n_agents,
        local_ratio=args.local_ratio,
        max_cycles=args.max_cycles,
        render_mode=None,
    )

    obs_dict, infos = train_env.reset(seed=args.seed)
    agent_names = list(train_env.possible_agents)
    obs_dim = train_env.observation_space(agent_names[0]).shape[0]
    act_dim = train_env.action_space(agent_names[0]).shape[0]
    state_dim = get_central_state(train_env, obs_dict, agent_names, obs_dim).shape[0]

    actor = Actor(obs_dim, act_dim, hidden_sizes).to(device)
    critic = Critic(state_dim, hidden_sizes).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, eps=1e-5)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr, eps=1e-5)

    start_episode = 1
    best_eval_reward = -float("inf")

    if args.resume_from:
        checkpoint = load_checkpoint(
            path=args.resume_from,
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            device=device,
        )
        start_episode = int(checkpoint["episode"]) + 1
        best_eval_reward = float(checkpoint.get("best_eval_reward", best_eval_reward))
        logger.info("Resumed from checkpoint: %s", args.resume_from)

    logger.info(
        "Env: agents=%d obs_dim=%d act_dim=%d state_dim=%d hidden_sizes=%s",
        len(agent_names),
        obs_dim,
        act_dim,
        state_dim,
        hidden_sizes,
    )

    metrics_path = run_dir / "metrics.csv"
    fieldnames = [
        "episode",
        "train_mean_team_reward",
        "train_episode_steps",
        "num_samples",
        "actor_loss",
        "critic_loss",
        "entropy",
        "eval_mean_team_reward",
        "eval_std_team_reward",
        "eval_mean_episode_length",
    ]

    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        writer.writeheader()

        for episode in range(start_episode, args.num_episodes + 1):
            obs_dict, infos = train_env.reset(seed=args.seed + episode)
            episode_data = {
                agent: {
                    "obs": [],
                    "states": [],
                    "actions": [],
                    "logprobs": [],
                    "rewards": [],
                    "dones": [],
                    "values": [],
                    "next_values": [],
                }
                for agent in agent_names
            }

            episode_team_reward = 0.0
            episode_step_count = 0

            actor.train()
            critic.train()

            while train_env.agents: # creates multiple timesteps within one episode
                live_agents = list(train_env.agents)
                state_t_np = get_central_state(train_env, obs_dict, agent_names, obs_dim)
                state_t = torch.tensor(state_t_np, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    value_t = float(critic(state_t).item())

                actions_dict, logprob_dict = select_actions(
                    actor=actor,
                    obs_dict=obs_dict,
                    live_agents=live_agents,
                    device=device,
                    deterministic=False,
                )

                next_obs_dict, rewards, terminations, truncations, infos = train_env.step(actions_dict)

                step_reward_sum = 0.0
                step_reward_count = 0
                for agent in live_agents:
                    step_reward_sum += rewards[agent]
                    step_reward_count += 1
                step_team_reward = step_reward_sum / step_reward_count
                episode_team_reward += step_team_reward
                episode_step_count += 1

                if train_env.agents:
                    state_tp1_np = get_central_state(train_env, next_obs_dict, agent_names, obs_dim)
                    state_tp1_t = torch.tensor(state_tp1_np, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        next_value = float(critic(state_tp1_t).item())
                else:
                    next_value = 0.0

                for agent in live_agents:
                    done = bool(
                        terminations.get(agent, False)
                        or truncations.get(agent, False)
                    )
                    episode_data[agent]["obs"].append(np.asarray(obs_dict[agent], dtype=np.float32))
                    episode_data[agent]["states"].append(state_t_np)
                    episode_data[agent]["actions"].append(actions_dict[agent])
                    episode_data[agent]["logprobs"].append(logprob_dict[agent])
                    episode_data[agent]["rewards"].append(step_team_reward)
                    episode_data[agent]["dones"].append(done)
                    episode_data[agent]["values"].append(value_t)
                    episode_data[agent]["next_values"].append(next_value)

                obs_dict = next_obs_dict

            batch = flatten_trajectories(
                episode_data=episode_data,
                agent_names=agent_names,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                device=device,
            )

            if batch is None:
                logger.warning("Episode %d produced no samples; skipping update", episode)
                continue

            actor_loss, critic_loss, entropy = ppo_update(
                actor=actor,
                critic=critic,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                batch=batch,
                args=args,
            )

            train_mean_team_reward = episode_team_reward / max(1, episode_step_count)
            num_samples = int(batch["obs"].shape[0])

            eval_metrics = {
                "eval_mean_team_reward": "",
                "eval_std_team_reward": "",
                "eval_mean_episode_length": "",
            }

            if episode % args.eval_every == 0:
                actor.eval()
                critic.eval()
                eval_metrics = evaluate_policy(
                    actor=actor,
                    args=args,
                    device=device,
                    agent_names=agent_names,
                    obs_dim=obs_dim,
                )

                logger.info(
                    "Episode %04d | TrainReward %.4f | EvalReward %.4f +/- %.4f | EvalLen %.2f | Samples %d",
                    episode,
                    train_mean_team_reward,
                    eval_metrics["eval_mean_team_reward"],
                    eval_metrics["eval_std_team_reward"],
                    eval_metrics["eval_mean_episode_length"],
                    num_samples,
                )

                if eval_metrics["eval_mean_team_reward"] > best_eval_reward:
                    best_eval_reward = eval_metrics["eval_mean_team_reward"]
                    save_checkpoint(
                        path=checkpoint_dir / "best.pt",
                        actor=actor,
                        critic=critic,
                        actor_optimizer=actor_optimizer,
                        critic_optimizer=critic_optimizer,
                        episode=episode,
                        best_eval_reward=best_eval_reward,
                        args=args,
                    )
            else:
                logger.info(
                    "Episode %04d | TrainReward %.4f | ActorLoss %.4f | CriticLoss %.4f | Entropy %.4f | Samples %d",
                    episode,
                    train_mean_team_reward,
                    actor_loss,
                    critic_loss,
                    entropy,
                    num_samples,
                )

            writer.writerow(
                {
                    "episode": episode,
                    "train_mean_team_reward": train_mean_team_reward,
                    "train_episode_steps": episode_step_count,
                    "num_samples": num_samples,
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "entropy": entropy,
                    "eval_mean_team_reward": eval_metrics["eval_mean_team_reward"],
                    "eval_std_team_reward": eval_metrics["eval_std_team_reward"],
                    "eval_mean_episode_length": eval_metrics["eval_mean_episode_length"],
                }
            )
            metrics_file.flush()

            if episode % args.save_every == 0:
                save_checkpoint(
                    path=checkpoint_dir / "latest.pt",
                    actor=actor,
                    critic=critic,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    episode=episode,
                    best_eval_reward=best_eval_reward,
                    args=args,
                )
                save_checkpoint(
                    path=checkpoint_dir / f"episode_{episode:04d}.pt",
                    actor=actor,
                    critic=critic,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    episode=episode,
                    best_eval_reward=best_eval_reward,
                    args=args,
                )

    save_checkpoint(
        path=checkpoint_dir / "final.pt",
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        episode=args.num_episodes,
        best_eval_reward=best_eval_reward,
        args=args,
    )
    train_env.close()
    logger.info("Training finished. Final checkpoint saved to %s", checkpoint_dir / "final.pt")


def run_eval_only(args):
    if not args.checkpoint_path:
        raise ValueError("--eval-only requires --checkpoint-path")

    checkpoint_config = load_checkpoint_config(args.checkpoint_path)
    args = apply_checkpoint_shape_config(args, checkpoint_config)

    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    device = resolve_device(args.device)
    torch.set_num_threads(args.torch_threads)
    set_seed(args.seed)

    env = make_env(
        n_agents=args.n_agents,
        local_ratio=args.local_ratio,
        max_cycles=args.max_cycles,
        render_mode=None,
    )
    obs_dict, infos = env.reset(seed=args.seed)
    agent_names = list(env.possible_agents)
    obs_dim = env.observation_space(agent_names[0]).shape[0]
    act_dim = env.action_space(agent_names[0]).shape[0]
    state_dim = get_central_state(env, obs_dict, agent_names, obs_dim).shape[0]
    env.close()

    actor = Actor(obs_dim, act_dim, hidden_sizes).to(device)
    critic = Critic(state_dim, hidden_sizes).to(device)
    checkpoint = load_checkpoint(
        path=args.checkpoint_path,
        actor=actor,
        critic=critic,
        actor_optimizer=None,
        critic_optimizer=None,
        device=device,
    )
    actor.eval()

    metrics = evaluate_policy(
        actor=actor,
        args=args,
        device=device,
        agent_names=agent_names,
        obs_dim=obs_dim,
    )

    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Checkpoint episode: {checkpoint.get('episode', 'unknown')}")
    print(f"Eval mean team reward: {metrics['eval_mean_team_reward']:.4f}")
    print(f"Eval std team reward:  {metrics['eval_std_team_reward']:.4f}")
    print(f"Eval mean length:      {metrics['eval_mean_episode_length']:.2f}")


def main():
    args = parse_args()
    if args.mode == "eval":
        run_eval_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
