import argparse
import csv
import json
import logging
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

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
    "shared_reward",
)


def parse_args():
    parser = argparse.ArgumentParser(description="MASAC v2 for PettingZoo MPE simple_spread_v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-episodes", type=int, default=5000)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--local-ratio", type=float, default=0.5)
    parser.add_argument("--hidden-sizes", type=str, default="128,128")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--init-alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-capacity", type=int, default=100000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--shared-reward", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="./MASAC_v2_checkpoints")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--checkpoint-path", type=str, default="./MASAC_v2_checkpoints")
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
    final_run_name = run_name or f"masac_v2_{timestamp}"
    run_dir = Path(base_dir) / final_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logger(run_dir):
    logger = logging.getLogger(f"masac_v2_{run_dir.name}")
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


def orthogonal_init(module, gain=1.0):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, actions, rewards, next_obs, dones):
        self.buffer.append((obs, actions, rewards, next_obs, dones))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)

        obs = np.array([item[0] for item in batch], dtype=np.float32)
        actions = np.array([item[1] for item in batch], dtype=np.float32)
        rewards = np.array([item[2] for item in batch], dtype=np.float32)
        next_obs = np.array([item[3] for item in batch], dtype=np.float32)
        dones = np.array([item[4] for item in batch], dtype=np.float32)

        return (
            torch.tensor(obs, dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.float32, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.tensor(next_obs, dtype=torch.float32, device=device),
            torch.tensor(dones, dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)


class GaussianActor(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        layers = []
        input_dim = obs_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(input_dim, act_dim)
        self.log_std_head = nn.Linear(input_dim, act_dim)

        self.backbone.apply(lambda module: orthogonal_init(module, gain=np.sqrt(2.0)))
        orthogonal_init(self.mean_head, gain=0.01)
        orthogonal_init(self.log_std_head, gain=0.01)

        self.register_buffer("action_scale", torch.tensor(0.5, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(0.5, dtype=torch.float32))

    def forward(self, obs):
        features = self.backbone(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        normal = Normal(mean, std)
        pre_tanh = normal.rsample()
        squashed = torch.tanh(pre_tanh)

        action = squashed * self.action_scale + self.action_bias
        log_prob = normal.log_prob(pre_tanh)
        log_prob = log_prob - torch.log(self.action_scale * (1.0 - squashed.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


class Critic(nn.Module):
    def __init__(self, total_obs_dim, total_act_dim, hidden_sizes):
        super().__init__()
        layers = []
        input_dim = total_obs_dim + total_act_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.q_head = nn.Linear(input_dim, 1)

        self.backbone.apply(lambda module: orthogonal_init(module, gain=np.sqrt(2.0)))
        orthogonal_init(self.q_head, gain=1.0)

    def forward(self, all_obs_flat, all_actions_flat):
        critic_input = torch.cat([all_obs_flat, all_actions_flat], dim=1)
        features = self.backbone(critic_input)
        return self.q_head(features)


class MASACAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        total_obs_dim,
        total_act_dim,
        hidden_sizes,
        actor_lr,
        critic_lr,
        alpha_lr,
        init_alpha,
        device,
    ):
        self.actor = GaussianActor(obs_dim, act_dim, hidden_sizes).to(device)

        self.q1 = Critic(total_obs_dim, total_act_dim, hidden_sizes).to(device)
        self.q2 = Critic(total_obs_dim, total_act_dim, hidden_sizes).to(device)
        self.target_q1 = Critic(total_obs_dim, total_act_dim, hidden_sizes).to(device)
        self.target_q2 = Critic(total_obs_dim, total_act_dim, hidden_sizes).to(device)

        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=critic_lr,
            eps=1e-5,
        )

        init_log_alpha = float(np.log(init_alpha))
        self.log_alpha = torch.tensor([init_log_alpha], dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr, eps=1e-5)
        self.target_entropy = -float(act_dim)

    @property
    def alpha(self):
        return self.log_alpha.exp()


def soft_update(target_net, source_net, tau):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)


def set_requires_grad(model, flag):
    for parameter in model.parameters():
        parameter.requires_grad = flag


def select_action(actor, obs_np, device, deterministic=False):
    obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action, _, mean_action = actor.sample(obs_tensor)
    chosen_action = mean_action if deterministic else action
    return chosen_action.squeeze(0).cpu().numpy().astype(np.float32)


def evaluate_policy(agents, args, device, agent_names, obs_dim):
    eval_env = make_env(
        n_agents=args.n_agents,
        local_ratio=args.local_ratio,
        max_cycles=args.max_cycles,
        render_mode=None,
    )
    name_to_idx = {name: idx for idx, name in enumerate(agent_names)}

    episode_rewards = []
    episode_lengths = []

    for eval_episode in range(args.num_eval_episodes):
        obs_dict, infos = eval_env.reset(seed=args.seed + 100_000 + eval_episode)
        obs_full = build_full_obs_dict(obs_dict, agent_names, obs_dim)
        episode_team_return = 0.0
        episode_length = 0

        while eval_env.agents:
            live_agents = list(eval_env.agents)
            actions_dict = {}

            for agent_name in live_agents:
                agent_idx = name_to_idx[agent_name]
                actions_dict[agent_name] = select_action(
                    actor=agents[agent_idx].actor,
                    obs_np=obs_full[agent_name],
                    device=device,
                    deterministic=True,
                )

            next_obs_dict, rewards, terminations, truncations, infos = eval_env.step(actions_dict)
            next_obs_full = build_full_obs_dict(next_obs_dict, agent_names, obs_dim)

            step_reward_sum = 0.0
            step_reward_count = 0
            for agent_name in live_agents:
                step_reward_sum += rewards[agent_name]
                step_reward_count += 1
            episode_team_return += step_reward_sum / step_reward_count

            obs_full = next_obs_full
            episode_length += 1

        episode_rewards.append(episode_team_return / max(1, episode_length))
        episode_lengths.append(episode_length)

    eval_env.close()
    return {
        "eval_mean_team_reward": float(np.mean(episode_rewards)),
        "eval_std_team_reward": float(np.std(episode_rewards)),
        "eval_mean_episode_length": float(np.mean(episode_lengths)),
    }


def save_checkpoint(path, agents, episode, global_step, best_eval_reward, replay_size, args):
    checkpoint = {
        "episode": episode,
        "global_step": global_step,
        "best_eval_reward": best_eval_reward,
        "replay_size": replay_size,
        "config": vars(args),
        "agents": [],
    }

    for agent in agents:
        checkpoint["agents"].append(
            {
                "actor_state_dict": agent.actor.state_dict(),
                "q1_state_dict": agent.q1.state_dict(),
                "q2_state_dict": agent.q2.state_dict(),
                "target_q1_state_dict": agent.target_q1.state_dict(),
                "target_q2_state_dict": agent.target_q2.state_dict(),
                "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
                "q_optimizer_state_dict": agent.q_optimizer.state_dict(),
                "log_alpha": agent.log_alpha.detach().cpu(),
                "alpha_optimizer_state_dict": agent.alpha_optimizer.state_dict(),
            }
        )

    torch.save(checkpoint, path)


def load_checkpoint(path, agents, device):
    checkpoint = torch.load(path, map_location=device)

    for agent, saved_agent in zip(agents, checkpoint["agents"]):
        agent.actor.load_state_dict(saved_agent["actor_state_dict"])
        agent.q1.load_state_dict(saved_agent["q1_state_dict"])
        agent.q2.load_state_dict(saved_agent["q2_state_dict"])
        agent.target_q1.load_state_dict(saved_agent["target_q1_state_dict"])
        agent.target_q2.load_state_dict(saved_agent["target_q2_state_dict"])
        agent.actor_optimizer.load_state_dict(saved_agent["actor_optimizer_state_dict"])
        agent.q_optimizer.load_state_dict(saved_agent["q_optimizer_state_dict"])

        saved_log_alpha = saved_agent["log_alpha"].to(device)
        agent.log_alpha.data.copy_(saved_log_alpha)
        agent.alpha_optimizer.load_state_dict(saved_agent["alpha_optimizer_state_dict"])

    return checkpoint


def load_checkpoint_config(path):
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint.get("config", {})


def apply_checkpoint_shape_config(args, checkpoint_config):
    for field in CHECKPOINT_SHAPE_FIELDS:
        if field in checkpoint_config:
            setattr(args, field, checkpoint_config[field])
    return args


def masac_update(agents, replay_buffer, batch_size, gamma, tau, device, max_grad_norm):
    obs, actions, rewards, next_obs, dones = replay_buffer.sample(batch_size, device)
    batch_count = obs.shape[0]

    obs_flat = obs.reshape(batch_count, -1)
    actions_flat = actions.reshape(batch_count, -1)
    next_obs_flat = next_obs.reshape(batch_count, -1)

    with torch.no_grad():
        next_actions_list = []
        next_logprobs_list = []
        for agent_index, agent in enumerate(agents):
            next_action, next_logprob, _ = agent.actor.sample(next_obs[:, agent_index, :])
            next_actions_list.append(next_action)
            next_logprobs_list.append(next_logprob)

        next_actions = torch.stack(next_actions_list, dim=1)
        next_actions_flat = next_actions.reshape(batch_count, -1)

    actor_losses = []
    critic_losses = []
    alpha_losses = []

    for agent_index, agent in enumerate(agents):
        with torch.no_grad():
            target_q1 = agent.target_q1(next_obs_flat, next_actions_flat)
            target_q2 = agent.target_q2(next_obs_flat, next_actions_flat)
            target_min_q = torch.min(target_q1, target_q2)

            y = rewards[:, agent_index:agent_index + 1] + gamma * (1.0 - dones[:, agent_index:agent_index + 1]) * (
                target_min_q - agent.alpha * next_logprobs_list[agent_index]
            )

        current_q1 = agent.q1(obs_flat, actions_flat)
        current_q2 = agent.q2(obs_flat, actions_flat)
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)

        critic_parameters = list(agent.q1.parameters()) + list(agent.q2.parameters())
        agent.q_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic_parameters, max_grad_norm)
        agent.q_optimizer.step()

        set_requires_grad(agent.q1, False)
        set_requires_grad(agent.q2, False)

        new_action_i, logprob_i, _ = agent.actor.sample(obs[:, agent_index, :])
        policy_actions = actions.clone()
        policy_actions[:, agent_index, :] = new_action_i
        policy_actions_flat = policy_actions.reshape(batch_count, -1)

        q1_pi = agent.q1(obs_flat, policy_actions_flat)
        q2_pi = agent.q2(obs_flat, policy_actions_flat)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (agent.alpha.detach() * logprob_i - min_q_pi).mean()

        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(agent.actor.parameters(), max_grad_norm)
        agent.actor_optimizer.step()

        set_requires_grad(agent.q1, True)
        set_requires_grad(agent.q2, True)

        alpha_loss = -(agent.log_alpha * (logprob_i + agent.target_entropy).detach()).mean()
        agent.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        agent.alpha_optimizer.step()

        soft_update(agent.target_q1, agent.q1, tau)
        soft_update(agent.target_q2, agent.q2, tau)

        actor_losses.append(float(actor_loss.item()))
        critic_losses.append(float(critic_loss.item()))
        alpha_losses.append(float(alpha_loss.item()))

    return (
        float(np.mean(actor_losses)),
        float(np.mean(critic_losses)),
        float(np.mean(alpha_losses)),
    )


def train(args):
    if args.resume_from:
        checkpoint_config = load_checkpoint_config(args.resume_from)
        args = apply_checkpoint_shape_config(args, checkpoint_config)

    args.shared_reward = bool(args.shared_reward)
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

    with (run_dir / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(vars(args), config_file, indent=2, sort_keys=True)

    env = make_env(
        n_agents=args.n_agents,
        local_ratio=args.local_ratio,
        max_cycles=args.max_cycles,
        render_mode=None,
    )

    obs_dict, infos = env.reset(seed=args.seed)
    agent_names = list(env.possible_agents)
    name_to_idx = {name: idx for idx, name in enumerate(agent_names)}
    obs_dim = env.observation_space(agent_names[0]).shape[0]
    act_dim = env.action_space(agent_names[0]).shape[0]
    total_obs_dim = len(agent_names) * obs_dim
    total_act_dim = len(agent_names) * act_dim

    agents = [
        MASACAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            total_obs_dim=total_obs_dim,
            total_act_dim=total_act_dim,
            hidden_sizes=hidden_sizes,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            alpha_lr=args.alpha_lr,
            init_alpha=args.init_alpha,
            device=device,
        )
        for _ in agent_names
    ]

    replay_buffer = ReplayBuffer(args.buffer_capacity)
    start_episode = 1
    global_step = 0
    best_eval_reward = -float("inf")

    if args.resume_from:
        checkpoint = load_checkpoint(args.resume_from, agents, device=device)
        start_episode = int(checkpoint["episode"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_eval_reward = float(checkpoint.get("best_eval_reward", best_eval_reward))
        logger.info("Resumed from checkpoint: %s", args.resume_from)

    logger.info(
        "Env: agents=%d obs_dim=%d act_dim=%d total_obs_dim=%d total_act_dim=%d hidden_sizes=%s shared_reward=%s",
        len(agent_names),
        obs_dim,
        act_dim,
        total_obs_dim,
        total_act_dim,
        hidden_sizes,
        args.shared_reward,
    )

    metrics_path = run_dir / "metrics.csv"
    fieldnames = [
        "episode",
        "train_mean_team_reward",
        "train_episode_steps",
        "replay_size",
        "actor_loss",
        "critic_loss",
        "alpha_loss",
        "mean_alpha",
        "eval_mean_team_reward",
        "eval_std_team_reward",
        "eval_mean_episode_length",
    ]

    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        writer.writeheader()

        for episode in range(start_episode, args.num_episodes + 1):
            obs_dict, infos = env.reset(seed=args.seed + episode)
            obs_full = build_full_obs_dict(obs_dict, agent_names, obs_dim)
            episode_team_return = 0.0
            episode_step_count = 0
            last_actor_loss = None
            last_critic_loss = None
            last_alpha_loss = None

            while env.agents:
                live_agents = list(env.agents)
                actions_dict = {}
                full_action_dict = {
                    name: np.zeros(act_dim, dtype=np.float32)
                    for name in agent_names
                }

                for agent_name in live_agents:
                    agent_index = name_to_idx[agent_name]
                    if global_step < args.warmup_steps:
                        action = env.action_space(agent_name).sample().astype(np.float32)
                    else:
                        action = select_action(
                            actor=agents[agent_index].actor,
                            obs_np=obs_full[agent_name],
                            device=device,
                            deterministic=False,
                        )
                    actions_dict[agent_name] = action
                    full_action_dict[agent_name] = action

                next_obs_dict, rewards, terminations, truncations, infos = env.step(actions_dict)
                next_obs_full = build_full_obs_dict(next_obs_dict, agent_names, obs_dim)

                step_reward_sum = 0.0
                step_reward_count = 0
                for agent_name in live_agents:
                    step_reward_sum += rewards[agent_name]
                    step_reward_count += 1
                step_team_reward = step_reward_sum / step_reward_count
                episode_team_return += step_team_reward
                episode_step_count += 1

                if args.shared_reward:
                    reward_array = np.full(len(agent_names), step_team_reward, dtype=np.float32)
                else:
                    reward_array = np.array(
                        [rewards.get(name, 0.0) for name in agent_names],
                        dtype=np.float32,
                    )

                obs_array = np.stack([obs_full[name] for name in agent_names], axis=0)
                action_array = np.stack([full_action_dict[name] for name in agent_names], axis=0)
                next_obs_array = np.stack([next_obs_full[name] for name in agent_names], axis=0)
                done_array = np.array(
                    [float(terminations.get(name, False) or truncations.get(name, False)) for name in agent_names],
                    dtype=np.float32,
                )

                replay_buffer.add(obs_array, action_array, reward_array, next_obs_array, done_array)

                obs_full = next_obs_full
                global_step += 1

                if len(replay_buffer) >= args.batch_size:
                    for _ in range(args.updates_per_step):
                        last_actor_loss, last_critic_loss, last_alpha_loss = masac_update(
                            agents=agents,
                            replay_buffer=replay_buffer,
                            batch_size=args.batch_size,
                            gamma=args.gamma,
                            tau=args.tau,
                            device=device,
                            max_grad_norm=args.max_grad_norm,
                        )

            mean_alpha = float(np.mean([agent.alpha.item() for agent in agents]))
            train_mean_team_reward = episode_team_return / max(1, episode_step_count)

            eval_metrics = {
                "eval_mean_team_reward": "",
                "eval_std_team_reward": "",
                "eval_mean_episode_length": "",
            }

            if episode % args.eval_every == 0:
                eval_metrics = evaluate_policy(
                    agents=agents,
                    args=args,
                    device=device,
                    agent_names=agent_names,
                    obs_dim=obs_dim,
                )
                logger.info(
                    "Episode %04d | TrainReward %.4f | EvalReward %.4f +/- %.4f | EvalLen %.2f | Replay %d | Alpha %.4f",
                    episode,
                    train_mean_team_reward,
                    eval_metrics["eval_mean_team_reward"],
                    eval_metrics["eval_std_team_reward"],
                    eval_metrics["eval_mean_episode_length"],
                    len(replay_buffer),
                    mean_alpha,
                )

                if eval_metrics["eval_mean_team_reward"] > best_eval_reward:
                    best_eval_reward = eval_metrics["eval_mean_team_reward"]
                    save_checkpoint(
                        path=checkpoint_dir / "best.pt",
                        agents=agents,
                        episode=episode,
                        global_step=global_step,
                        best_eval_reward=best_eval_reward,
                        replay_size=len(replay_buffer),
                        args=args,
                    )
            else:
                logger.info(
                    "Episode %04d | TrainReward %.4f | ActorLoss %s | CriticLoss %s | AlphaLoss %s | Alpha %.4f | Replay %d",
                    episode,
                    train_mean_team_reward,
                    "None" if last_actor_loss is None else f"{last_actor_loss:.4f}",
                    "None" if last_critic_loss is None else f"{last_critic_loss:.4f}",
                    "None" if last_alpha_loss is None else f"{last_alpha_loss:.4f}",
                    mean_alpha,
                    len(replay_buffer),
                )

            writer.writerow(
                {
                    "episode": episode,
                    "train_mean_team_reward": train_mean_team_reward,
                    "train_episode_steps": episode_step_count,
                    "replay_size": len(replay_buffer),
                    "actor_loss": "" if last_actor_loss is None else last_actor_loss,
                    "critic_loss": "" if last_critic_loss is None else last_critic_loss,
                    "alpha_loss": "" if last_alpha_loss is None else last_alpha_loss,
                    "mean_alpha": mean_alpha,
                    "eval_mean_team_reward": eval_metrics["eval_mean_team_reward"],
                    "eval_std_team_reward": eval_metrics["eval_std_team_reward"],
                    "eval_mean_episode_length": eval_metrics["eval_mean_episode_length"],
                }
            )
            metrics_file.flush()

            if episode % args.save_every == 0:
                save_checkpoint(
                    path=checkpoint_dir / "latest.pt",
                    agents=agents,
                    episode=episode,
                    global_step=global_step,
                    best_eval_reward=best_eval_reward,
                    replay_size=len(replay_buffer),
                    args=args,
                )
                save_checkpoint(
                    path=checkpoint_dir / f"episode_{episode:04d}.pt",
                    agents=agents,
                    episode=episode,
                    global_step=global_step,
                    best_eval_reward=best_eval_reward,
                    replay_size=len(replay_buffer),
                    args=args,
                )

    save_checkpoint(
        path=checkpoint_dir / "final.pt",
        agents=agents,
        episode=args.num_episodes,
        global_step=global_step,
        best_eval_reward=best_eval_reward,
        replay_size=len(replay_buffer),
        args=args,
    )
    env.close()
    logger.info("Training finished. Final checkpoint saved to %s", checkpoint_dir / "final.pt")


def run_eval_only(args):
    if not args.checkpoint_path:
        raise ValueError("--eval-only requires --checkpoint-path")

    checkpoint_config = load_checkpoint_config(args.checkpoint_path)
    args = apply_checkpoint_shape_config(args, checkpoint_config)
    args.shared_reward = bool(args.shared_reward)

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
    total_obs_dim = len(agent_names) * obs_dim
    total_act_dim = len(agent_names) * act_dim
    env.close()

    agents = [
        MASACAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            total_obs_dim=total_obs_dim,
            total_act_dim=total_act_dim,
            hidden_sizes=hidden_sizes,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            alpha_lr=args.alpha_lr,
            init_alpha=args.init_alpha,
            device=device,
        )
        for _ in agent_names
    ]
    checkpoint = load_checkpoint(args.checkpoint_path, agents, device=device)

    metrics = evaluate_policy(
        agents=agents,
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
