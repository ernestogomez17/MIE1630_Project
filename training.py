# Import necessary libraries
import numpy as np
import math
import os
import time
from copy import deepcopy
import gym
import torch
from tqdm import tqdm

# Import safety-gymnasium
import safety_gymnasium

# Importing the necessary algorithms and utilities
from safe_rl.policy import SAC, TD3, DDPG, SACLagrangian, DDPGLagrangian, TD3Lagrangian, CVPO
from safe_rl.util.logger import EpochLogger, setup_logger_kwargs
from safe_rl.util.run_util import load_config, setup_eval_configs
from safe_rl.util.torch_util import export_device_env_variable, seed_torch
from safe_rl.worker import OffPolicyWorker, OnPolicyWorker

def train_agent(env_id, agent_params, training_params, model_directory, identifier):
    env = safety_gymnasium.make(env_id)

    # Extract parameters
    gamma = agent_params['gamma']
    polyak = agent_params['tau']

    num_episodes = training_params.get('num_episodes', 1000)
    batch_size = training_params.get('batch_size', 64)
    max_episode_steps = training_params.get('max_episode_steps', 1000)
    save_interval = training_params.get('save_interval', 10)

    # Setup device
    os.environ["MODEL_DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(os.environ["MODEL_DEVICE"])
    print(f"Using device: {device}")

    logger_kwargs = setup_logger_kwargs(identifier, seed=42, data_dir=model_directory)
    logger = EpochLogger(**logger_kwargs)

    # Agent setup based on model type
    if training_params['model_type'] == 'DDPG':
        agent = DDPG(env, logger, gamma=gamma, polyak=polyak)
    elif training_params['model_type'] == 'DDPGLagrangian':
        agent = DDPGLagrangian(env, logger, gamma=gamma, polyak=polyak)
    elif training_params['model_type'] == 'TD3':
        agent = TD3(env, logger, gamma=gamma, polyak=polyak)
    elif training_params['model_type'] == 'TD3Lagrangian':
        agent = TD3Lagrangian(env, logger, gamma=gamma, polyak=polyak)
    elif training_params['model_type'] == 'SAC':
        agent = SAC(env, logger, gamma=gamma, polyak=polyak)
    elif training_params['model_type'] == 'SACLagrangian':
        agent = SACLagrangian(env, logger, gamma=gamma, polyak=polyak)
    elif training_params['model_type'] == 'CVPO':
        agent = CVPO(env, logger, gamma=gamma, polyak=polyak)
    else:
        raise ValueError("Unknown model type provided")

    # Move models to the correct device (actor, critic)
    agent.actor.to(device)
    agent.critic.to(device)

    worker = OffPolicyWorker(env=env, policy=agent, logger=logger, 
                             batch_size=batch_size, timeout_steps=max_episode_steps)

    total_steps = 0

    for episode in range(num_episodes):
        # Reset environment and get initial observation
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_cost = 0
        episode_steps = 0

        while not done:
            # Convert observation to tensor and move to the correct device
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            # Debug print for observation and action shapes
            # print(f"Episode {episode}, Step {episode_steps}")
            # print(f"Observation tensor shape: {obs_tensor.shape}, dtype: {obs_tensor.dtype}")
            
            # Get action from agent (assuming agent.act() returns a tuple)
            action_tuple = agent.act(obs_tensor)
            action = action_tuple[0]  # Unpack the first element (the action)

            # Check if the action is a tensor or NumPy array and handle accordingly
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()  # Move action back to NumPy if it's a tensor

            # print(f"Action shape: {action.shape}, dtype: {action.dtype}")

            # Step environment and receive next state, reward, done, and info (with cost)
            obs_next, reward, cost, done, _, info = env.step(action)
            cost = info.get('cost', 0)  # Extract cost from info
            
            episode_reward += reward
            episode_cost += cost
            episode_steps += 1
            total_steps += 1


            agent.learn_on_batch({
                'obs': obs_tensor,
                'act': torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device),
                'rew': torch.tensor(reward, dtype=torch.float32).unsqueeze(0).to(device),
                'obs2': torch.tensor(obs_next, dtype=torch.float32).unsqueeze(0).to(device),
                'done': torch.tensor(done, dtype=torch.float32).unsqueeze(0).to(device),
                'cost': torch.tensor(cost, dtype=torch.float32).unsqueeze(0).to(device)
            })

            # Update observation
            obs = obs_next

            # End episode if max steps are reached
            if episode_steps >= max_episode_steps:
                done = True

        # Update logger steps
        logger.steps = total_steps

        # Logging at the end of the episode
        logger.store(EpRet=episode_reward, EpLen=episode_steps, EpCost=episode_cost)
        logger.log_tabular('Episode', episode)
        logger.log_tabular('EpRet', episode_reward)
        logger.log_tabular('EpLen', episode_steps)
        logger.log_tabular('EpCost', episode_cost)
        logger.log_tabular('TotalEnvInteracts', total_steps)
        logger.dump_tabular()

        # Save the model periodically
        if (episode % save_interval == 0) or (episode == num_episodes - 1):
            agent.save_model()

    # Optionally, save final model
    agent.save_model()