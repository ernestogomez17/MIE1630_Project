import numpy as np
import math
import os
import time
from copy import deepcopy
import gymnasium as gym
import torch
from tqdm import tqdm
import safety_gymnasium

from safe_rl.policy import SAC, TD3, DDPG, SACLagrangian, DDPGLagrangian, TD3Lagrangian, CVPO
from safe_rl.util.logger import EpochLogger, setup_logger_kwargs
from safe_rl.util.run_util import load_config, setup_eval_configs
from safe_rl.util.torch_util import export_device_env_variable, seed_torch
from safe_rl.worker import OffPolicyWorker, OnPolicyWorker

def _log_metrics(logger, epoch, total_steps, time=None, verbose=True):
    logger.log_tabular('CostLimit', 1000)  # Adjust the cost limit if required
    logger.log_tabular('Epoch', epoch)
    logger.log_tabular('TotalEnvInteracts', total_steps)
    for key in logger.logger_keys:
        logger.log_tabular(key, average_only=True)
    if time is not None:
        logger.log_tabular('Time', time)
    # Dump the tabular data into the logger
    logger.dump_tabular(x_axis="TotalEnvInteracts", verbose=verbose)

def train_off_policy_agent(env_id, agent_params, training_params, model_directory, identifier):
    seed = training_params.get('seed')
    device = training_params.get('device')
    device_id = training_params.get('device_id')
    threads = training_params.get('threads')
    
    seed_torch(seed)
    torch.set_num_threads(threads)
    export_device_env_variable(device, id=device_id)
    
    env = safety_gymnasium.make(env_id)
    obs, info = env.reset()

    # Extract parameters
    gamma = agent_params['gamma']
    polyak = agent_params['tau']
    hidden_sizes = agent_params['hidden_layers']

    batch_size = training_params.get('batch_size')
    max_episode_steps = training_params.get('max_episode_steps')
    save_interval = training_params.get('plot_save_frequency')
    
    logger_kwargs = setup_logger_kwargs(identifier, seed=42, data_dir=model_directory)
    logger = EpochLogger(**logger_kwargs)

    # Agent setup based on model type
    if agent_params['model_type'] == 'DDPG':
        agent = DDPG(env, logger, gamma=gamma, polyak=polyak, hidden_sizes=hidden_sizes)
    elif agent_params['model_type'] == 'DDPGLagrangian':
        agent = DDPGLagrangian(env, logger, gamma=gamma, polyak=polyak, hidden_sizes=hidden_sizes)
    elif agent_params['model_type'] == 'TD3':
        agent = TD3(env, logger, gamma=gamma, polyak=polyak, hidden_sizes=hidden_sizes)
    elif agent_params['model_type'] == 'TD3Lagrangian':
        agent = TD3Lagrangian(env, logger, gamma=gamma, polyak=polyak, hidden_sizes=hidden_sizes)
    elif agent_params['model_type'] == 'SAC':
        agent = SAC(env, logger, gamma=gamma, polyak=polyak, hidden_sizes=hidden_sizes)
    elif agent_params['model_type'] == 'SACLagrangian':
        agent = SACLagrangian(env, logger, gamma=gamma, polyak=polyak, hidden_sizes=hidden_sizes)
    elif agent_params['model_type'] == 'CVPO':
        agent = CVPO(env, logger, gamma=gamma, polyak=polyak, hidden_sizes=hidden_sizes)
    else:
        raise ValueError("Unknown model type provided")

    worker = OffPolicyWorker(env=env, policy=agent, logger=logger, 
                             batch_size=batch_size, timeout_steps=max_episode_steps)

    epochs = training_params.get('num_epochs')

    total_steps = 0
    start_time = time.time()

    scores = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        epoch_steps = 0
        for i in tqdm(range(training_params['sample_episode_num']), desc='Collecting trajectories'):
            steps = worker.work()
            epoch_steps += steps

        train_steps = training_params['episode_rerun_num'] * epoch_steps // batch_size
        for i in tqdm(range(train_steps), desc=f'Training Epoch {epoch + 1}/{epochs}'):
            data = worker.get_sample()
            agent.learn_on_batch(data)

        total_steps += epoch_steps

        for i in range(training_params['evaluate_episode_num']):
            worker.eval()

        if hasattr(agent, "post_epoch_process"):
            agent.post_epoch_process()

        _log_metrics(logger, epoch, total_steps, time.time() - start_time)

        # Save the model periodically after epochs
        if (epoch % save_interval == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)
            agent.save_model()

    # Optionally, save final model
    agent.save_model()
