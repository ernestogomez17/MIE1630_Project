import os
import argparse
from training import train_off_policy_agent

# Setup directories and load data
folder_path_initial = '/home/c/cglee/egomez/MIE1630_Project/'
folder_path_output = '/scratch/c/cglee/egomez/MIE1630_Project/'
model_directory = os.path.join(folder_path_output, 'experiments')
os.makedirs(model_directory, exist_ok=True)

# Argument parser setup with defaults
parser = argparse.ArgumentParser(description='Run Safety-Gymnasium Experiments.')
parser.add_argument('--env_id', type=str, default='SafetyPointGoal1-v0', help='Choose the environment (from the Safety-Gymnasium Library, e.g.: SafetyPointGoal1-v0)')
parser.add_argument('--model_type', type=str, default='SAC', help='Choose: DDPG, TD3, SAC, DDPGLagrangian, TD3Lagrangian, SACLagrangian, or CVPO')
args = parser.parse_args()

env_id = args.env_id # 'SafetyPointGoal1-v0' # Change environment depending on the experiments
model_type = args.model_type

base_agent_params = {
    'tau': 0.95, 
    'gamma': 0.99,
    'model_type': model_type,
    'hidden_layers': [256, 256]
}

base_training_params = {
    'num_epochs': 25,
    'sample_episode_num': 50,
    'episode_rerun_num': 10,
    'evaluate_episode_num': 10,
    'plot_save_frequency': 1,
    'max_episode_steps': 1000,
    'batch_size': 128,
    'seed': 0,
    'device': "gpu",
    'device_id': 0,
    'threads': 2
}

# Run experiment
identifier = f"experiment_{env_id}_{model_type}"
exp_dir = os.path.join(model_directory, identifier)
os.makedirs(exp_dir, exist_ok=True)
agent_params = base_agent_params.copy()
agent_params.update({
    'chkpt_dir': exp_dir,
})
training_params = base_training_params.copy()
train_off_policy_agent(env_id, agent_params, training_params, exp_dir, identifier)
