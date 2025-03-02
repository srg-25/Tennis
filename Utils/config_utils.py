"""
Configuration Utilities
"""

import numpy as np
import torch
import random


def create_model_name(name_prefix, config, config_keys):
    """
    Concatenate 'name_prefix' with 'config' parameters like f'_param_{param}'
    :param name_prefix: prefix of a name to create
    :param config: a configuration dictionary
    :param config_keys: array like of keys in the configuration
    :return: full name
    """
    name = name_prefix
    for key in config_keys:
        name = name + '_' + key + '_' + f'{config[key]}'
    return name


def set_agent_config_seed(seed, agent_config):
    """
    Set a seed in to an agent
    :param seed: long integer
    :param agent_config: a configuration dictionary
    :return: the config with this seed
    """
    agent_config['seed'] = seed
    return agent_config


def get_agent_config_seed(agent_config):
    return agent_config['seed']


def set_global_seed_in_libs(agent_config):
    """
    set agent_config['seed'] to libraries and tools
    :param agent_config: an agent configuration dictionary
    :return: None
    """
    torch.manual_seed(agent_config['seed'])
    np.random.seed(agent_config['seed'])
    random.seed(agent_config['seed'])


# ----------------- Checks ----------------

from Config.CC_MADDPG_config import config_agent


def check_create_model_name():
    task_name = 'check_create_model_name'
    print(f'{task_name}: -------------- Start ---------------')
    name = create_model_name('ddpg_agent', config_agent['config_train'], ['gamma', 'tau', 'lr_actor', 'lr_critic'])
    print(f'name =\n{name}')
    print(f'{task_name}: -------------- End ---------------')


if __name__ == '__main__':
    check_create_model_name()
