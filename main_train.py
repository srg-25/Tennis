"""
Use this to play and train MADDPG agents.

[1] Base Weights files used to achieve Agents scores 0.5 avg #Load trained model weights where got 0.5 in 1800 episodes itself
"""
import copy
import os.path
import pathlib
from shutil import copytree
import sys
import time
import argparse

import torch
from torch import dtype
from unityagents import UnityEnvironment
import numpy as np
from collections import OrderedDict


from Utils.logger_utils import create_logger
from Utils.os_io import copy_tree

from Config.CC_MADDPG_config import config_agent
from Agent.ddpg_env_agent import Environment_Agent, CategoricalReplayBuffers_statistics
from Utils.ddpg_utils import check_plot_average_episode_score, plot_training_sessions_history, check_gradient_loss, \
    plot_score_1_dim


# ----------------- Training/Playing -------------------

def play_env_agents(n_episodes=1, env_seed=12345,
                    local_actor_path='checkpoint_actor',
                    env=None):
    """
    Play a trained agent with environment
    :param n_episodes: (int) number episodes to play
    :param env_seed: (int) a seed to initialize environment with.
    :param local_actor_path: (str) path to an actor model weights dictionary
    :param env: Tennis environment
    :return: None
    """
    task_name = 'play_env_agents'
    task_logger = create_logger(root_dir='.', log_name=f'log_{task_name}')
    task_logger.info(f'{task_name}: -------------- Start ---------------')

    if env is None:
        env_file_name = './Tennis_Windows_x86_64/Tennis.exe'
        env = UnityEnvironment(file_name=env_file_name, seed=env_seed)
    brain_name = env.brain_names[0]
    cfg = config_agent.copy()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    task_logger.info(f'Device is {device}')

    env_agent = Environment_Agent(env, brain_name, env_seed, cfg, task_logger, device)
    average_scores = env_agent.play(n_episodes=n_episodes,
                                    local_actor_path=local_actor_path)
    average_scores_file_name = env_agent.save_scores(
        agent_type_name=type(env_agent).__name__, agent_name='game_average_score_of_episodes', scores=average_scores)
    task_logger.info(f'{task_name}: average scores of the game saved in the file \'{average_scores_file_name}\'')
    task_logger.info(f'{task_name}: -------------- End ---------------')


def train_env_agents(n_episodes, tmax, env=None, env_seed=None,
                     local_actor_path='checkpoint_actor',
                     local_critic_path='checkpoint_critic',
                     load_reply_buffer_path='replay_buffer',
                     save_dir='.'):
    """
    To train an agent with environment.
    :param n_episodes: number episodes to train by.
    :param tmax: maximum number time steps in each episode.
    :param env: Tennis environment
    :param env_seed: seed of pseudo randomness.
    :param local_actor_path: (str) path to an actor model weights dictionary
    :param local_critic_path: (str) path to an actor model weights dictionary
    :param load_reply_buffer_path: (str) path to a reply buffer sub-buffers
    :param save_dir:(str) a folder to save results in.
    :return: None
    """

    def rand_u(l=0., h=0.):
        """
        Use it to add a variance in a configuration of a float parameter
        """
        return np.random.uniform(low=l, high=h, size=None)

    def rand_int(l=-1, h=0):
        """
        Use it to add a variance in a configuration of a float parameter
        """
        return np.random.randint(low=l, high=h, size=None)


    task_name = 'train_env_agents'
    task_logger = create_logger(root_dir=save_dir, log_name=f'log_{task_name}')
    task_logger.info(f'{task_name}: -------------- Start ---------------')

    # Copy folders with python source files only
    save_code_dir  = 'code'
    copy_tree(source_dir='.', destination_dir=os.path.join(save_dir, save_code_dir),
              file_extension='.py', ignore_dirs=[save_code_dir])

    if env_seed is None:
        env_seed = 92736
    if env is None:
        env_file_name = './Tennis_Windows_x86_64/Tennis.exe'
        env = UnityEnvironment(file_name=env_file_name, seed=env_seed)
    brain_name = env.brain_names[0]
    cfg = config_agent.copy()
    cfg['config_train']['buffer_size']      = int(1e5)  # int(1e6)  # replay buffer size
    cfg['config_train']['batch_size']       = 256       # 256  # 512  # 1024   # minibatch size
    cfg['config_train']['gamma']            = [0.99, 0.99]  # 0.95  # discount factor
    cfg['config_train']['tau']              = [0.01, 0.01]  # 1e-2  # scale factor for soft update of target parameters
    cfg['config_train']['lr_actor']         = [1e-4, 1e-4]  # 1e-4  # learning rate of the actor
    cfg['config_train']['lr_critic']        = [0.0003, 0.0003]  # 3e-4  # learning rate of the critic
    cfg['config_train']['weight_decay']     = [0, 0]       # 0     # L2 weight decay
    cfg['config_train']['actor_update_frequency'] = 0.75   # 0.75  # Udacity GPT: update an actor NN weights frequency per iteration
    cfg['config_train']['critic_update_frequency'] = 1.25  # 1.25 # Udacity GPT: update an critic NN weights frequency per iteration

    cfg['config_train']['min_epoch_reward_to_collect_experiences'] = 0.11  # Collect an epoch experiments
                                                                           # if the epoch received such reward or more.
                                                                           # 0.06 Senthil[https://knowledge.udacity.com/questions/303326]

    cfg['config_train']['replay_batch_positive_rewards_part'] = 0.75  # Udacity GPT. It is the part of positive reward experiences
                                                                    # in a batch
    cfg['config_train']['swap_agents_experience']   = False         # Set True to Swap agent[0] experience
                                                                    # with agent[1] experience
    cfg['config_train']['swap_agents_probability'] = 0.5            # Set 1.0 to swap agents in each sample of
                                                                    # an experience. Set to a value from 0.0 to 1.0
                                                                    # to swap agents in samples with the probability
                                                                    # equals to this value.

    cfg['config_train']['sample_all_experiences_to_all_agents'] = False  # If True: agent[0] may receive experiences
                                                                    # from agents a[0] and a[1] at different times.
                                                                    # The same is regard agent[1].
    cfg['config_train']['save_reply_buffer']                = True  # Set True To save reply buffer
    cfg['config_train']['load_restore_reply_buffer_asis']   = False # Set True To restore reply buffer
                                                                    # with seeds and other metadata
    cfg['config_train']['replay_sub_buf_imbalance'] = 0.00001       # It is a maximum unbalance between
                                                                    # positive and negative sub-buffers.
                                                                    # Which means that a positive sample will not be added
                                                                    # to the sub_positive buffer if length of sub_positive
                                                                    # greater than
                                                                    # (positive_part + imbalance) * (len(sub_positive) + len(sub_negative) + len(sub_zero))
                                                                    # Which also means that a negative sample
                                                                    # will not be added to the sub_negative buffer
                                                                    # If len(sub_negative)  greater than
                                                                    # (negative_part +imbalance) * (len(sub_positive) + len(sub_negative) + len(sub_zero))
    cfg['config_train']['save_reply_buffer_frequency']  = n_episodes  # A reply buffer save frequency (in episodes).
                                                                    # Set it to n_episodes
                                                                    # to save once per training (session)
    cfg['config_train']['add_other_agent_rewards_part'] = 0.        # 0.25 # Another agent reward part
                                                                    # to add to this agent reward
    cfg['config_train']['use_this_agent_rewards_part'] = 1.         # 1  # 0.75  # A factor to scale this agent reward.
    cfg['config_train']['negative_rewards_scale']   = 10.0          # To scale negative ReplyBuffer rewards.
    cfg['config_train']['positive_rewards_scale']   = 1.0           # To scale positive ReplyBuffer rewards.
    cfg['config_train']['zero_rewards_to_negative'] = 0  # -0.01    # To change zero reward to a negative value
                                                                    # if another agent receive positive reward

    cfg['config_train']['number_samples_to_start_learning'] = 512   # 20000 # Number samples to collect before learning
    cfg['config_train']['number_samples_before_each_update'] = 1    # 100  # Number samples to collect before each call to learning
    cfg['config_train']['add_noise']        = [True, True]          # Set True To add random noise during training
    cfg['config_train']['noise_sampling_uniformly'] = [False, False]  # Set True To add random noise sampled uniformly.
                                                                    # Otherwise, Use Normal distribution
    cfg['config_train']['add_noise']        = [True, True]          # True  # To train with or without agents noise
    cfg['config_train']['noise_mu']         = [0., 0.]              # 0.  # noise mean
    cfg['config_train']['noise_theta']      = [0.15, 0.155]         # 0.15 # noise scale factor of (mu - noise_state)
    cfg['config_train']['noise_sigma']      = [0.44, 0.44]          # 0.44  # noise scale factor of (mu - noise_state)
    cfg['config_train']['noise_sigma_reduction'] = [0.99, 0.99]     # 0.97     # To reduce sigma during an episode
                                                                    # to encourage less noise in longer episodes.
                                                                    # from Udacity GPT,
                                                                    # c, and others

    cfg['config_train']['critic_loss']          = 'mse'             # 'mse', 'sqrt_mse' - a critic loss function
    cfg['config_train']['actor_loss']           = 'critic'          # 'critic', 'inverse_critic' - an actor loss function
    cfg['config_train']['model_name']           = 'maddpg_mlp'      # 'maddpg_mlp', 'maddpg_cnn_actor' - model name suffix

                                                                    # Values bellow are from https://knowledge.udacity.com/
    cfg['config_train']['model_fc1_units']      = 400               # Number neurones in the first hidden layer of actor and critic NN
    cfg['config_train']['model_fc2_units']      = 300               # Number neurones in the second hidden layer of actor and critic NN
    cfg['config_train']['actor_regularization'] = 'BatchNormalization'         # 'DropOut', 'BatchNormalization' Actor Regularization method
    cfg['config_train']['critic_regularization'] = 'DropOut'        # 'No', 'DropOut' 'BatchNormalization' Citic Regularization method
    cfg['config_train']['drop_out_val']         = [0.25, 0.25]      # 0.25 A percent to 'DropOut'. Udacity knowledge
    cfg['config_train']['init_target_by_local_nn'] = False          # Set True to initialize target NN models by local
                                                                    # models at the beginning of any training session.

    cfg['config_train']['agents_to_train']      = [0, 1]            # [0, 1] - An array of agent IDs to train.
                                                                    # It may be [0, 1], [1, 0], [0] or [1]

    # ------------------- Debugging ----------------------------
    cfg['config_train']['debug_save_frequency'] = 1.0               # Frequency to save debug info as part from number
                                                                    # of all training episodes. For example,
                                                                    # if there is scheduled to train during 100 episodes
                                                                    # with debug_save_frequency = 0.1,
                                                                    # then debug info saved every 100*0.1 = 10 episodes
                                                                    # and after last episode
    cfg['config_train']['debug_learning']   = True      # False     # True to accumulate and print learning debug info.
    cfg['config_train']['clip_critic_grad_norm']        = False     # True to clip critic model gradient norm
                                                                    # - It is looks like critic does not influence
                                                                    # actor gradient
    cfg['config_train']['critic_grad_clip_norm']        = 0.1       # To clipp Critic to this maximum of gradient norm
    cfg['config_train']['clip_actor_grad_norm']         = False     # True to clip actor model gradient norm
    cfg['config_train']['actor_grad_clip_norm']         = 1         # Actor is clipping to this maximum of gradient norm

    cfg['config_train']['clip_critic_loss']             = None  # [0, 100]  # Set [min_val, max_val] to clip critic loss
                                                                    # in this range. None otherwise
    cfg['config_train']['clip_actor_loss']              = None  # [-100, 100]  # Set [min_val, max_val] to clip actor loss
                                                                    # in this range. None otherwise
    results_file_name_suffix = ''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    task_logger.info(f'Device is {device}')

    env_agent = Environment_Agent(env, brain_name, env_seed, cfg, task_logger, device)
    task_logger.info(f'{task_name}: Going to train agents ...')
    episode_scores, max_episode_scores, ep_steps = env_agent.train(
        n_episodes=n_episodes, max_t=tmax, local_actor_path=local_actor_path, local_critic_path=local_critic_path,
        load_reply_buffer_path=load_reply_buffer_path, save_dir=save_dir)
    agent_name = config_agent['config_train']['model_name']
    scores_file_name = env_agent.save_scores(
        agent_type_name=type(env_agent).__name__, agent_name=agent_name + '_scores', scores=episode_scores,
        save_to_dir=save_dir)
    task_logger.info(f'{task_name}: scores saved in the file \'{scores_file_name}\'')
    average_max_scores_file_name = env_agent.save_scores(
        agent_type_name=type(env_agent).__name__, agent_name=agent_name + '_average_max_scores',
        scores=max_episode_scores, save_to_dir=save_dir)
    task_logger.info(f'{task_name}: average max scores saved in the file \'{average_max_scores_file_name}\'')
    ep_steps_file_name = env_agent.save_scores(
        agent_type_name=type(env_agent).__name__, agent_name=agent_name + '_num_episode_steps',
        scores=ep_steps, save_to_dir=save_dir)
    task_logger.info(f'{task_name}: number of episode steps saved in the file \'{ep_steps_file_name}\'')

    # ------------------ Plot results ---------------------
    if cfg['config_train']['debug_learning']:
        check_gradient_loss(dir_name=save_dir, file_name_suffix=results_file_name_suffix, model_name=agent_name)

        r_buf_stat = CategoricalReplayBuffers_statistics()
        r_buf_stat.plot_statistics(os.path.join(save_dir, CategoricalReplayBuffers_statistics.f_actor_prefix))
        r_buf_stat.plot_statistics(os.path.join(save_dir, CategoricalReplayBuffers_statistics.f_critic_prefix))

    title = f'Number_Steps_per_Episode'
    plot_score_1_dim(score_path=ep_steps_file_name, title=title, x_label='Episode #', y_label='Steps',
                     file_name_suffix='', show_figure=False, start_pos=0)
    task_logger.debug(f'{task_name}: average_max_scores_file_name=\n{average_max_scores_file_name},'
                      f'\nn_episodes={n_episodes}, results_file_name_suffix={results_file_name_suffix}')

    max_average_score_episode, max_average_score = check_plot_average_episode_score(
        average_scores_path=average_max_scores_file_name, tail_sz=n_episodes, file_name_suffix=results_file_name_suffix)
    task_logger.info(f'{task_name}: max_average_score={max_average_score:.4f}, '
                     f'max_average_score_episode={max_average_score_episode}')

    task_logger.info(f'{task_name}: -------------- End ---------------')
    return max_average_score_episode, max_average_score


def train_session(root_dir, session_call_id, start_seed=92736, n_episodes=2500, tmax=1000):
    """
    Train set of agents consequently and store all results in a root_dir sub-folder.
    The root_dir folder should contain a 'session_0' sub-folder with all data
    needed to be loaded at the first training session.
    Params:
    =====
        root_dir: (str) is a folder which contains all results of the trained agents
        session_call_id: (int) is the current training set call ID.
                    If it is 1 then the root folder should contain 'session_0' sub-folder only.
                    Otherwise, there should be session_0 and a one other at least.
        start_seed: (long) is a seed to begin the set training from. Consecutive trainings sessions might change it.
        n_episodes: (int) is a number of episodes per a training session in the set of trainings.
        tmax: (int) maximum number iteration per episode
    """
    def check_train_set_metadata_existence(episodes_file_name, scores_file_name, session_ids_file_name):
        """
        Check that these files exist.
        Return True if these files are exist.
        """
        metadata_exists = os.path.isfile(episodes_file_name) \
            and os.path.isfile(scores_file_name) \
            and os.path.isfile(session_ids_file_name)
        return metadata_exists

    def get_session_to_load(max_average_scores_set, session_ids_set, logger, is_get_best=False):
        """
        Get a session to preload a training data from.
        There are two possibilities:
            1. Get session id of a session with maximum score
            2. Get a random session ID from a range of sessions of greatest scores.
        Params:
        =======
            max_average_scores_set: (1d np.array) is a best score array of all existing sessions
            session_ids_set: (1d np.array) is session IDs of scores from 'max_average_scores_set'
            logger: a logger
            is_get_best: (bool) Return a session with maximum score if it is True.
                                Otherwise, return random ID from a predefined range of best sessions.
        """
        if is_get_best:
            candidate_set_sequential_order = np.argmax(max_average_scores_set)
        else: # get a random session with a maximum average score from a predefined range of top score sessions.
            sorted_ids = max_average_scores_set.argsort(kind='stable')[::-1]  # descending order
            select_candidates_sz = 3  # 5  # It is a maximum number of sessions to consider as a pretrained candidates.
            select_candidates_len = min(len(sorted_ids), select_candidates_sz)
            candidate_sorted_id = np.random.randint(0, select_candidates_len)
            candidate_set_sequential_order = sorted_ids[candidate_sorted_id]
        session_id = session_ids_set[candidate_set_sequential_order]
        session_max_average_score = max_average_scores_set[candidate_set_sequential_order]
        logger.info(f'Load session ID:{session_id}, with maximum average score: {session_max_average_score} '
                    f'selected by is_get_best= {is_get_best} algorithm')

        return session_id, session_max_average_score

    task_name = 'train_session'
    task_logger = create_logger(root_dir=root_dir, log_name=f'log_{task_name}')
    task_logger.info(f'{task_name}: -------------- Start ---------------')
    task_logger.info(f'root_dir=\n{root_dir}, \nsession_call_id={session_call_id}, '
                     f'start_seed={start_seed}, n_episodes={n_episodes}, tmax={tmax}')

    # Paths to save/load train set metadata: set_max_average_score_episode and set_max_average_score
    train_set_max_average_score_episodes_file_name = os.path.join(root_dir, f'train_set_max_average_score_episodes.npy')
    train_set_max_average_scores_file_name = os.path.join(root_dir, f'train_set_max_average_scores.npy')
    train_session_ids_file_name = os.path.join(root_dir, f'train_session_ids.npy')

    if session_call_id == 1:  # Start new set of trainings
        if check_train_set_metadata_existence(
                train_set_max_average_score_episodes_file_name, train_set_max_average_scores_file_name,
                train_session_ids_file_name):
            task_logger.info('Error: Meta data exists. Therefore it is not a new training session. ')
            return 1

        #  --------- Check NN models existence in session_0 to load these models if its are -------------
        db_load_dir = os.path.join(root_dir, f'session_0')
        local_actor_path = os.path.join(db_load_dir, 'checkpoint_best_actor')
        if os.path.isfile(os.path.join(db_load_dir, 'checkpoint_best_actor_0.pth')):
            task_logger.info(f'Actor model file to load:\n{local_actor_path} ')
        else:
            local_actor_path = None
            task_logger.info(f'Actor model files are not exist. Start training without it.')

        local_critic_path = os.path.join(db_load_dir, 'checkpoint_best_critic')
        if os.path.isfile(os.path.join(db_load_dir, 'checkpoint_best_critic_0.pth')):
            task_logger.info(f'Critic model files to load:\n{local_critic_path} ')
        else:
            local_critic_path = None
            task_logger.info(f'Critic model files are not exist. Start training without it.')

        load_reply_buffer_path = os.path.join(db_load_dir, 'replay_buffer')
        if os.path.isdir(load_reply_buffer_path):
            task_logger.info(f'Replay buffer folder :\n{load_reply_buffer_path} ')
        else:
            load_reply_buffer_path = None
            task_logger.info(f'Replay Buffer folder does not exists. Start training without it.')

        s_id        = 1
        env_seed    = start_seed + (s_id - 1) * 10
        np.random.seed(env_seed)    # This is the top level where we set the seed. Other levels will set seeds also.
                                    # But we should perform it here, to select 'session_to_load_id' reproducibly
                                    # (see bellow)
        db_save_dir             = os.path.join(root_dir, f'session_{s_id}')     # Create the session sub-folder.
        pathlib.Path(db_save_dir).mkdir(parents=True, exist_ok=True)

        # Train sessions
        max_average_score_episode, max_average_score = train_env_agents(
            n_episodes, tmax, env_seed=env_seed, local_actor_path=local_actor_path,
            local_critic_path=local_critic_path, load_reply_buffer_path=load_reply_buffer_path, save_dir=db_save_dir)
        task_logger.info(f'session_{session_call_id}: max_average_score={max_average_score:.4f}, '
                         f'max_average_score_episode={max_average_score_episode}')

        # Prepare the first session metadata
        set_max_average_score_episode, set_max_average_score, set_session_ids \
            = np.ones(1, dtype=int) * max_average_score_episode, \
            np.ones(1, dtype=float) * max_average_score, \
            np.ones(1, dtype=int) * s_id
    else:  # Additional sessions in the existent training set
        if not check_train_set_metadata_existence(
                train_set_max_average_score_episodes_file_name, train_set_max_average_scores_file_name,
                train_session_ids_file_name):
            task_logger.info('Error: Meta data NOT exists. Therefore it is not an existing training session. '
                             'Set session_call_id = 1 to start a new training session if it is what you wish. '
                             'Otherwise, set session_id to a number more than 1')
            return 1
        # Load training set metadata of previous sessions
        with open(train_set_max_average_score_episodes_file_name, 'rb'):
            set_max_average_score_episode = np.load(train_set_max_average_score_episodes_file_name)
        with open(train_set_max_average_scores_file_name, 'rb'):
            set_max_average_score = np.load(train_set_max_average_scores_file_name)
        with open(train_session_ids_file_name, 'rb'):
            set_session_ids = np.load(train_session_ids_file_name)

        # Use the training set metadata to find and load a previous session actor, critic models and reply buffer
        # so that this session will try to increase scores of the previous one:
        session_to_load_id, session_to_load_max_average_score = \
            get_session_to_load(set_max_average_score, set_session_ids, task_logger)
        db_load_dir = os.path.join(root_dir, f'session_{session_to_load_id}')
        local_actor_path        = os.path.join(db_load_dir, 'checkpoint_best_actor')
        local_critic_path       = os.path.join(db_load_dir, 'checkpoint_best_critic')
        # Get last RBuf because it has more samples which also are newest:
        rbuf_load_dir = os.path.join(root_dir, f'session_{np.max(set_session_ids)}')
        load_reply_buffer_path  = os.path.join(rbuf_load_dir, 'replay_buffer')
        task_logger.info(f'Replay buffer folder :\n{load_reply_buffer_path} ')

        s_id        = 1 + np.max(set_session_ids)  # Create actual session ID in increasing order
        env_seed    = start_seed + (s_id - 1) * 10
        np.random.seed(env_seed)    # This is the top level where we set the seed. Other levels will set seeds also.
                                    # But we should perform it here, to select 'session_to_load_id' reproducibly
                                    # (see bellow)
        db_save_dir             = os.path.join(root_dir, f'session_{s_id}')  # Create this session sub-folder
        pathlib.Path(db_save_dir).mkdir(parents=True, exist_ok=True)
        # Train this session
        max_average_score_episode, max_average_score = \
            train_env_agents(n_episodes, tmax, env_seed=env_seed,
                             local_actor_path=local_actor_path, local_critic_path=local_critic_path,
                             load_reply_buffer_path=load_reply_buffer_path, save_dir=db_save_dir)
        task_logger.info(f'session_{session_call_id}: max_average_score={max_average_score:.4f}, '
                         f'max_average_score_episode={max_average_score_episode}')
        # Prepare a session metadata to be saved
        set_max_average_score_episode = np.concatenate((set_max_average_score_episode,
                                                        np.ones(1, dtype=int)*max_average_score_episode))
        set_max_average_score = np.concatenate((set_max_average_score, np.ones(1, dtype=float) * max_average_score))
        set_session_ids = np.concatenate((set_session_ids, np.ones(1, dtype=int)*s_id))

    # TODO: Plot Training set results
    # Save sessions training Metadata
    try:
        with open(train_set_max_average_score_episodes_file_name, 'wb') as f:
            np.save(f, set_max_average_score_episode)
        with open(train_set_max_average_scores_file_name, 'wb') as f:
            np.save(f, set_max_average_score)
        with open(train_session_ids_file_name, 'wb') as f:
            np.save(f, set_session_ids)
    except FileNotFoundError:
        task_logger.info(f'One of a file not found: '
                         f'\n{train_set_max_average_score_episodes_file_name} '
                         f'\n{train_set_max_average_scores_file_name}'
                         f'\n{train_session_ids_file_name}'
                         )
        raise ValueError('Cannot save set training metadata files.')
    except IsADirectoryError:
        task_logger.info(f'A directory error: '
                         f'\n{train_set_max_average_score_episodes_file_name} '
                         f'\n{train_set_max_average_scores_file_name}'
                         f'\n{train_session_ids_file_name}')
        raise ValueError('Cannot save set training metadata files.')
    except:
        task_logger.info(f'Cannot save: '
                         f'\n{train_set_max_average_score_episodes_file_name} '
                         f'\n{train_set_max_average_scores_file_name}'
                         f'\n{train_session_ids_file_name}')
        raise ValueError('Cannot save set training metadata files.')
    plot_training_sessions_history(load_dir=root_dir, save_dir=root_dir, logger=task_logger, show_figure=False)
    plot_training_sessions_history(load_dir=root_dir, save_dir=os.path.join(root_dir, f'session_{s_id}'),
                                   logger=task_logger, show_figure=False)
    task_logger.info(f'{task_name}: -------------- End ---------------')
    pass


def train_session_parse_arguments():
    # root_dir, start_average_max_score=-sys.float_info.max, n_sessions=2, start_seed=92736, n_episodes=2500, tmax=1000
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='./debug/dbg_agents_set', type=str, required=True,
                        help='It is a root folder to store/load data during set training sessions.')
    parser.add_argument('--session_call_id', default=1, type=int, required=True,
                        help='1 if it is a new training set. Otherwise, an integer greater then 1')
    parser.add_argument('--n_episodes', default=2500, type=float, required=False,
                        help='A number of episodes to train per session.')
    parser.add_argument('--start_seed', default=92736, type=int, required=False,
                        help='A seed in first session.')
    args = parser.parse_args()
    print(f'args The training set args:  \n{args}')
    return args


if __name__ == '__main__':
    # ===========================================
    " --------- Play ---------------"
    # ===========================================

    # To play untrained models
    # local_actor_path    =  None
    # local_critic_path   =  None

    " ------------------ To play trained models -------------------------------"
    # db_dir = './database_reply/(6736, 6743)_7_7'
    # local_actor_path    = os.path.join(db_dir, 'checkpoint_actor')
    # # local_critic_path    = os.path.join(db_dir, 'checkpoint_critic')
    #
    # n_episodes          = 200  # 200
    # env_seed            = 563109 + 1000*(10-1) # 92736  # 12345
    # play_env_agents(n_episodes=n_episodes, env_seed=env_seed, local_actor_path=local_actor_path)
    
    # ===========================================
    " --------- Train ---------------"
    # ===========================================

    " ----------------- To train untrained models ---------------------"
    local_actor_path    = None
    local_critic_path   = None
    load_reply_buffer_path = None

    n_episodes = 25000  # 25000  # 15000  # 2500  # 200  # 60000  # 1000  # 800
    tmax = 1000  # 1000
    env_seed = 92736
    save_dir    = './debug/episode_update_RBuf/3_collect-rbuf-1_ep-25000'
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    train_env_agents(n_episodes, tmax, env_seed=env_seed,
                     local_actor_path=local_actor_path, local_critic_path=local_critic_path,
                     load_reply_buffer_path=load_reply_buffer_path, save_dir=save_dir)

    " -------------- To train pretrained models --------------------------"
    # # db_dir = './database_replay/(1816, 1835)_16_18'
    # db_dir = './train_2_agent[1]/4_agent_1'
    # local_actor_path        = os.path.join(db_dir, 'checkpoint_best_actor')
    # local_critic_path       = os.path.join(db_dir, 'checkpoint_best_critic')
    # # db_dir = './database_replay/(2987, 3017)_31_33'
    # load_reply_buffer_path  = os.path.join(db_dir, 'replay_buffer')
    
    # n_episodes = 25000  # 200  # 60000  # 1000  # 800
    # tmax = 1000
    # env_seed = 92736 # !!! Change seed by on each additional training !!!
    # save_dir    = './train_2_agent[1]/7_agent-1_sigmaReduction-0.99_aLr-1.5e-4_ep-25000'
    # train_env_agents(n_episodes, tmax, env_seed=env_seed,
                     # local_actor_path=local_actor_path, local_critic_path=local_critic_path,
                     # load_reply_buffer_path=load_reply_buffer_path, save_dir=save_dir)
    
    # # results_file_name_suffix = 'main_test'
    # # check_gradient_loss(dir_name='./', file_name_suffix=results_file_name_suffix)

    " ---------------------- To Train Set of Trainings ---------------------"
    # args = train_session_parse_arguments()
    #
    # agents_set_root         = args.root_dir
    # session_call_id         = args.session_call_id
    # start_seed              = args.start_seed
    # n_episodes              = args.n_episodes
    # tmax                    = 1000
    #
    # main_log_name = 'log_train_set'
    # main_logger = create_logger(root_dir=agents_set_root, log_name=main_log_name)
    # main_logger.info(f'args The training set args:  \n{args}')
    #
    # status = train_session(root_dir=agents_set_root, session_call_id=session_call_id,
    #                        start_seed=start_seed, n_episodes=n_episodes)
    # sys.exit(status)

    # ===========================================
    " -------------------- Checks -----------------------"
    # ===========================================

    # check_configuration_variance()
    # check_replay_database()

    pass
