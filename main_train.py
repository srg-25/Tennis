"""
Use this to play and train MADDPG agents.
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
from Agent.ddpg_env_agent import Environment_Agent

from Utils.ddpg_utils import check_plot_average_episode_score, plot_training_sessions_history, check_gradient_loss


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
    cfg['config_train']['buffer_size']      = int(1e6)  # int(1e6)  # replay buffer size
    cfg['config_train']['batch_size']       = 128       # 128      # minibatch size
    cfg['config_train']['gamma']            = 0.95      # 0.91 + rand_u(0.05)      # 0.95  # discount factor
    cfg['config_train']['tau']              = 1e-2      # 8.7e-3 + rand_u(0.5e-2)  # 1e-2  # scale factor for soft update of target parameters
    cfg['config_train']['lr_actor']         = 1e-4      # 6.7e-5 + rand_u(0.5e-4)  # 1e-4  # learning rate of the actor
    cfg['config_train']['lr_critic']        = 3e-4      # 7e-4 + rand_u(0.5e-3)    # 3e-4  # learning rate of the critic
    cfg['config_train']['weight_decay']     = 0         # 0         # L2 weight decay
    cfg['config_train']['actor_update_frequency'] = 0.75  # 0.75  # Udacity GPT: update an actor NN weights frequency per iteration
    cfg['config_train']['critic_update_frequency'] = 1.25 # 1.25 # Udacity GPT: update an critic NN weights frequency per iteration

    cfg['config_train']['replay_non_zero_rewards_only'] = False     # True  # Collect experiments with
                                                                    # non zeros rewards only.
                                                                    # Otherwise, collect all experiments
    cfg['config_train']['replay_batch_positive_rewards_part'] = 0.75  # Udacity GPT. It is the part of positive reward experiences
                                                                    # in a batch
    cfg['config_train']['replay_batch_negative_rewards_part'] = 0.20  # Udacity GPT. It is the part of negative reward experiences
                                                                    # in a batch.
                                                                    # Other part contains zero reward experiences
    cfg['config_train']['sample_all_experiences_to_all_agents'] = False  # If True: agent[0] may receive experiences
                                                                    # from agents a[0] and a[1] at different times.
                                                                    # The same is regard agent[1].
    cfg['config_train']['save_reply_buffer']                = True  # Set True To save reply buffer
    cfg['config_train']['load_restore_reply_buffer_asis']   = False # Set True To restore reply buffer
                                                                    # with seeds and other metadata
    cfg['config_train']['replay_sub_buf_imbalance'] = 0.0001        # It is a maximum unbalance between
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
    cfg['config_train']['add_other_agent_rewards_part'] = 0.25      # 0.25 # Another agent reward part
                                                                    # to add to this agent reward
    cfg['config_train']['use_this_agent_rewards_part'] = 1.         # 1  # 0.75  # A factor to scale this agent reward.

    cfg['config_train']['number_samples_to_start_learning'] = 200   # 20000 # Number samples to collect before learning
    cfg['config_train']['add_noise']        = True                  # Set True To add random noise during training
    cfg['config_train']['noise_sampling_uniformly'] = False         # Set True To add random noise sampled uniformly.
                                                                    # Otherwise, Use Normal distribution
    cfg['config_train']['noise_mu']         = 0.        # 0.        # noise mean
    cfg['config_train']['noise_theta']      = 0.15      # 0.15      # noise scale factor of (mu - noise_state)
    cfg['config_train']['noise_sigma']      = 0.44      # 0.44      # noise scale factor of (mu - noise_state)
    cfg['config_train']['noise_sigma_reduction'] = 0.95  # 0.97     # To reduce sigma during an episode
                                                                    # to encourage less noise in longer episodes.
                                                                    # from Udacity GPT,
                                                                    # Jonas from https://knowledge.udacity.com/questions/65068, and others

    cfg['config_train']['critic_loss']          = 'mse'             # 'mse', 'sqrt_mse' - a critic loss function
    cfg['config_train']['actor_loss']           = 'critic'          # 'critic', 'inverse_critic' - an actor loss function
    cfg['config_train']['model_name']           = 'maddpg_mlp'      # 'maddpg_mlp', 'maddpg_cnn_actor' - model name suffix

                                                                    # Values bellow are from https://knowledge.udacity.com/
    cfg['config_train']['model_fc1_units']      = 400               # Number neurones in the first hidden layer of actor and critic NN
    cfg['config_train']['model_fc2_units']      = 300               # Number neurones in the second hidden layer of actor and critic NN
    cfg['config_train']['actor_regularization'] = 'DropOut'         # 'DropOut', 'BatchNormalization' Actor Regularization method
    cfg['config_train']['critic_regularization'] = 'DropOut'        # 'No', 'DropOut' 'BatchNormalization' Citic Regularization method
    cfg['config_train']['drop_out_val']         = 0.25              # 0.25 A percent to 'DropOut'. Udacity knowledge

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
    episode_scores, max_episode_scores = env_agent.train(
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

    # ------------------ Plot results ---------------------
    if cfg['config_train']['debug_learning']:
        check_gradient_loss(dir_name=save_dir, file_name_suffix=results_file_name_suffix, model_name=agent_name)
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
            select_candidates_sz = 5  # It is a maximum number of sessions to consider as a pretrained andidates.
            select_candidates_len = min(len(sorted_ids), select_candidates_sz)
            candidate_sorted_id = np.random.randint(0, select_candidates_len)
            candidate_set_sequential_order = sorted_ids[candidate_sorted_id]
        session_id = session_ids_set[candidate_set_sequential_order]
        session_max_average_score = max_average_scores_set[candidate_set_sequential_order]
        logger.info(f'Load session ID:{session_id}, with maximum average score: {session_max_average_score} '
                    f'selected by is_get_best= {is_get_best} algorithm')

        return session_id, session_max_average_score

    task_name = 'train_session'
    task_logger = create_logger(root_dir='.', log_name=f'log_{task_name}')
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

        s_id        = 1
        env_seed    = start_seed + (s_id - 1) * 10
        np.random.seed(env_seed)    # This is the top level where we set the seed. Other levels will set seeds also.
                                    # But we should perform it here, to select 'session_to_load_id' reproducibly
                                    # (see bellow)
        db_save_dir             = os.path.join(root_dir, f'session_{s_id}')     # Create the session sub-folder.
        pathlib.Path(db_save_dir).mkdir(parents=True, exist_ok=True)
        # Train sessions
        max_average_score_episode, max_average_score = train_env_agents(
            n_episodes, tmax, env_seed=env_seed,local_actor_path=None,
            local_critic_path=None,load_reply_buffer_path=None, save_dir=db_save_dir)
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
        load_reply_buffer_path  = os.path.join(db_load_dir, 'replay_buffer')

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


def train_session_parse_arguments(logger):
    # root_dir, start_average_max_score=-sys.float_info.max, n_sessions=2, start_seed=92736, n_episodes=2500, tmax=1000
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='./debug/dbg_agents_set', type=str, required=True,
                        help='It is a root folder to store/load data during set training sessions.')
    parser.add_argument('--session_call_id', default=1, type=int, required=True,
                        help='1 if it is a new training set. Otherwise, an integer greater then 1')
    parser.add_argument('--n_episodes', default=2500, type=float, required=False,
                        help='A number of episodes to train per session.')
    parser.add_argument('--start_seed', default=92736, type=int, required=False,
                        help='A number of episodes to train per session.')
    args = parser.parse_args()
    logger.info(f'args The training set args:  \n{args}')
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
    load_reply_buffer_path = './ReplayBuf/RBuf_pos-0.75_neg-0.2-zero-0.05_min-sz-30000_imbalance_0.0001/replay_buffer'
    # load_reply_buffer_path = './debug/actor_loss/RBuf_200(.75-.20-.05)/replay_buffer'
    # load_reply_buffer_path = None  # './replayBuf_212_sigma-0.44_Drop/replay_buffer'

    n_episodes = 500  # 2500  # 200  # 60000  # 1000  # 800
    tmax = 1000  # 1000
    env_seed = 92736
    save_dir = './debug/actor_loss/16_Drop'
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    train_env_agents(n_episodes, tmax, env_seed=env_seed,
                     local_actor_path=local_actor_path, local_critic_path=local_critic_path,
                     load_reply_buffer_path=load_reply_buffer_path, save_dir=save_dir)

    " -------------- To train pretrained models --------------------------"
    # # db_dir = './database_replay/(1816, 1835)_16_18'
    # # db_dir = './database_replay/(1302, 1316)_0-6_16'
    # # db_dir = './database_replay/(1257, 1383)_7-8'
    # # db_dir = './database_replay/(1238, 1251)_18_22'
    # # db_dir = './database_replay/(1638, 1655)_22_25'
    # db_dir = './database_replay/(2919, 2949)_25_31'
    # local_actor_path        = os.path.join(db_dir, 'checkpoint_best_actor')
    # local_critic_path       = os.path.join(db_dir, 'checkpoint_best_critic')
    # db_dir = './database_replay/(2987, 3017)_31_33'
    # load_reply_buffer_path  = os.path.join(db_dir, 'replay_buffer')
    #
    # n_episodes = 2500  # 200  # 60000  # 1000  # 800
    # tmax = 1000
    # env_seed = 92736 + 2  # 563109 + 1000*(10-1)
    # train_env_agents(n_episodes, tmax, env_seed=env_seed,
    #                  local_actor_path=local_actor_path, local_critic_path=local_critic_path,
    #                  load_reply_buffer_path=load_reply_buffer_path)
    #
    # # results_file_name_suffix = 'main_test'
    # # check_gradient_loss(dir_name='./', file_name_suffix=results_file_name_suffix)

    " ---------------------- To Train Set of Trainings ---------------------"
    # main_log_name = 'log_train_set.log'
    # main_logger = task_logger = create_logger(root_dir='.', log_name=main_log_name)
    # args = train_session_parse_arguments(main_logger)
    #
    # agents_set_root         = args.root_dir
    # session_call_id         = args.session_call_id
    # start_seed              = args.start_seed
    # n_episodes              = args.n_episodes
    # tmax                    = 1000
    # status = train_session(root_dir=agents_set_root, session_call_id=session_call_id,
    #                        start_seed=start_seed, n_episodes=n_episodes)
    # sys.exit(status)

    # ===========================================
    " -------------------- Checks -----------------------"
    # ===========================================

    # check_configuration_variance()
    # check_replay_database()

    pass
