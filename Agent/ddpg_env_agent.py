"""
MADDPG Environment and Agent interaction.
[1 ]Main parts are from the Udacity DDPG code
[2] The article Lowe at all. Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
[3] Udacity GPT, Udacity knowledge[https://knowledge.udacity.com/].
Especially: Senthil[https://knowledge.udacity.com/questions/303326] - update Reply Buffer by a good episode experiences.
"""
import os.path
import pathlib

from torch.cuda import device
from unityagents import UnityEnvironment
import random
import numpy as np
from collections import namedtuple, deque, OrderedDict
import copy
import sys
import shelve
import torch

from Utils.logger_utils import create_logger
from Utils.config_utils import set_agent_config_seed, set_global_seed_in_libs, get_agent_config_seed
from Utils.ddpg_utils import plot_score_1_dim
from Agent.ddpg_agent import Agent

# ---------------- Flow -------------------------------


class Environment_Agent:
    """
    Environment - Agent interaction flow
    This is a class which connect between environment and an environment agnostic Agent class algorithms
    """

    def __init__(self, env, brain_name, env_seed, cfg, logger, device):
        self.logger = logger
        self.env = env
        # get the default brain
        self.brain_name = brain_name
        self.logger.info(f'{type(self).__name__}.__init__(): brain_name = {self.brain_name}')

        # get the default brain
        self.brain = self.env.brains[brain_name]

        # reset the environment
        env_info = self.env.reset(train_mode=False)[brain_name]
        # number of agents
        self.num_agents = len(env_info.agents)
        self.logger.info(f'{type(self).__name__}.__init__(): Number of agents = {self.num_agents}')
        # size of each action
        self.action_size = self.brain.vector_action_space_size
        self.logger.info(f'{type(self).__name__}.__init__(): Size of each action = {self.action_size}')
        # examine the state space
        states = env_info.vector_observations
        self.state_size = states.shape[1]
        self.logger.info(f'{type(self).__name__}.__init__(): There are {states.shape[0]} observations by all agents. '
                         f'Each observes a state with length = {self.state_size}.')

        self.device = device
        self.logger.info(f'{type(self).__name__}.__init__(): device = {self.device}')

        # Set the agent seed
        cfg = set_agent_config_seed(seed=env_seed + 10*self.num_agents, agent_config=cfg)
        set_global_seed_in_libs(cfg['seed'])
        self.cfg = cfg
        self.logger.info(f'{type(self).__name__}.__init__(): configuration = \n{self.cfg}')
        self.seed = get_agent_config_seed(cfg)
        self.logger.info(f'{type(self).__name__}.__init__(): seed = {self.seed}')

        self.agents_to_train = self.cfg['config_train']['agents_to_train']
        self.agent = [Agent(state_size=self.state_size, action_size=self.action_size,
                            agent_id=agent_id, n_agents=self.num_agents,
                            cfg=self.cfg, logger=self.logger, device=self.device)
                      for agent_id in range(self.num_agents)]

        self.actor_update_frequency         = self.cfg['config_train']['actor_update_frequency']
        self.critic_update_frequency        = self.cfg['config_train']['critic_update_frequency']
        self.init_target_by_local_nn        = self.cfg['config_train']['init_target_by_local_nn']

        # To train with or without agents noise
        self.add_noise = self.cfg['config_train']['add_noise']

        # Reply buffer
        self.number_samples_to_start_learning = self.cfg['config_train']['number_samples_to_start_learning']
        self.number_samples_before_each_update = self.cfg['config_train']['number_samples_before_each_update']
        self.save_reply_buffer = self.cfg['config_train']['save_reply_buffer']
        self.save_reply_buffer_frequency = self.cfg['config_train']['save_reply_buffer_frequency']
        self.replay_buffer_path = 'replay_buffer'
        self.save_pretrained_reply_buffer = self.save_reply_buffer  # This will be set to False,
                                                                    # when the pretrained buffer saved
        self.load_restore_reply_buffer_asis = self.cfg['config_train']['load_restore_reply_buffer_asis']
        self.pretrained_reply_buffer_path = 'pretrained_replay_buffer'

        # Replay memory
        self.add_other_agent_rewards_part = self.cfg['config_train']['add_other_agent_rewards_part']
        self.use_this_agent_rewards_part  = self.cfg['config_train']['use_this_agent_rewards_part']
        self.positive_rewards_scale = cfg['config_train']['positive_rewards_scale']
        self.negative_rewards_scale = cfg['config_train']['negative_rewards_scale']
        self.zero_rewards_to_negative = cfg['config_train']['zero_rewards_to_negative']
        self.sample_all_experiences_to_all_agents = self.cfg['config_train']['sample_all_experiences_to_all_agents']
        self.buffer_size = self.cfg['config_train']['buffer_size']
        self.batch_size  = self.cfg['config_train']['batch_size']
        self.batch_positive_part    = self.cfg['config_train']['replay_batch_positive_rewards_part']
        self.swap_agents_experience = cfg['config_train']['swap_agents_experience']
        self.swap_agents_probability = cfg['config_train']['swap_agents_probability']
        self.replay_sub_buf_imbalance = cfg['config_train']['replay_sub_buf_imbalance']
        # self.memory used to accumulate experiences to train Critics and Actors
        self.memory = CategoricalReplayBuffers(
            self.buffer_size, self.batch_size, self.batch_positive_part,
            self.replay_sub_buf_imbalance,
            self.add_other_agent_rewards_part, self.use_this_agent_rewards_part,
            self.positive_rewards_scale, self.negative_rewards_scale, self.zero_rewards_to_negative,
            self.sample_all_experiences_to_all_agents, self.swap_agents_experience, self.swap_agents_probability,
            self.seed, self.device, self.logger, self.num_agents)
        # self.epoch_memory used to accumulate experiences one by one from the environment during one epoch only
        # and export it to the update memory in case
        # when the epoch received at least 'min_epoch_reward_to_collect_experiences'
        self.min_epoch_reward_to_collect_experiences = \
            self.cfg['config_train']['min_epoch_reward_to_collect_experiences']
        self.epoch_memory = CentralizedReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=1,
            add_other_agent_rewards_part=0, use_this_agent_rewards_part=1.0,
            positive_rewards_scale=1.0, negative_rewards_scale=1.0, zero_rewards_to_negative=0.0,
            sample_all_experiences_to_all_agents=False, swap_agents_experience=False, swap_agents_probability=0.,
            seed=self.seed, device=self.device, logger=self.logger, num_agents=self.num_agents)
        self.debug_save_frequency = cfg['config_train']['debug_save_frequency']

    def play(self, n_episodes=1, local_actor_path='checkpoint_actor'):
        """
        It plays a number of consecutive epochs
        :param n_episodes: (int) number episodes to play
        :param local_actor_path: (str) path to an actor model dictionary (without suffix '_id' and extension '.pth')
        :return: average scores per episode
        """
        env_info = self.env.reset(train_mode=False)[self.brain_name]  # reset the environment
        # get the current state (for each agent) ndarray shape [2,24] which is [agent_id][observation_id]
        state = env_info.vector_observations
        max_scores = []
        deque_max_len = 100
        average_scores_deque = deque(maxlen=deque_max_len)

        # Load all agents
        if local_actor_path is not None:
            self.load(policy_path=local_actor_path, critic_path=None)

        for ep in range(n_episodes):
            ep_score = np.zeros(self.num_agents)   # initialize the episode score (for each agent)
            number_longest_episode_steps, t = 0, 0
            while True:                         # Perform time-steps until the episode is done
                # get actions from states ndarray shape (2,2) hich is [agent_id][action_id]
                actions = self.act(state, add_noise=False)
                actions = np.clip(actions, -1, 1)                   # all actions between -1 and 1
                env_info = self.env.step(actions)[self.brain_name]  # send all actions to tne environment
                next_state = env_info.vector_observations           # get next state (for each agent)
                rewards = env_info.rewards                          # get reward (for each agent)
                dones = env_info.local_done                         # see if episode finished
                ep_score += rewards                                 # accumulate the episode score (for each agent)
                state = next_state                                  # roll over states to next time step
                t += 1
                if np.any(dones):                                   # exit loop if episode finished
                    if t > number_longest_episode_steps:
                        number_longest_episode_steps = t
                    if not np.all(dones):
                        self.logger.info(f'{type(self).__name__}.play(): '
                                         f'Part of agents still in play\ndones = \n{dones}')
                    break
            max_scores.append(np.max(ep_score))                     # collect agents max episode score
            average_scores_deque.append(max_scores[-1])
            self.save_scores(agent_type_name=type(self).__name__, agent_name='dbg_average_score_of_episodes',
                             scores=max_scores)                     # save scores for debug purpose.
            with np.printoptions(precision=3, suppress=True):       # log this episode scores
                self.logger.info(f'{type(self).__name__}.play(): Episode {ep+1} maximum score {max_scores[-1]:.2f}, '
                                 f'average score {np.mean(average_scores_deque):.2f} '
                                 f'and maximum score {np.max(average_scores_deque):.2f} '
                                 f'of last {min(ep+1, deque_max_len)} episodes')
        return max_scores

    def train(self, n_episodes=1000, max_t=300, print_every=100,
              local_actor_path=None, local_critic_path=None, load_reply_buffer_path=None, save_dir='.'):
        """
        It trains an agent during a predefined number episodes
        :param n_episodes: (int) number episodes to train on
        :param max_t: (int) maximum number time steps per episode (horizont)
        :param print_every: (int) print info every 'print_every' time steps of the training
        :param local_actor_path: (str) path to an actor model dictionary (without suffix '_id' and extension '.pth')
        :param local_critic_path: (str) path to a critic model dictionary (without suffix '_id' and extension '.pth')
        :param load_reply_buffer_path: (str) path to load reply buffer (without suffix and extension)
        :param save_dir: (str) A folder to save data in.
        :return: scores and maximum scores per episode
        """

        # Initialize best model storage

        model_name = self.agent[0].model_name  # First duplicate here a part of metadata from Agent.
        maddpg_mlp_name = self.agent[0].maddpg_mlp_name
        if model_name == maddpg_mlp_name:
            agent_best_models = [OrderedDict([
                ('actor_local', self.agent[i].get_actor_type()(
                    self.agent[i].actor_local.state_size, self.agent[i].actor_local.action_size,
                    fc1_units=self.agent[i].actor_local.fc1_units, fc2_units=self.agent[i].actor_local.fc2_units,
                    regularization=self.agent[i].actor_regularization, drop_out_val=self.agent[i].drop_out_val)),
                ('actor_target', self.agent[i].get_actor_type()(
                    self.agent[i].actor_target.state_size, self.agent[i].actor_target.action_size,
                    fc1_units=self.agent[i].actor_target.fc1_units, fc2_units=self.agent[i].actor_target.fc2_units,
                    regularization=self.agent[i].actor_regularization, drop_out_val=self.agent[i].drop_out_val)),
                ('critic_local', self.agent[i].get_critic_type()(
                    self.agent[i].critic_local.state_size, self.agent[i].critic_local.action_size,
                    fcs1_units=self.agent[i].critic_local.fcs1_units, fc2_units=self.agent[i].critic_local.fc2_units,
                    regularization=self.agent[i].critic_regularization, drop_out_val=self.agent[i].drop_out_val)),
                ('critic_target', self.agent[i].get_critic_type()(
                    self.agent[i].critic_target.state_size, self.agent[i].critic_target.action_size,
                    fcs1_units=self.agent[i].critic_target.fcs1_units, fc2_units=self.agent[i].critic_local.fc2_units,
                    regularization=self.agent[i].critic_regularization, drop_out_val=self.agent[i].drop_out_val))
            ]) for i in range(self.num_agents)]
        else:  # TODO: If Actor is convolutional
            assert(False)  # TODO: not implemented yet

        # Train from the beginning or from an existing models
        assert(local_actor_path is None and local_critic_path is None or
               local_actor_path is not None and local_critic_path is not None)
        if (self.init_target_by_local_nn or  # Copy params if models are not pretrained
                local_actor_path is None and local_critic_path is None):
            self.hard_update()
        if local_actor_path is not None:
            self.load(policy_path=local_actor_path, critic_path=local_critic_path)
        if load_reply_buffer_path is not None:
            as_is = self.load_restore_reply_buffer_asis
            self.memory.load(load_reply_buffer_path, restore_as_was=as_is)      # load reply buffer to continue training
                                                                                # from last saved experience
            check_categorical_buffer_force_update(load_reply_buffer_path, logger=self.logger)  # Reply Buffer Statistics

        ep_scores = []                                      # to collect scores of episodes
        ep_max_scores = []                                  # maximum episode score of two agents
        ep_max_scores_deque = deque(maxlen=print_every)     # maximum of agent episode scores "list"
                                                            # of 100 last episodes
        best_mean_last_episodes_score = 0
        dbg_ep_max_steps = 0                                # maximum number of steps per episode
        ep_steps = []                                       # steps per episode starting from the episode 1
        dbg_critic_update_statistics = CategoricalReplayBuffers_statistics()  # To calculate and accumulate critic update statistics
        dbg_actor_update_statistics  = CategoricalReplayBuffers_statistics()  # To calculate and accumulate actor update statistics

        # ------------------- Training Counters ------------------------

        i_episode = 0                                       # Episode '0' is used
                                                            # to accumulate samples BEFORE learning.
        i_iteration = 0                                     # Number of time steps
        n_next_update = 0                                   # It is counter of performed updates It is used to decide
                                                            # which network to update (critic/actor) on an nex update
        n_next_actor_update = 1                             # next actor update number
        n_next_critic_update = 1                            # next critic update number
        # n_new_samples used to accumulate a number of new samples before each learning.
        # Initialize it to the number_samples_before_each_update to perform first learning
        # right after the minimum samples to start learning has been accumulated
        n_new_samples = self.number_samples_before_each_update
        while i_episode < n_episodes:
            # Accumulate enough samples to start learning and then learn n_episodes
            # if all(np.array(self.memory.internal_buffers_length()) > self.min_samples_to_start_learn()):
            if len(self.memory) > self.min_samples_to_start_learn():
                i_episode += 1
            else:  # Reset training counters on experience accumulation stage
                i_iteration                 = 0
                n_next_update               = 0
                n_next_actor_update         = 1
                n_next_critic_update        = 1
                n_new_samples               = self.number_samples_before_each_update  # to enlarge 'n_next_update' at start of the first episode

            dbg_episode_max_reward = -sys.float_info.max                        # Initialize maximum and minimum rewards
            dbg_episode_min_reward = sys.float_info.max
            dbg_critic_episode_experiences = []                                 # Used to collect statisitcs of episode experiences
            dbg_actor_episode_experiences  = []                                 # Used to collect statisitcs of episode experiences
            # to investigate episode statistics.
            env_info = self.env.reset(train_mode=True)[self.brain_name]         # To train set train_mode=True.
            state = env_info.vector_observations                                # get first state (for each agent)
            self.reset()
            ep_score = np.zeros(self.num_agents)                                # Initialize a score of this episode
            for t in range(max_t):                                              # update overall number of iterations
                if i_episode > 0:                                               # over training episodes
                    i_iteration += 1                                            # e.g., when i_episode > 0
                action = self.act(state, add_noise=self.add_noise)              # Train with or without noise
                action = np.clip(action, -1, 1)                                 # all actions between -1 and 1
                env_info = self.env.step(action)[self.brain_name]               # send all actions to tne environment
                next_state = env_info.vector_observations                       # get next states of all parallel agents
                reward = env_info.rewards                                       # get reward (for each agent)
                done = env_info.local_done                                      # see if episode finished

                if dbg_episode_min_reward > min(reward):                        # update dbg_episode_min_reward
                    dbg_episode_min_reward = min(reward)
                if dbg_episode_max_reward < max(reward):                        # update dbg_episode_max_reward
                    dbg_episode_max_reward = max(reward)

                n_next_update, n_next_critic_update, n_next_actor_update, n_new_samples, \
                    dbg_critic_update_experiences, dbg_actor_update_experiences \
                    = self.step(state, action, np.array(reward).reshape((len(reward), 1)), next_state,
                                np.array(done).reshape((len(done), 1)),
                                n_next_update, n_next_critic_update, n_next_actor_update, n_new_samples, save_dir=save_dir)
                state = next_state
                ep_score += np.array(reward)                                    # update the score of this episode
                self.scale_agent_noise()                                        # scale an agent noise range
                                                                                # by noise_sigma_reduction
                if i_episode >= 1:                                              # Zero episode not collect statistics
                                                                                # Therefore
                                                                                # dbg_critic_update_experiences is None
                                                                                # dbg_actor_update_experiences is None
                    if dbg_critic_update_experiences is not None:
                        dbg_critic_episode_experiences.append(dbg_critic_update_experiences)
                    if dbg_actor_update_experiences is not None:
                        dbg_actor_episode_experiences.append(dbg_actor_update_experiences)
                if np.any(done):                                                # Break the episode loop if it is done
                    # Fill the "update" memory by the episode experience if the episode reward is reach enough:
                    if np.max(ep_score) > self.min_epoch_reward_to_collect_experiences:
                        n_new_samples += self.export_epoch_memory_to_update_memory()
                    else:
                        self.epoch_memory.reset()
                    if i_episode >= 1:                                          # Begin collect statistics for ep >= 1
                        dbg_critic_update_statistics.update_statistics(rbufs=dbg_critic_episode_experiences)
                        dbg_actor_update_statistics.update_statistics(rbufs=dbg_actor_episode_experiences)
                        ep_steps.append(t)
                        if dbg_ep_max_steps < t:
                            dbg_ep_max_steps = t
                    break
            if i_episode == 0:                                                  # Accumulate Replay Buffer only
                print(f'\rEp_{i_episode}: accumulated {self.memory.internal_buffers_length()} experiences', end='')
                continue
            # -------------- After each episode -------------
            ep_scores.append(ep_score)                                          # add this episode score to the list
            episode_max_score = np.max(ep_score)                                # calculate maximum between two agents
            ep_max_scores.append(episode_max_score)                             # collect agents max episode score
            ep_max_scores_deque.append(episode_max_score)                       # add the maximum score to the window
            mean_last_episodes_score = np.mean(ep_max_scores_deque)             # average score in the window
            agent_name = self.cfg['config_train']['model_name']
            print('\rEp_{} \tmax score = {:.4f} \tAvrg Max Score of last {} '
                  'episodes: {:.4f}\tEpMaxReward: {:.4f}\tEpMinReward: {:.4f}, \tEpSteps: {}, \tEpMaxSteps: {}, '
                  '\tAccumulated Samples: {}'.
                  format(i_episode, episode_max_score, len(ep_max_scores_deque), mean_last_episodes_score,
                         dbg_episode_max_reward, dbg_episode_min_reward, t, dbg_ep_max_steps,
                         self.memory.internal_buffers_length()), end='')

            # Update current best models
            if best_mean_last_episodes_score < mean_last_episodes_score:
                best_mean_last_episodes_score = mean_last_episodes_score
                for i in range(self.num_agents):
                    agent_best_models[i]['actor_local'].load_state_dict(self.agent[i].actor_local.state_dict())
                    agent_best_models[i]['actor_target'].load_state_dict(self.agent[i].actor_target.state_dict())
                    agent_best_models[i]['critic_local'].load_state_dict(self.agent[i].critic_local.state_dict())
                    agent_best_models[i]['critic_target'].load_state_dict(self.agent[i].critic_target.state_dict())
            if i_episode % print_every == 0:
                self.logger.info('\nEp_{} \tReplayBufLength = {}, '
                                 '\tmax score = {:.4f} \tAvrgMaxScore{} '
                                 'episodes: {:.4f}\n\t\t\t\t\t\tEpMaxReward: {:.4f}\tEpMinReward: {:.4f}, '
                                 '\tEpSteps: {}, \tEpMaxSteps: {}'.
                                 format(i_episode, self.memory.internal_buffers_length(),
                                        ep_max_scores_deque[-1], len(ep_max_scores_deque),
                                        mean_last_episodes_score, dbg_episode_max_reward, dbg_episode_min_reward,
                                        t, dbg_ep_max_steps))
            if self.save_reply_buffer and \
                    (i_episode == n_episodes or i_episode > 0 and i_episode % self.save_reply_buffer_frequency == 0):
                replay_buffer_path = os.path.join(save_dir, self.replay_buffer_path)
                self.logger.info(f'\nGoing to save Replay Buffer to {os.path.join(save_dir, self.replay_buffer_path)}...')
                self.memory.save(replay_buffer_path)
                self.logger.info(f'Replay Buffer saved.')
                check_categorical_buffer_force_update(replay_buffer_path,
                                                      logger=self.logger)  # Reply Buffer Statistics

        # Save scores and models one time per session e.g., when all episodes done
        self.save_scores(
            agent_type_name=type(self).__name__, agent_name=agent_name + '_dbg_scores', scores=ep_scores,
            save_to_dir=save_dir)
        self.save_scores(
            agent_type_name=type(self).__name__, agent_name=agent_name + '_dbg_max_scores',
            scores=ep_max_scores, save_to_dir=save_dir)
        self.save_scores(
            agent_type_name=type(self).__name__, agent_name=agent_name + '_dbg_num_episode_steps',
            scores=ep_steps, save_to_dir=save_dir)

        # Save NNs one time per session e.g., when all episodes done
        for i in range(self.num_agents):
            torch.save(self.agent[i].actor_local.state_dict(), os.path.join(save_dir, f'checkpoint_actor_{i}.pth'))
            torch.save(self.agent[i].critic_local.state_dict(), os.path.join(save_dir, f'checkpoint_critic_{i}.pth'))
            torch.save(self.agent[i].actor_target.state_dict(),
                       os.path.join(save_dir, f'checkpoint_actor_{i}_target.pth'))
            torch.save(self.agent[i].critic_target.state_dict(),
                       os.path.join(save_dir, f'checkpoint_critic_{i}_target.pth'))

        # Save best NN models one time per session e.g., when all episodes done
        for i in range(self.num_agents):
            torch.save(agent_best_models[i]['actor_local'].state_dict(),
                       os.path.join(save_dir, f'checkpoint_best_actor_{i}.pth'))
            torch.save(agent_best_models[i]['actor_target'].state_dict(),
                       os.path.join(save_dir, f'checkpoint_best_actor_{i}_target.pth'))
            torch.save(agent_best_models[i]['critic_local'].state_dict(),
                       os.path.join(save_dir, f'checkpoint_best_critic_{i}.pth'))
            torch.save(agent_best_models[i]['critic_target'].state_dict(),
                       os.path.join(save_dir, f'checkpoint_best_critic_{i}_target.pth'))

        # Save debug info
        # if save_debug_info and self.cfg['config_train']['debug_learning']:
        if self.cfg['config_train']['debug_learning']:
            agent_name = self.cfg['config_train']['model_name']
            for i in range(self.num_agents):
                self.save_scores(
                    agent_type_name=type(self.agent[i]).__name__, agent_name=agent_name + f'_dbg_actor_loss_{i}',
                    scores=np.array(self.agent[i].dbg_actor_loss), save_to_dir=save_dir)
                self.save_scores(
                    agent_type_name=type(self.agent[i]).__name__, agent_name=agent_name + f'_dbg_actor_gradient_norm_{i}',
                    scores=np.array(self.agent[i].dbg_actor_gradient_norm), save_to_dir=save_dir)
                self.save_scores(
                    agent_type_name=type(self.agent[i]).__name__, agent_name=agent_name + f'_dbg_critic_loss_{i}',
                    scores=np.array(self.agent[i].dbg_critic_loss), save_to_dir=save_dir)
                self.save_scores(
                    agent_type_name=type(self.agent[i]).__name__, agent_name=agent_name + f'_dbg_critic_gradient_norm_{i}',
                    scores=np.array(self.agent[i].dbg_critic_gradient_norm), save_to_dir=save_dir)

            dbg_critic_update_statistics.save_statistics(
                path_to_save=os.path.join(save_dir, CategoricalReplayBuffers_statistics.f_critic_prefix))
            dbg_actor_update_statistics.save_statistics(
                path_to_save=os.path.join(save_dir, CategoricalReplayBuffers_statistics.f_actor_prefix))

        return ep_scores, ep_max_scores, ep_steps

    def reset(self):
        """ Reset noise actually """
        for i in range(self.num_agents):
            self.agent[i].reset()

    def scale_agent_noise(self):
        """ Scale agents noise sigma """
        for i in range(self.num_agents):
            self.agent[i].noise.scale_sigma()

    def act(self, state, add_noise=True):
        """ Perform action by each agent """
                                               # It is batched
        actions = np.array([self.agent[i].act(state[i], add_noise=add_noise[i]) for i in range(self.num_agents)])
        return actions

    def step(self, state,  action, reward, next_state, done, n_next_update,
             n_next_critic_update, n_next_actor_update, n_new_samples, save_dir):
        """
        Store experience in replay memory, and use random sample from buffer to learn each agent.
        Params
        ======
            state: observations from all agents
            action: actions from all agents
            reward: rewards from all agents
            next_state: next observations from all agents
            done: is an agent done?
            n_next_update (int) : number of updates counter (begin from i_episode == 1)
            n_next_critic_update (int): next critic update number
            n_actor_update (int); next actor update number
            save_dir (str): A folder to save data in
        """
        # Store experience / reward to the Replay Buffer
        self.epoch_memory.add(state, action, reward, next_state, done)  # Add samples preliminary to the epoch-memory

        # Learn, if enough samples are available in memory
        dbg_critic_update_experiences, dbg_actor_update_experiences = None, None
        if len(self.memory) > self.min_samples_to_start_learn() \
                and n_new_samples >= self.number_samples_before_each_update:
            if n_new_samples >= self.number_samples_before_each_update:
                n_next_update += 1  # We accumulated enough new samples to perform a network update
                n_new_samples -= self.number_samples_before_each_update   # Decries n_new_samples to accumulate
                # a next set of new samples up to 'number_samples_before_each_update' at least before next learning
            if self.save_pretrained_reply_buffer:
                self.save_pretrained_reply_buffer = False  # Save on time per training
                self.memory.save(os.path.join(save_dir, self.pretrained_reply_buffer_path))
            episode_done = np.any(done)  # True if it is end of the episode
            n_next_critic_update, n_next_actor_update, dbg_critic_update_experiences, dbg_actor_update_experiences = \
                self.learn(n_next_update, n_next_critic_update, n_next_actor_update, episode_done)
        return n_next_update, n_next_critic_update, n_next_actor_update, n_new_samples, \
            dbg_critic_update_experiences, dbg_actor_update_experiences

    def min_samples_to_start_learn(self):
        """ It is a number of samples to agents start learning actually after pretrained buffer has filled."""
        return self.number_samples_to_start_learning

    def export_epoch_memory_to_update_memory(self):
        """ Move experiences from self.epoch_memory to self.memory """
        num_samples_aquiered = 0
        while len(self.epoch_memory) > 0:
            e = self.epoch_memory.memory.pop()
            state, action, reward, next_state, done = e
            num_samples_aquiered += self.memory.add(state, action, reward, next_state, done)
        return num_samples_aquiered

    @staticmethod
    def create_critic_learn_tensors(experiences, agents, device):
        """
        Used to create tensors for learn critic function which uses it to optimize critic model
        Parameters:
            experiences: (namedtuple) of a (state, action, reward, next_state, done ) samples
            agents (Agent): 2 agents of the Tennis game
            device: (str) 'cpu' or cuda0'
        """
        _, _, _, next_states_s, _ = experiences
        next_states = next_states_s.clone()
        num_agents = len(agents)

        # -------- Target actions on next states -------------
        # it is a batch of states
        target_actions_on_next_state_list = [agents[i].target_action(next_states[:, i, :])
                                             for i in range(num_agents)]
        target_actions_on_next_state = torch.from_numpy(
            np.concatenate(target_actions_on_next_state_list, axis=1).reshape(
                (target_actions_on_next_state_list[0].shape[0], -1))).float().to(device)

        return target_actions_on_next_state

    @staticmethod
    def create_actor_learn_tensors(experiences: object, agents: object, device: object):
        """
        Used to create tensors for learn function as in the MADDPG article [2] algorithm. It also solves error below:
        RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they
        have already been freed).
        Saved intermediate values of the graph are freed when you call .backward() or autograd.grad()
        Return: target_actions_on_next_state, local_action_on_current_state.
                local_action_on_current_state is None if update_actor_nn is False.
        Parameters:
            experiences: (namedtuple) of a (state, action, reward, next_state, done ) samples
            agents (Agent): 2 agents of the Tennis game
            update_actor_nn (bool): True, if an actor should be updated and local_action_on_current_state be calculated
            device: (str) 'cpu' or cuda0'
        """
        current_states_s, actions, _, _, _ = experiences
        current_states = current_states_s.clone()
        num_agents = len(agents)

        # -------- If need then Calculate Local actions on current states -------------

        # Bellow is a technical problem disscusion. But non valid algoritmicaly. See the article [2].
        # local_action_on_current_state_list = [agents[i].local_action(current_states[:, i, :])
        #                                       for i in range(num_agents)]
        # local_action_on_current_state = torch.cat(local_action_on_current_state_list, dim=1)
        # TODO: I think that problem is that the for j != i we calculate the j'-clone-tensor gradient at 'i' iteration,
        #  But the j'-clone-tensor still connected to 'j'-optimizer.
        #  Therefore, when we use it in 'i' iteration on 'i'-loss calculation
        #  it calculates gradient, which registered in 'j'-optimizer.
        #  Therefore, when there is the 'j'-iteration, the 'j'-optimizer has deleted gradient from this 'j'-clone tensor
        # TODO: Therefore if j != i then convert the tensor to numpy and then reconvert it to tensor.
        #  This will break it's connection to the optimizer J.
        local_action_on_current_state = []
        for i in range(num_agents):
            las_i = []
            for j in range(num_agents):
                a_j = actions[:, j, :] if j != i else agents[i].local_action(current_states[:, i, :])
                las_i.append(a_j)
            local_action_on_current_state.append(torch.cat(las_i, dim=1))

        return local_action_on_current_state

    def learn(self, n_next_update, n_next_critic_update, n_next_actor_update, episode_done):
        """
        This method used To call agent's actual learning procedures:
            First, this method determines by bellow conditions if critic or actor should be optimized  :
                Update(optimize) a critic weights every `critic_update_frequency` per possible update (when i_episode>=1).
                Update(optimize) an actor weights every `actor_update_frequency` per possible update (when i_episode>=1).
            Then, if an update needed:
                Sample experiences by Replay Buffer.
                Prepare (modify) experiences specifically for each agent and critic's or actor's model.
                Calls actual learning procedure method of Agent class with these experiences.
            Nota bene: Experiences itself sampled by Replay Buffer 'sample' method. This sampling has two options:
                1 - Construct an experience sample of agent[0] and agent[1]
                    from the same step and from the same agent e.g., agent[0] supplied with a sample from agent[0]
                    and agent[1] supplied with a sample from agent[1] of the same experience.
                           --->>> I use only this method currently <<<---
                2 - Mix an experience sample for each agent from all agents e.g.
                    agent[0] may receive a sample from agent[0] or agent[1]
                    agent[1] may receive a sample from agent[0] or agent[1]
                Look details in CentralizedReplayBuffer sample method.
        Parameters:
            n_next_update (int): a number of performed updates + 1  from i_episode==1.
            n_next_critic_update (int): next critic update number.
            n_next_actor_update (int): next actor update number.
            episode_done (bool): True if the 'i_episode' finished. NOT USED CURRENTLY
        Return: n_next_critic_update, n_next_actor_update
        """
        update_critic_nn = n_next_critic_update <= self.critic_update_frequency * n_next_update
        update_actor_nn = n_next_actor_update <= self.actor_update_frequency * n_next_update
        dbg_max_updates = max(max(100, self.critic_update_frequency + 1), self.actor_update_frequency + 1)  # It is used To prevent endless looping
        # Initialize experience storages to investigate its statistics
        critic_pos_neg_samples = np.array([0, 0, 0])
        dbg_critic_update_experiences = CategoricalReplayBuffers(
            buffer_size=self.batch_size*dbg_max_updates,
            batch_size=self.batch_size,
            batch_positive_part=self.batch_positive_part,
            replay_sub_buf_imbalance=self.replay_sub_buf_imbalance,
            add_other_agent_rewards_part=1.,                        # These scales equal 1 because we are not sampling
            use_this_agent_rewards_part=1.,                         # These scales equal 1 because we are not sampling
            positive_rewards_scale=1., negative_rewards_scale=1.,   # These scales equal 1 because we are not sampling,
                                                                    # adding only.
            zero_rewards_to_negative=0,                             # This equal to zero, because we are not sampling,
                                                                    # adding only.
            sample_all_experiences_to_all_agents=False,             # This  equal to False because we are not sampling
            swap_agents_experience=False,                           # This  equal to False because we are not sampling
            swap_agents_probability=0,                              # This  equal to 0 because we are not sampling
            seed=self.seed, device=self.device, logger=self.logger, num_agents=2)
        actor_pos_neg_samples = np.array([0, 0, 0])
        dbg_actor_update_experiences = CategoricalReplayBuffers(
            buffer_size=self.batch_size*dbg_max_updates,
            batch_size=self.batch_size,
            batch_positive_part=self.batch_positive_part,
            replay_sub_buf_imbalance=self.replay_sub_buf_imbalance,
            add_other_agent_rewards_part=1.,                        # These scales equal 1 because we are not sampling
            use_this_agent_rewards_part=1.,                         # These scales equal 1 because we are not sampling
            positive_rewards_scale=1., negative_rewards_scale=1.,   # These scales equal 1 because we are not sampling,
                                                                    # adding only.
            zero_rewards_to_negative=0,                             # This equal to zero, because we are not sampling,
                                                                    # adding only.
            sample_all_experiences_to_all_agents=False,             # This  equal to False because we are not sampling
            swap_agents_experience=False,                           # This  equal to False because we are not sampling
            swap_agents_probability=0,                              # This  equal to 0 because we are not sampling
            seed=self.seed, device=self.device, logger=self.logger, num_agents=2)
        while update_critic_nn or update_actor_nn:
            dbg_max_updates -= 1
            if dbg_max_updates <= 0:
                self.logger.warning(f'\n\n ------------ !!! Learning: Endless loop detected !!! ------------- \n\n')
            experiences = self.memory.sample()

            # convert all elements to numpy array
            def to_numpy(v):
                if v.type() is not np.array:
                    v = np.array(v)
                return v

            stt, act, rwrd, nxt_stt, dn = experiences
            stt, act, rwrd, nxt_stt, dn = to_numpy(stt), to_numpy(act), to_numpy(rwrd), to_numpy(nxt_stt), to_numpy(dn)
            if update_critic_nn:
                critic_pos_neg_samples += dbg_critic_update_experiences.force_to_add(stt, act, rwrd, nxt_stt, dn)
                target_actions_on_next_state = self.create_critic_learn_tensors(experiences, self.agent, self.device)

                # for i in range(self.num_agents):
                for i in self.agents_to_train:
                    self.agent[i].learn_critic(experiences, target_actions_on_next_state=target_actions_on_next_state)
                n_next_critic_update += 1  # update global training counter
            if update_actor_nn:
                actor_pos_neg_samples += dbg_actor_update_experiences.force_to_add(stt, act, rwrd, nxt_stt, dn)
                local_actions_on_current_state = self.create_actor_learn_tensors(experiences, self.agent, self.device)
                # for i in range(self.num_agents):
                for i in self.agents_to_train:
                    self.agent[i].learn_actor(experiences, local_actions_on_current_state=local_actions_on_current_state[i])
                n_next_actor_update += 1  # update global training counter
            # Update the loop end/continue expressions
            update_critic_nn = n_next_critic_update <= self.critic_update_frequency * n_next_update
            update_actor_nn = n_next_actor_update <= self.actor_update_frequency * n_next_update

        return n_next_critic_update, n_next_actor_update, dbg_critic_update_experiences, dbg_actor_update_experiences

    def hard_update(self):
        """Copy local NNs to target NNs"""
        for i in range(self.num_agents):
            self.agent[i].hard_update(self.agent[i].critic_local, self.agent[i].critic_target)
            self.agent[i].hard_update(self.agent[i].actor_local, self.agent[i].actor_target)

    def load(self, policy_path, critic_path):
        """
        Load a policy(actor) and critic  from policy_path and critic_path files.
        :param policy_path: path to a policy model file without agent ID and extension '.pth'
        :param critic_path: path to a critic model file without agent ID and extension '.pth'
        :return:
        """
        for agent_id in range(self.num_agents):
            agent_local_actor_path = policy_path + f'_{agent_id}.pth'
            agent_target_actor_path = policy_path + f'_{agent_id}_target.pth'
            if critic_path is not None:
                agent_local_critic_path = critic_path + f'_{agent_id}.pth'
                agent_target_critic_path = critic_path + f'_{agent_id}_target.pth'
            else:
                agent_local_critic_path = None
                agent_target_critic_path = None
                agent_target_actor_path = None
            self.agent[agent_id].load_models(
                local_actor_path=agent_local_actor_path, local_critic_path=agent_local_critic_path,
                target_actor_path=agent_target_actor_path, target_critic_path=agent_target_critic_path)
        return policy_path, critic_path

    @staticmethod
    def _create_model_file_name_(model_name, agent_type_name, agent_name, agent_id):
        """ create a file name to save actor or critic model"""
        model_file_name = f'{agent_type_name}.{model_name}_of_{agent_name}_{agent_id}.pt'
        return model_file_name

    @staticmethod
    def save_scores(agent_type_name, agent_name, scores, save_to_dir='.'):
        """
        Save training or playing scores by creating a file name from an agent metadata
        :param agent_type_name:
        :param agent_name:
        :param scores: scores to save
        :param save_to_dir:(str) is a folder to save scores in.
        :return:
        """
        scores_file_name = os.path.join(save_to_dir, f'{agent_type_name}.{agent_name}.npy')
        with open(scores_file_name, 'wb') as f:
            np.save(f, scores)
        return scores_file_name

    def dbg_calc_grad_norm_statistics(self, grad_norms: list, model_name: str):
        """
        Calculate gradient norm for debugging purpose.
        :param grad_norms:
        :param model_name:
        :return:
        """
        grad_norms = np.array(grad_norms)
        max_norm, min_norm, mean_norm, std_norm, median_norm = np.max(grad_norms), np.min(grad_norms), \
            np.mean(grad_norms), np.std(grad_norms), np.median(grad_norms)

        with np.printoptions(precision=3, suppress=True):
            self.logger.debug(f'{model_name}:\nmax_norm={max_norm}, min_norm={min_norm}, '
                              f'mean_norm={mean_norm}, std_norm={std_norm}, median_norm={median_norm}')


class CentralizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, add_other_agent_rewards_part, use_this_agent_rewards_part,
                 positive_rewards_scale, negative_rewards_scale, zero_rewards_to_negative,
                 sample_all_experiences_to_all_agents, swap_agents_experience, swap_agents_probability,
                 seed, device, logger, num_agents=2):
        """Initialize a CentralizedReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            add_other_agent_rewards_part (float): the other agent reward part to add to this agent reward part
            use_this_agent_rewards_part (float): a scale factor to scale this agent reward.
            positive_rewards_scale, negative_rewards_scale, zero_rewards_to_negative (float): a scale factors
            to scale sample rewards while sampling
            sample_all_experiences_to_all_agents (bool): Set to True, to send trainings samples from all agents
                                                         to each agent randomly.
            swap_agents_experience (bool): Set True to Swap agent[0] experience  with agent[1] experience
            swap_agents_probability (float): Set n [0, 1.0] to Swap agent[0] experience  with agent[1] experience
                                             with this probability.
            seed      (long): this class uses inner random generator.
                              Therefore, set different seeds for different objects to avoid "synchronous" behaviour.
            device          : 'cpu' or GPU
            logger          : an external logger
            num_agents (int): number of agents. (2 for now)
        """
        self.num_agents = num_agents
        self.logger = logger
        self.seed           = seed
        self.memory         = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size     = batch_size
        self.add_other_agent_rewards_part = add_other_agent_rewards_part
        self.use_this_agent_rewards_part = use_this_agent_rewards_part
        self.positive_rewards_scale = positive_rewards_scale
        self.negative_rewards_scale = negative_rewards_scale
        self.zero_rewards_to_negative = zero_rewards_to_negative
        self.sample_all_experiences_to_all_agents = sample_all_experiences_to_all_agents
        self.swap_agents_experience  = swap_agents_experience
        self.swap_agents_probability = swap_agents_probability
        self.experience     = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.class_random   = random.Random(seed)     # Initialise inner random generator object
        self.device         = device

    def reset(self):
        """Clear all experiences"""
        self.memory.clear()

    def add(self, state, action, reward, next_state, done, add_as_batch=False):
        """
        Add a new experience to memory.
        Return amount of samples in the experience
        """
        # Below may be used when add a one sample only.
        # It is usually the case when collect a new sample from the environment
        # But it is not so, when we collect a batch with size greater than 1 to calculate statistics of experiences.

        # # Obsolete
        # if not add_as_batch:
        #     reward = np.array(reward).reshape((len(reward), 1))  # TODO: ? reward = np.array(reward.squeeze()) ?
        #     done = np.array(done).reshape((len(done), 1))  # TODO: ? done = np.array(done.squeeze()) ?
        #     n_samples = 1
        # else:
        #     n_samples = state.shape[0]
        #
        # e = self.experience(state, action, reward, next_state, done)
        # self.memory.append(e)
        # return n_samples

        if len(state.shape) == 2:  # There does not batch dimension
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)
            n_samples = 1
        else:  # There is a batch dimension
            n_samples = state.shape[0]
            for s, a, r, ns, d in zip(state, action, reward, next_state, done):
                e = self.experience(s, a, r, ns, d)
                self.memory.append(e)
        return n_samples

    def sample(self):
        """
            This is a training experience sampling.
            Randomly sample during training a batch of experiences from memory
            with replacement if there are less than batch_size elements.
            This sampling method has two options:
                1 - Construct an experience sample of agent[0] and agent[1]
                    from the same step and from the same agent e.g., agent[0] supplied with a sample from agent[0]
                    and agent[1] supplied with a sample from agent[1] of the same step.
                    --->>> I use only this method currently <<<---
                2 - Mix an experience sample for each agent from all agents e.g.
                    agent[0] may receive a sample from agent[0] or agent[1]
                    agent[1] may receive a sample from agent[0] or agent[1]
            Note: Use experiences(*sample()) to convert return to the 'experience' named tuple.
        """
        rwp_o = self.add_other_agent_rewards_part   # this synonym to shorten name only
        rwp_t = self.use_this_agent_rewards_part    # this synonym to shorten name only
        if self.__len__() == 0:
            err_msg = f'The buffer is empty'
            self.logger.warning(err_msg)
            # raise ValueError(err_msg)
            return None, None, None, None, None
        if not self.sample_all_experiences_to_all_agents:
            if self.__len__() >= self.batch_size:
                experiences = self.class_random.sample(self.memory, k=self.batch_size)
            else:
                experiences = self.class_random.choices(self.memory, k=self.batch_size)

            states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(self.device)
            actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().to(self.device)

            # negate zero rewards of an agent if another agent has positive reward
            rewards = np.array([e.reward for e in experiences if e is not None])
            for agent_this_id, agent_other_id in [[0, 1], [1, 0]]:
                pos_this_ids = rewards[:, agent_this_id, :] > 0
                zero_other_ids = rewards[:, agent_other_id, :] == 0
                pos_zero_ids = pos_this_ids * zero_other_ids
                if rewards[pos_zero_ids[:, 0]].any():
                    rewards[pos_zero_ids[:, 0], agent_other_id, :] = self.zero_rewards_to_negative

            rewards = torch.from_numpy(np.array([[rwp_t*r[0] + rwp_o*r[1],  # combine rewards from this(0) and another(1) agent
                                                  rwp_t*r[1] + rwp_o*r[0]]  # combine rewards from this(1) and another(0) agent
                                       for r in rewards if r is not None])).float().to(self.device)

            # Scale positive and negative rewards
            positive_reward_ids = rewards > 0
            rewards[positive_reward_ids] *= self.positive_rewards_scale
            negative_reward_ids = rewards < 0
            rewards[negative_reward_ids] *= self.negative_rewards_scale

            next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(self.device)
            dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        else:
            #  It is Not implemented yet. See Debug problem bellow
            # assert False
            # Create experiences in a such a way, that each agent receives random samples from all agents
            # and not own samples only.
            if self.__len__() >= self.batch_size:
                exs1 = self.class_random.sample(self.memory, k=self.batch_size)
                exs2 = self.class_random.sample(self.memory, k=self.batch_size)
            else:
                exs1 = self.class_random.choices(self.memory, k=self.batch_size)
                exs2 = self.class_random.choices(self.memory, k=self.batch_size)
            # There are an IDs from which agent get each experience to train the agent[0]
            a1_id = self.class_random.choices(range(self.num_agents), k=self.batch_size)
            # There are an IDs from which agent get each experience to train the agent[1]
            a2_id = self.class_random.choices(range(self.num_agents), k=self.batch_size)

            # TODO: Change to select samples from agent_id axis, not from batch axis!
            #  all samples has shape = (1, 2, :).
            #  Therefore, (for example) state should be sampled like
            #  dbg_sample = [e1.state[:, a1, :], e2.state[:, a2, :]]
            for (e1, a1, e2, a2) in zip(exs1, a1_id, exs2, a2_id):
                dbg_sample = [e1.state[a1], e2.state[a2]]

            states = torch.from_numpy(np.array([[e1.state[a1], e2.state[a2]]
                                                for (e1, a1, e2, a2) in zip(exs1, a1_id, exs2, a2_id)
                                                if all((e1, a1, e2, a2)) is not None])).float().to(self.device)
            actions = torch.from_numpy(np.array([[e1.action[a1], e2.action[a2]]
                                                for (e1, a1, e2, a2) in zip(exs1, a1_id, exs2, a2_id)
                                                if all((e1, a1, e2, a2)) is not None])).float().to(self.device)
            rewards = torch.from_numpy(np.array([[rwp_t*e1.reward[a1] + rwp_o*e1.reward[1-a1],  # combine rewards from this(a1) and another(a2) agent
                                                  rwp_t*e2.reward[a2] + rwp_o*e2.reward[1-a2]]  # combine rewards from this(a2) and another(a1) agent
                                                for (e1, a1, e2, a2) in zip(exs1, a1_id, exs2, a2_id)
                                                if all((e1, a1, e2, a2)) is not None])).float().to(self.device)

            # Scale positive and negative rewards
            positive_reward_ids = rewards > 0
            rewards[positive_reward_ids] *= self.positive_rewards_scale
            negative_reward_ids = rewards < 0
            rewards[negative_reward_ids] *= self.negative_rewards_scale

            next_states = torch.from_numpy(np.array([[e1.next_state[a1], e2.next_state[a2]]
                                                    for (e1, a1, e2, a2) in zip(exs1, a1_id, exs2, a2_id)
                                                    if all((e1, a1, e2, a2)) is not None])).float().to(self.device)
            dones = torch.from_numpy(np.array([[e1.done[a1], e2.done[a2]]
                                               for (e1, a1, e2, a2) in zip(exs1, a1_id, exs2, a2_id)
                                               if all((e1, a1, e2, a2)) is not None])).float().to(self.device)

        if self.swap_agents_experience:                 # swap agent[0] and agent[1] experiences
            # Choice which samples to swap.
            swap_p = self.swap_agents_probability
            swap_agents_id = np.random.binomial(n=1, p=swap_p, size=self.batch_size).astype(bool)

            def swap_arr(arr, swap_ids):
                """Swap samples which ids in swap_ids"""
                arr_to_swap = arr[swap_ids, :, :]
                arr_to_swap[:, [0, 1], :]        = arr_to_swap[:, [1, 0], :]
                arr[swap_ids, :, :]     = arr_to_swap
                return arr

            states      = swap_arr(states, swap_agents_id)
            actions     = swap_arr(actions, swap_agents_id)
            rewards     = swap_arr(rewards, swap_agents_id)
            next_states = swap_arr(next_states, swap_agents_id)
            dones       = swap_arr(dones, swap_agents_id)

            # states[:, [0, 1], :]        = states[:, [1, 0], :]
            # actions[:, [0, 1], :]       = actions[:, [1, 0], :]
            # rewards[:, [0, 1], :]       = rewards[:, [1, 0], :]
            # next_states[:, [0, 1], :]   = next_states[:, [1, 0], :]
            # dones[:, [0, 1], :]         = dones[:, [1, 0], :]
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def internal_buffers_length(self):
        """ return the buffer length """
        return tuple(self.__len__())

    def internal_buffers_batch_size(self):
        """ Return tuple (batch size of the buffer)"""
        return tuple(self.memory.batch_size)

    def __str__(self):
        format_str = ''.join([f'\n experience[{i}]: \n{e}' for i, e in enumerate(self.memory)])
        return format_str

    def save(self, path):
        """
        Save a buffer object to a disk
        :param path: a path to a new file to save. It should be  WITHOUT extension.
        :return:
        """
        with shelve.open(path) as db:
            db['memory_unpacked']       = [e._asdict() for e in self.memory]
            db['batch_size']            = self.batch_size
            db['class_random_state']    = self.class_random.getstate()
            db['seed']                  = self.seed

    def load(self, path, restore_as_was=True):
        """
        Load a buffer object from a disk
        :param path: a path to a file to load.  It should be  WITHOUT extension
        :param restore_as_was: Set to True, if you need 'batch_size', 'class_random' state and 'memory' size
                                to be restored. Otherwise, this metadata not changed.
        :return: None
        """
        self.logger.info(f'Going to load Centralized reply buffer from the path =\n{path}')  # TODO:
        try:
            with shelve.open(path) as db:
                temp_memory         = db['memory_unpacked']
                batch_size          = db['batch_size']
                class_random_state  = db['class_random_state']
                seed                = db['seed']
        except:
            self.logger.error(f'Metadadata or File not exist\n{path}\n ')
            return
        if restore_as_was:  # Restore data and metadata
            self.class_random.setstate(class_random_state)
            self.seed       = seed
            self.batch_size = batch_size
            self.memory     = deque([self.experience(
                e['state'], e['action'], e['reward'], e['next_state'], e['done']) for e in temp_memory])
        else:  # Restore data only
            if len(temp_memory) > self.memory.maxlen:
                self.memory = deque(maxlen=len(temp_memory))
            self.memory     = deque([self.experience(
                e['state'], e['action'], e['reward'], e['next_state'], e['done']) for e in temp_memory])


class CategoricalReplayBuffers:
    """Fixed-size buffer to store experience tuples in three internal CentralizedReplayBuffer's .
    NOTE: positive and pos-neg sub-buffers divide "positive"
    """
    storage_prefix  = 'pretrained_reply_buffer' # storage file name prefix to add to a file path to save/load it.
    positive_suffix = '_positive'   # the suffix to add to a positive samples sub-buffer file path to save/load it.
    pos_neg_suffix = '_pos-neg'     # the suffix to add to a pos-neg samples sub-buffer file path to save/load it.
    negative_suffix = '_negative'   # the suffix to add to a negative samples sub-buffer file path to save/load it.

    def __init__(self, buffer_size, batch_size, batch_positive_part, replay_sub_buf_imbalance,
                 add_other_agent_rewards_part, use_this_agent_rewards_part,
                 positive_rewards_scale, negative_rewards_scale, zero_rewards_to_negative,
                 sample_all_experiences_to_all_agents, swap_agents_experience, swap_agents_probability,
                 seed, device, logger, num_agents=2):
        """Initialize a CategoricalReplayBuffers object.
        It contains three CentralizedReplayBuffer objects with positive, negative and zero reward samples.
        These sub-buffers are used to collect more positive than negative and zero samples.
        For example, 80% positive, 15% negative and 5% zero samples
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            batch_positive_part (float): for example, if it is 0.80 then 80% samples has positive rewards
                                         and 20% are negative and zero rewards in the buffer and in a batch.
            replay_sub_buf_imbalance (float):  determine, how much amount of positive or negative or zero
                                                samples may differ from predefined partition.
            add_other_agent_rewards_part (float): another agent reward part to add to this agent reward
            use_this_agent_rewards_part (float): A factor to scale this agent reward.
            positive_rewards_scale (float): To scale rewards of positive sub-buffer while sampling
            negative_rewards_scale (float): To scale rewards of negative sub-buffer while sampling
            zero_rewards_to_negative (float): a scale factor to negate zero reward of an agent during sampeing
                                                when the other agent has a positive reward
            sample_all_experiences_to_all_agents (bool): Set to True, to send trainings samples from all agents
                                                         to each agent randomly. I use False currently.
            swap_agents_experience (bool): Set True to Swap agent[0] experience  with agent[1] experience
            swap_agents_probability (float): Set in [0.0, 1.0] to Swap agent[0] experience samples  with agent[1]
                                             experience samples with this probability.
            seed      (long): this class uses inner random generator in CentralizedReplayBuffer objects.
                              Therefore, set different seeds for different objects to avoid "synchronous" behaviour.
            device          : 'cpu' or GPU
            num_agents (int): number of agents. It should be the same as in an experiences returned by 'self.sample()'
        """
        self.logger = logger
        self.batch_positive_part    = batch_positive_part
        self.batch_negative_part    = 1 - batch_positive_part
        if self.batch_negative_part <= 0.:
            self.logger.error(f'batch_negative_part (={self.batch_negative_part}) is not positive, '
                              f'because batch_positive_part = {batch_positive_part} >= 1.0')
            raise ValueError
        self.swap_agents_experience     = swap_agents_experience
        self.swap_agents_probability    = swap_agents_probability
        self.replay_sub_buf_imbalance   = replay_sub_buf_imbalance
        self.num_agents = num_agents
        buf_positive_size = int(max(100, buffer_size * self.batch_positive_part))
        buf_negative_size = int(max(100, buffer_size - buf_positive_size))
        # if buf_negative_size < 1:
        #     self.logger.error(f'Negative rewards buffer has invalid replay buffer size  = {buf_negative_size} ')
        #     raise ValueError

        batch_positive_size = int(max(10, batch_size * self.batch_positive_part))
        batch_negative_size = int(max(10, batch_size - batch_positive_size))
        # if batch_negative_size < 1:
        #     self.logger.error(f'Negative rewards buffer has invalid batch size = {batch_negative_size}')
        #     raise ValueError

        # memory_positive_rewards and memory_pos_neg_rewards considered as part of "positive" buffers.
        # Therefore, its divide between them "positive" resources. For example: Batch size, ...
        self.memory_positive_rewards = CentralizedReplayBuffer(
            buffer_size=buf_positive_size,
            batch_size=int(batch_positive_size / 2),
            add_other_agent_rewards_part=add_other_agent_rewards_part,
            use_this_agent_rewards_part=use_this_agent_rewards_part,
            positive_rewards_scale=positive_rewards_scale,
            negative_rewards_scale=negative_rewards_scale,
            zero_rewards_to_negative=zero_rewards_to_negative,
            sample_all_experiences_to_all_agents=sample_all_experiences_to_all_agents,
            swap_agents_experience=swap_agents_experience,
            swap_agents_probability=swap_agents_probability,
            seed=seed+1, device=device, logger=self.logger, num_agents=self.num_agents)
        self.memory_pos_neg_rewards = CentralizedReplayBuffer(
            buffer_size=buf_positive_size,
            batch_size=int(batch_positive_size / 2),
            add_other_agent_rewards_part=add_other_agent_rewards_part,
            use_this_agent_rewards_part=use_this_agent_rewards_part,
            positive_rewards_scale=positive_rewards_scale,
            negative_rewards_scale=negative_rewards_scale,
            zero_rewards_to_negative=0,                                 #  This buffer does not negate zero rewars
            sample_all_experiences_to_all_agents=sample_all_experiences_to_all_agents,
            swap_agents_experience= swap_agents_experience,
            swap_agents_probability=swap_agents_probability,
            seed=seed + 2, device=device, logger=self.logger, num_agents=self.num_agents)
        self.memory_negative_rewards = CentralizedReplayBuffer(
            buffer_size=buf_negative_size,
            batch_size=batch_negative_size,
            add_other_agent_rewards_part=add_other_agent_rewards_part,
            use_this_agent_rewards_part=use_this_agent_rewards_part,
            positive_rewards_scale=positive_rewards_scale,
            negative_rewards_scale=negative_rewards_scale,
            zero_rewards_to_negative=0,                                   #  This buffer does not negate zero rewars
            sample_all_experiences_to_all_agents=sample_all_experiences_to_all_agents,
            swap_agents_experience= swap_agents_experience,
            swap_agents_probability=swap_agents_experience,
            seed=seed + 3, device=device, logger=self.logger, num_agents=self.num_agents)

    @staticmethod
    def is_possible_imbalance(len_1, len_2, len_1_part, imbalance):
        """ Return True if 'len_1' has imbalance in the predefined boundaries """
        is_imbalance_ok = len_1 <= np.ceil((len_1_part + imbalance) * (len_1 + len_2))
        return is_imbalance_ok

    def add(self, state, action, reward, next_state, done, add_as_batch=False):
        """
        Add a new experience to memory
        if this experience will not destroy
        predefined possible imbalance
        between positive and negative sub-buffers.
        Return 1 if an experience added. Otherwise, return 0
        """
        # TODO: add single sample experience as batch with batch size == 1
        #  TODO: Does "deadlock" impossible?
        if any(np.array(reward) > 0):
            if reward.all():  # TODO: add imbalance check mechanism
                                           #  when there are comparable amount of samples
                                           #  relative to memory_positive_rewards number of samples.
                self.memory_pos_neg_rewards.add(state, action, reward, next_state, done, add_as_batch)
                return 1
            elif self.is_possible_imbalance(len(self.memory_positive_rewards) + 1,
                                          len(self.memory_negative_rewards),
                                          self.batch_positive_part, self.replay_sub_buf_imbalance):
                self.memory_positive_rewards.add(state, action, reward, next_state, done, add_as_batch)
                return 1
        elif any(np.array(reward) < 0):
            if self.is_possible_imbalance(len(self.memory_negative_rewards) + 1,
                                          len(self.memory_positive_rewards),
                                          self.batch_negative_part, self.replay_sub_buf_imbalance):
                self.memory_negative_rewards.add(state, action, reward, next_state, done, add_as_batch)
                return 1
        return 0

    def force_to_add(self, state, action, reward, next_state, done):
        """
        Add a batch of new experiences to memory
        Without checking possible imbalance between positive and negative sub-buffers.
        Each parameter has size = (batch_sz, num_agents, parameter_sz)
        Each parameter should be a numpy array not tensor
        Return (pos_samples, pos_neg_samples, neg_samples) amounts of positive, positive-negative and negative samples added.
        Parameters:
        ==========
        state
        action
        reward
        net_next_state
        done
        """
        # TODO: Make ability to add a batch  #  TODO: !! DEBUG !!
        pos_samples, pos_neg_samples, neg_samples = 0, 0, 0
        axis_id = 0 if len(reward.shape) == 2 else 1

        pos_neg_reward_ids = reward.all(axis=axis_id)
        positive_reward_ids = (reward > 0).any(axis=axis_id)
        positive_reward_ids[pos_neg_reward_ids] = False  # Remove pos-negative IDs

        pr_nr_i = pos_neg_reward_ids.squeeze()  # shorter name
        pr_i = positive_reward_ids.squeeze()    # shorter name
        nr_i = np.ones_like(pr_i)
        if pr_nr_i.any():
            pos_neg_samples = self.memory_pos_neg_rewards.add(state[pr_nr_i], action[pr_nr_i], reward[pr_nr_i], next_state[pr_nr_i], done[pr_nr_i], True)
        if pr_i.any():
            pos_samples = self.memory_positive_rewards.add(state[pr_i], action[pr_i], reward[pr_i], next_state[pr_i], done[pr_i], True)
        # nr_i[(pr_i == True or pr_nr_i == True)] = False
        nr_i[pr_i == True] = False
        nr_i[pr_nr_i == True] = False
        if nr_i.any():
            neg_samples = self.memory_negative_rewards.add(state[nr_i], action[nr_i], reward[nr_i], next_state[nr_i], done[nr_i], True)
        return np.array([pos_samples, pos_neg_samples, neg_samples])

    def force_to_append_from_buffer(self, rbuf):
        """ Copy data from other categorical replay buffer
            Do not check on imbalance.
        """

        def copy_from_centralized_buf(bf: CentralizedReplayBuffer):
            added_samples = 0
            for experience in bf.memory:
                s, a, r, n_s, d = experience
                added_samples += self.force_to_add(s, a, r, n_s, d)
            return added_samples

        assert type(rbuf) is CategoricalReplayBuffers
        try:
            copy_from_centralized_buf(rbuf.memory_positive_rewards)
        except AttributeError:
            self.logger.warning(f'CategoricalReplayBuffers has not rbuf.memory_positive_rewards')
        try:
            copy_from_centralized_buf(rbuf.memory_pos_neg_rewards)
        except AttributeError:
            self.logger.warning(f'CategoricalReplayBuffers has not rbuf.memory_pos_neg_rewards')
        try:
            copy_from_centralized_buf(rbuf.memory_negative_rewards)
        except AttributeError:
            self.logger.warning(f'CategoricalReplayBuffers has not rbuf.memory_negative_rewards')

    def sample(self):
        """
            Randomly sample a batch of experiences from memory
            with replacement if there are less than batch_size elements.
        """
        e_to_cat = []
        experiences_positive = self.memory_positive_rewards.sample()
        if experiences_positive[0] is not None:
            e_to_cat.append(experiences_positive)
        experiences_pos_neg  = self.memory_pos_neg_rewards.sample()
        if experiences_pos_neg[0] is not None:
            e_to_cat.append(experiences_pos_neg)
        experiences_negative = self.memory_negative_rewards.sample()
        if experiences_negative[0] is not None:
            e_to_cat.append(experiences_negative)
        if len(e_to_cat) > 1:
            batch = [torch.cat(samples) for samples in zip(*e_to_cat)]
        else:
            batch = [*e_to_cat[0]]

        # batch = [torch.cat((positive, pos_neg, negative), dim=0)
        #          for positive, pos_neg, negative in zip(experiences_positive, experiences_pos_neg, experiences_negative)]
        return tuple(batch)

    def internal_buffers_length(self):
        """ Return tuple (buffer length of positive reward samples,
                            buffer length of negative reward samples,
                            buffer length of zero reward samples)"""
        return len(self.memory_positive_rewards), len(self.memory_pos_neg_rewards), len(self.memory_negative_rewards)

    def internal_buffers_batch_size(self):
        """
        Return tuple (batch size of positive reward samples buffer, batch size of negative reward samples buffer, ...)
        """
        return self.memory_positive_rewards.batch_size, self.memory_pos_neg_rewards.batch_size, \
            self.memory_negative_rewards.batch_size

    def __len__(self):
        """
        Return the sum of current size of internal memory buffers.
        """
        return len(self.memory_positive_rewards) + len(self.memory_pos_neg_rewards) + len(self.memory_negative_rewards)

    def __str__(self):
        format_str_positive = str(self.memory_positive_rewards)
        format_str_pos_neg = str(self.memory_pos_neg_rewards)
        format_str_negative = str(self.memory_negative_rewards)
        format_str = ''.join(['\n  positive reward memory:\n' + format_str_positive,
                              '\n  pos_neg reward memory:\n'  + format_str_pos_neg,
                              '\n\nnegative reward memory:\n' + format_str_negative])
        return format_str

    def save(self, store_folder=None):
        """
        Save sub-buffers to its paths with different suffixes.
        Parameters:
            store_folder: a folder to save data in. Store in the current folder if it is None
        Return: None
        """
        if store_folder is None:
            store_folder = './'
        # See
        # https://stackoverflow.com/questions/273192/how-do-i-create-a-directory-and-any-missing-parent-directories
        # Quote
        # pathlib.Path.mkdir as used above recursively creates the directory and does not raise an exception
        # if the directory already exists.
        # If you don't need or want the parents to be created, skip the parents argument.
        pathlib.Path(store_folder).mkdir(parents=True, exist_ok=True)
        path = os.path.join(store_folder, self.storage_prefix)
        with shelve.open(path) as db:
            db['batch_positive_part']       = self.batch_positive_part
            db['batch_negative_part']       = self.batch_negative_part
            db['replay_sub_buf_imbalance']  = self.replay_sub_buf_imbalance
        self.memory_positive_rewards.save(path=path + self.positive_suffix)
        self.memory_pos_neg_rewards.save(path=path + self.pos_neg_suffix)
        self.memory_negative_rewards.save(path=path + self.negative_suffix)

    def load(self, store_folder=None, restore_as_was=True):
        """
        Load sub-buffers from its paths with different suffixes.
        Parameters:
            store_folder: (str) a path to a folder which contains stored data.
            restore_as_was: (str) restore all metadata: seed, buffer sizes, batch sizes ...
        Return: None
        """
        if store_folder is None:
            store_folder = './'
        path = os.path.join(store_folder, self.storage_prefix)
        self.logger.info(f'Going to load Categorical reply buffer from the path =\n{path}')  # TODO:
        with shelve.open(path) as db:
            batch_positive_part        = db['batch_positive_part']
            batch_negative_part        = db['batch_negative_part']
            replay_sub_buf_imbalance   = db['replay_sub_buf_imbalance']
        if restore_as_was:
            self.batch_positive_part        = batch_positive_part
            self.batch_negative_part        = batch_negative_part
            self.replay_sub_buf_imbalance   = replay_sub_buf_imbalance
        try:
            self.memory_positive_rewards.load(path=path + self.positive_suffix, restore_as_was=restore_as_was)
        except (IOError, OSError) as e:
            self.logger.error(f'File not found:\n{path + self.positive_suffix}')
        try:
            self.memory_pos_neg_rewards.load(path=path + self.pos_neg_suffix, restore_as_was=restore_as_was)
        except (IOError, OSError) as e:
            self.logger.error(f'File not found:\n{path + self.pos_neg_suffix}')
        try:
            self.memory_negative_rewards.load(path=path + self.negative_suffix, restore_as_was=restore_as_was)
        except (IOError, OSError) as e:
            self.logger.error(f'File not found:\n{path + self.negative_suffix}')
        positive_negative_len = self.internal_buffers_length()
        self.logger.info(f'\nReplay Buffer size {len(self):.3f} (Positive, Positive-Negative, Negative) '
                         f'sub-buffers size = {positive_negative_len}, \n'
                         f'positive part = {positive_negative_len[0] / len(self):.3f} '
                         f'positive-negative part = {(positive_negative_len[1]) / len(self):.3f} '
                         f'negative part = {(positive_negative_len[2]) / len(self):.3f}'
                         )


class CentralizedReplayBuffers_statistics:
    """
    Used to analyze a Centralized Replay Buffer data.
    There are two main cases:
    *** Environment Replay Buffer, which contains samples collected from environment.
    *** Training Replay Buffer which contains experiments that has been sampled from Environment Replay Buffer
        to update Critic or Actor networks during a training session.
    """

    f_suffix_reward_buf_sign_stat   = '_reward_buf_sign_stat'
    f_suffix_reward_buf_val_stat    = '_reward_buf_val_stat'

    # TODO: Remove
    # f_suffix_reward_mean_norm       = '_reward_mean_norm'
    # f_suffix_reward_mean            = '_reward_mean'

    def __init__(self):
        self.reward_buf_sign    = []  # To collect agents sign reward from CentralizedReplayBuffer
        self.reward_buf_val     = []  # To collect agents value reward from CentralizedReplayBuffer

    def accumulate_reward_sign_stat(self, rbuf: CentralizedReplayBuffer):
        """
        Accumulate amount of experiences with positive, negative and zero rewards.
        Return np.array with shape (3, 3),
            which is 3x3 2D array containing number of reward samples
            in coordinate system of [negative, zero, positive] X [negative, zero, positive] for agent[0] and agent[1].
        Where for each agent
        id = 0 means negative reward ID.
        id = 1 means zero reward ID.
        id = 2 means positive reward ID.

                      positive reward of the agent[1]
                   /\
                   |
                   |
                   |
                   |
        -----------|----------> positive reward of the agent[0]
                   |
                   |
                   |

        -----------------------------------------------------------
        """
        neg_id = 0
        zero_id = 1
        pos_id = 2
        reward_sign_stat = np.zeros((3, 3))
        for _, _, rwrd, _, _ in rbuf.memory:
            # TODO: change to work with batch too
            if len(rwrd.shape) == 3 and rwrd.shape[0] == 1:
                rwrd = rwrd[0, :, :]
            if rwrd[0] == 0 and rwrd[1] == 0:
                reward_sign_stat[zero_id, zero_id] += 1
            elif rwrd[0] == 0 and rwrd[1] > 0:
                reward_sign_stat[zero_id, pos_id] += 1
            elif rwrd[0] == 0 and rwrd[1] < 0:
                reward_sign_stat[zero_id, neg_id] += 1

            elif rwrd[0] < 0 and rwrd[1] == 0:
                reward_sign_stat[neg_id, zero_id] += 1
            elif rwrd[0] < 0 and rwrd[1] > 0:
                reward_sign_stat[neg_id, pos_id] += 1
            elif rwrd[0] < 0 and rwrd[1] < 0:
                reward_sign_stat[neg_id, neg_id] += 1

            elif rwrd[0] > 0 and rwrd[1] == 0:
                reward_sign_stat[pos_id, zero_id] += 1
            elif rwrd[0] > 0 and rwrd[1] > 0:
                reward_sign_stat[pos_id, pos_id] += 1
            elif rwrd[0] > 0 and rwrd[1] < 0:
                reward_sign_stat[pos_id, neg_id] += 1
            else:
                assert False
        # Transpose because first axis is rows and second axis is columns but should be first axis is columns ... .
        # Then reverse rows , because they in the direction from top to bottom, but should be from bottom to top
        if False:  # TODO: debug
            print(f'----------------- a = \n{reward_sign_stat}')
            print(f'----------------- a.T = \n{reward_sign_stat.transpose()}')
            print(f'----------------- flipud(a.T) = \n{np.flipud(reward_sign_stat.transpose())}')
        reward_sign_stat = np.flipud(reward_sign_stat.transpose())
        self.reward_buf_sign.append(reward_sign_stat)
        return reward_sign_stat

    def accumulate_reward_val_per_sign_stat(self, rbuf: CentralizedReplayBuffer):
        """
        Accumulate reward value of experiences with positive, negative and zero rewards.
        Return np.array with shape (3, 3, 2),
            which is 3x3 2D array containing rewards of agent[0] and agent[1].
            in coordinate system of [negative, zero, positive] X [negative, zero, positive] for agent[0] and agent[1].
        Where for each agent
        id = 0 means negative reward ID.
        id = 1 means zero reward ID.
        id = 2 means positive reward ID.

                      positive reward of the agent[1]
                   /\
                   |
                   |
                   |
                   |
        -----------|----------> positive reward of the agent[0]
                   |
                   |
                   |

        -----------------------------------------------------------
        """
        neg_id = 0
        zero_id = 1
        pos_id = 2
        reward_val_per_sign_stat = np.zeros((3, 3, 2))
        for _, _, rwrd, _, _ in rbuf.memory:
            if len(rwrd.shape) == 3 and rwrd.shape[0] == 1:
                rwrd = rwrd[0, :, :]
            if rwrd[0] == 0 and rwrd[1] == 0:
                reward_val_per_sign_stat[zero_id, zero_id] += rwrd.reshape(2)
            elif rwrd[0] == 0 and rwrd[1] > 0:
                reward_val_per_sign_stat[zero_id, pos_id] += rwrd.reshape(2)
            elif rwrd[0] == 0 and rwrd[1] < 0:
                reward_val_per_sign_stat[zero_id, neg_id] += rwrd.reshape(2)

            elif rwrd[0] < 0 and rwrd[1] == 0:
                reward_val_per_sign_stat[neg_id, zero_id] += rwrd.reshape(2)
            elif rwrd[0] < 0 and rwrd[1] > 0:
                reward_val_per_sign_stat[neg_id, pos_id] += rwrd.reshape(2)
            elif rwrd[0] < 0 and rwrd[1] < 0:
                reward_val_per_sign_stat[neg_id, neg_id] += rwrd.reshape(2)

            elif rwrd[0] > 0 and rwrd[1] == 0:
                reward_val_per_sign_stat[pos_id, zero_id] += rwrd.reshape(2)
            elif rwrd[0] > 0 and rwrd[1] > 0:
                reward_val_per_sign_stat[pos_id, pos_id] += rwrd.reshape(2)
            elif rwrd[0] > 0 and rwrd[1] < 0:
                reward_val_per_sign_stat[pos_id, neg_id] += rwrd.reshape(2)
            else:
                assert False
        # Transpose because first axis is rows and second axis is columns but should be first axis is columns ... .
        # Then reverse rows , because they in the direction from top to bottom, but should be from bottom to top
        if False:  # TODO: debug
            print(f'=======================')
            print(f'----------------- b[:, :, 0] = \n{reward_val_per_sign_stat[:, :, 0]}')
            print(f'----------------- b.T[:, :, 0] = \n{reward_val_per_sign_stat.transpose(1, 0, 2)[:, :, 0]}')
            print(f'----------------- flipud(b.T)[:, :, 0] = \n{np.flipud(reward_val_per_sign_stat.transpose(1, 0, 2))[:, :, 0]}')
            print(f'=======================')
            print(f'----------------- b[:, :, 1] = \n{reward_val_per_sign_stat[:, :, 1]}')
            print(f'----------------- b.T[:, :, 1] = \n{reward_val_per_sign_stat.transpose(1, 0, 2)[:, :, 1]}')
            print(f'----------------- flipud(b.T)[:, :, 1] = \n{np.flipud(reward_val_per_sign_stat.transpose(1, 0, 2))[:, :, 1]}')

        reward_val_per_sign_stat = np.flipud(reward_val_per_sign_stat.transpose(1, 0, 2))
        self.reward_buf_val.append(reward_val_per_sign_stat)
        return reward_val_per_sign_stat

    def save_statistics(self, path_to_save_str):
        reward_buf_sign_path = path_to_save_str + self.f_suffix_reward_buf_sign_stat + '.npy'
        with open(reward_buf_sign_path, 'wb') as f:
            np.save(f, self.reward_buf_sign)

        reward_buf_val_path = path_to_save_str + self.f_suffix_reward_buf_val_stat + '.npy'
        with open(reward_buf_val_path, 'wb') as f:
            np.save(f, self.reward_buf_val)

    def load_statistics(self, path_to_load_str):

        reward_buf_sign_path = path_to_load_str + self.f_suffix_reward_buf_sign_stat + '.npy'
        self.reward_buf_sign = np.load(reward_buf_sign_path)

        reward_buf_val_path = path_to_load_str + self.f_suffix_reward_buf_val_stat + '.npy'
        self.reward_buf_val = np.load(reward_buf_val_path)

    def plot_statistics(self, path_to_load_str: str = None):
        """
        Plot reward statistics
        There are two type of statistics:
        reward sign statistics list of 3x3 numpy arrays  (accumulate_reward_sign_stat methods for detailed description)
        reward value statistics list of 3x3 numpy arrays (accumulate_reward_val_per_sign_stat method for detailed description)

        """
        if path_to_load_str is not None:
            self.load_statistics(path_to_load_str)
        pass
        # TODO: Remove and Refactor
        # name_base = os.path.splitext(os.path.basename(path_to_load_str))[0]
        # reward_mean_norm_path = path_to_load_str + self.f_suffix_reward_mean_norm + '.npy'
        # plot_score_1_dim(score_path=reward_mean_norm_path, title=name_base + self.f_suffix_reward_mean_norm,
        #                  x_label='Update Iteration #', y_label='Reward Norm',
        #                  file_name_suffix='', show_figure=False, start_pos=0)
        #
        # reward_mean_path = path_to_load_str + self.f_suffix_reward_mean + '.npy'
        # plot_score_1_dim(score_path=reward_mean_path, title=name_base + self.f_suffix_reward_mean,
        #                  x_label='Update Iteration #', y_label='Reward',
        #                  file_name_suffix='', show_figure=False, start_pos=0)


class CategoricalReplayBuffers_statistics:
    """
    Used to analyze a Categorical Replay Buffer data.
    There are two main cases:
    *** An Environment Replay Buffer, which contains samples collected from the environment.
    *** An Update Replay Buffer which contains experiments that has been sampled from
        an Environment Replay Buffer during a training session to update Critic or Actor networks.
    """
    f_environment_prefix    = 'environment_stat'    # Statistics collected during playing with an environment.
    f_actor_prefix          = 'actor_stat'          # Statistics collected to update an Actor NN
    f_critic_prefix         = 'critic_stat'         # Statistics collected to update a Critic NN

    f_suffix_positive       = '_positive'           # Samples with positive rewards for agent[0] or agent[1]
                                                    # and zero for other
    f_suffix_pos_neg        = '_pos_neg'            # Samples with positive rewards for agent[0] or agent[1]
                                                    # and negative for other
    f_suffix_negative       = '_negative'           # Samples without positive rewards, which has a negative reward
                                                    # from agent[0] or agent[1]

    def __init__(self):
        self.positive_reward_statistics = CentralizedReplayBuffers_statistics()
        self.pos_neg_reward_statistics  = CentralizedReplayBuffers_statistics()
        self.negative_reward_statistics = CentralizedReplayBuffers_statistics()

    def accumulate_reward_sign_stat(self, rbuf: CategoricalReplayBuffers):
        """
        Accumulate amount of experiences with positive, negative and zero rewards
        for each CentralizedReplayBuffer sub-buffer.
        Return np.array with shape (3, 3),
            which is 3x3 2D array containing number of reward samples
            in coordinate system of [negative, zero, positive] X [negative, zero, positive] for agent[0] and agent[1].
        Where for each agent
        id = 0 means negative reward ID
        id = 1 means zero reward ID
        id = 2 means positive reward ID

                      positive reward of the agent[1]
                   /\
                   |
                   |
                   |
                   |
        -----------|----------> positive reward of the agent[0]
                   |
                   |
                   |

        -----------------------------------------------------------
        """
        positive_buffer_reward_sign_stat    = self.positive_reward_statistics.accumulate_reward_sign_stat(rbuf.memory_positive_rewards)
        pos_neg_buffer_reward_sign_stat     = self.pos_neg_reward_statistics.accumulate_reward_sign_stat(rbuf.memory_pos_neg_rewards)
        negative_buffer_reward_sign_stat    = self.negative_reward_statistics.accumulate_reward_sign_stat(rbuf.memory_negative_rewards)
        return positive_buffer_reward_sign_stat, pos_neg_buffer_reward_sign_stat, negative_buffer_reward_sign_stat

    def accumulate_reward_val_per_sign_stat(self, rbuf: CategoricalReplayBuffers):
        """
        Accumulate reward value experiences per positive, negative and zero rewards
        for each CentralizedReplayBuffer sub-buffer.
        Return np.array with shape (3, 3, 2),
            which is 3x3 2D array containing rewards of agent[0] and agent[1] in 3rd dimension.
            in coordinate system of [negative, zero, positive] X [negative, zero, positive] for eagh agent.
        Where for each agent
        id = 0 means negative reward ID.
        id = 1 means zero reward ID.
        id = 2 means positive reward ID.

                      positive reward value of the agent[1]
                   /\
                   |
                   |
                   |
                   |
        -----------|----------> positive reward of the agent[0]
                   |
                   |
                   |

        -----------------------------------------------------------
        """
        positive_buffer_reward_val_stat = self.positive_reward_statistics.accumulate_reward_val_per_sign_stat(rbuf.memory_positive_rewards)
        pos_neg_buffer_reward_val_stat = self.positive_reward_statistics.accumulate_reward_val_per_sign_stat(rbuf.memory_pos_neg_rewards)
        negative_buffer_reward_val_stat = self.positive_reward_statistics.accumulate_reward_val_per_sign_stat(rbuf.memory_negative_rewards)
        return positive_buffer_reward_val_stat, pos_neg_buffer_reward_val_stat, negative_buffer_reward_val_stat

    def update_statistics(self, rbufs: [CategoricalReplayBuffers]):
        """Update statistics from list of CategoricalReplayBuffers"""
        for rbuf in rbufs:
            positive_buffer_reward_sign_stat, pos_neg_buffer_reward_sign_stat, negative_buffer_reward_sign_stat \
                = self.accumulate_reward_sign_stat(rbuf)

            positive_buffer_reward_val_stat, pos_neg_buffer_reward_val_stat, negative_buffer_reward_val_stat \
                = self.accumulate_reward_val_per_sign_stat(rbuf)
        pass

    def save_statistics(self, path_to_save: str):
        self.positive_reward_statistics.save_statistics(path_to_save + self.f_suffix_positive)
        self.pos_neg_reward_statistics.save_statistics(path_to_save + self.f_suffix_pos_neg)
        self.negative_reward_statistics.save_statistics(path_to_save + self.f_suffix_negative)

    def load_statistics(self, path_to_load: str):
        self.positive_reward_statistics.load_statistics(path_to_load + self.f_suffix_positive)
        self.pos_neg_reward_statistics.load_statistics(path_to_load + self.f_suffix_pos_neg)
        self.negative_reward_statistics.load_statistics(path_to_load + self.f_suffix_negative)

    def plot_statistics(self, path_to_load: str = None):
        if path_to_load is not None:
            self.load_statistics(path_to_load=path_to_load)
        self.positive_reward_statistics.plot_statistics()
        self.pos_neg_reward_statistics.plot_statistics()
        self.negative_reward_statistics.plot_statistics()
        pass

    @staticmethod
    def print_statistics(logger, reward_sign_statistics, reward_val_per_sign_statistics, statistics_name: str):
        """Calculate statistics of a CategoricalReplayBuffers"""
        logger.info(f'\nThe {statistics_name} Replay Buffer statistics: >>>>>>>>>>>>>>>')
        logger.info(f'\nPositive Replay Buffer reward sign statistics = \n{reward_sign_statistics[0]}\n')
        logger.info(f'--------------------------------------------------')
        logger.info(
            f'\nPositive Replay Buffer reward statistics of agent[0] = \n{reward_val_per_sign_statistics[0][:, :, 0]}\n')
        logger.info(f'--------------------------------------------------')
        logger.info(
            f'\nPositive Replay Buffer reward statistics of agent[1] = \n{reward_val_per_sign_statistics[0][:, :, 1]}\n')
        logger.info(f'==================================================')
        logger.info(f'\nPositive and Negative Replay Buffer reward sign statistics = \n{reward_sign_statistics[1]}\n')
        logger.info(f'--------------------------------------------------')
        logger.info(
            f'\nPositive and Negative Replay Buffer reward statistics of agent[0] = \n{reward_val_per_sign_statistics[1][:, :, 0]}\n')
        logger.info(f'--------------------------------------------------')
        logger.info(
            f'\nPositive and Negative Replay Buffer reward statistics of agent[1] = \n{reward_val_per_sign_statistics[1][:, :, 1]}\n')
        logger.info(f'==================================================')
        logger.info(f'\nNegative Replay Buffer reward sign statistics = \n{reward_sign_statistics[2]}\n')
        logger.info(f'--------------------------------------------------')
        logger.info(
            f'\nNegative Replay Buffer reward statistics of agent[0] = \n{reward_val_per_sign_statistics[2][:, :, 0]}\n')
        logger.info(f'--------------------------------------------------')
        logger.info(
            f'\nNegative Replay Buffer reward statistics of agent[1] = \n{reward_val_per_sign_statistics[2][:, :, 1]}\n')
        return reward_sign_statistics, reward_val_per_sign_statistics


class TO_REMOVE_CategoricalReplayBuffers:
    """Fixed-size buffer to store experience tuples in three internal CentralizedReplayBuffer's ."""
    storage_prefix  = 'pretrained_reply_buffer' # storage file name prefix to add to a file path to save/load it.
    positive_suffix = '_positive'   # the suffix to add to a positive samples sub-buffer file path to save/load it.
    negative_suffix = '_negative'   # the suffix to add to a negative samples sub-buffer file path to save/load it.
    zero_suffix     = '_zero'       # the suffix to add to zero samples sub-buffer file path to save/load it.

    def __init__(self, buffer_size, batch_size, batch_positive_part, batch_negative_part, replay_sub_buf_imbalance,
                 add_other_agent_rewards_part, use_this_agent_rewards_part,
                 sample_all_experiences_to_all_agents, seed, device, logger, num_agents=2):
        """Initialize a CategoricalReplayBuffers object.
        It contains three CentralizedReplayBuffer objects with positive, negative and zero reward samples.
        These sub-buffers are used to collect more positive than negative and zero samples.
        For example, 80% positive, 15% negative and 5% zero samples
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            batch_positive_part (float): for example, if it is 0.80 then 80% samples has positive rewards
                                         and 20% are negative and zero rewards in the buffer and in a batch.
            batch_negative_part (float): for example, if it is 0.80 positive and 0.15 negative
                                        then 80% samples has positive rewards
                                         and 15% are negative and 5% zero rewards in the buffer and in a batch.
            replay_sub_buf_imbalance (float):  determine, how much amount of positive or negative or zero
                                                samples may differ from predefined partition.
            add_other_agent_rewards_part (float): another agent reward part to add to this agent reward
            use_this_agent_rewards_part (float): A factor to scale this agent reward.
            sample_all_experiences_to_all_agents (bool): Set to True, to send trainings samples from all agents
                                                         to each agent randomly. I use False currently.
            seed      (long): this class uses inner random generator in CentralizedReplayBuffer objects.
                              Therefore, set different seeds for different objects to avoid "synchronous" behaviour.
            device          : 'cpu' or GPU
            num_agents (int): number of agents. It should be the same as in an experiences returned by 'self.sample()'
        """
        self.logger = logger
        self.batch_positive_part    = batch_positive_part
        self.batch_negative_part    = 1 - batch_negative_part
        self.batch_zero_part        = 1 - (batch_positive_part + batch_negative_part)  # calculate zero part
        if self.batch_zero_part <= 0.:
            self.logger.error(f'batch_zero_part (={self.batch_zero_part}) is negative, '
                              f'because batch_positive_part + batch_negative_part = '
                              f'{batch_positive_part} + {batch_negative_part} >= 1.0')
            raise ValueError
        self.replay_sub_buf_imbalance = replay_sub_buf_imbalance
        self.num_agents = num_agents
        buf_positive_size = int(buffer_size * self.batch_positive_part)
        buf_negative_size = int(buffer_size * self.batch_negative_part)
        buf_zero_size = int(buffer_size - buf_positive_size - buf_negative_size)
        if buf_zero_size < 1:
            self.logger.error(f'Zero rewards has negative replay buffer size, '
                              f'because buf_positive_size = {buf_positive_size} '
                              f'and buf_negative_size = {buf_negative_size} fill full reply buffer size = {buffer_size} ')
            raise ValueError

        batch_positive_size = int(batch_size * self.batch_positive_part)
        batch_negative_size = int(batch_size * self.batch_negative_part)
        batch_zero_size = int(batch_size - batch_positive_size - batch_negative_size)
        if batch_zero_size < 1:
            self.logger.error(f'Zero rewards has negative replay batch size, '
                              f'because batch_positive_size = {batch_positive_size} '
                              f'and batch_negative_size = {batch_negative_size} fill full reply batch size = {buffer_size} ')
            raise ValueError

        self.memory_positive_rewards = CentralizedReplayBuffer(buf_positive_size, batch_positive_size,
                                                               add_other_agent_rewards_part, use_this_agent_rewards_part,
                                                               sample_all_experiences_to_all_agents,
                                                               seed+1, device, self.logger, self.num_agents)
        self.memory_negative_rewards = CentralizedReplayBuffer(buf_negative_size, batch_negative_size,
                                                               add_other_agent_rewards_part, use_this_agent_rewards_part,
                                                               sample_all_experiences_to_all_agents,
                                                               seed+2, device, self.logger, self.num_agents)
        self.memory_zero_rewards = CentralizedReplayBuffer(buf_zero_size, batch_zero_size ,
                                                               add_other_agent_rewards_part, use_this_agent_rewards_part,
                                                               sample_all_experiences_to_all_agents,
                                                               seed+3, device, self.logger, self.num_agents)

    @staticmethod
    def is_possible_imbalance(len_1, len_2, len_1_part, imbalance):
        """ Return True if 'len_1' has imbalance in the predefined boundaries """
        is_imbalance_ok = len_1 <= np.ceil((len_1_part + imbalance) * (len_1 + len_2))
        return is_imbalance_ok

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory
        if this experience will not destroy
        predefined possible imbalance
        between positive and negative sub-buffers.
        Return 1 if an experience added. Otherwise, return 0
        """

        #  TODO: Does "deadlock" impossible?
        if any(np.array(reward) > 0):
            if self.is_possible_imbalance(len(self.memory_positive_rewards) + 1,
                                          len(self.memory_negative_rewards) + len(self.memory_zero_rewards),
                                          self.batch_positive_part, self.replay_sub_buf_imbalance):
                self.memory_positive_rewards.add(state, action, reward, next_state, done)
                return 1
        elif any(np.array(reward) < 0):
            if self.is_possible_imbalance(len(self.memory_negative_rewards) + 1,
                                          len(self.memory_positive_rewards) + len(self.memory_zero_rewards),
                                          self.batch_negative_part, self.replay_sub_buf_imbalance):
                self.memory_negative_rewards.add(state, action, reward, next_state, done)
                return 1
        else:  # zero rewards
            if self.is_possible_imbalance(len(self.memory_zero_rewards) + 1,
                                          len(self.memory_positive_rewards) + len(self.memory_negative_rewards),
                                          self.batch_zero_part, self.replay_sub_buf_imbalance):
                self.memory_zero_rewards.add(state, action, reward, next_state, done)
                return 1
        return 0

    def sample(self):
        """
            Randomly sample a batch of experiences from memory
            with replacement if there are less than batch_size elements.
        """
        experiences_positive = self.memory_positive_rewards.sample()
        experiences_negative = self.memory_negative_rewards.sample()
        experiences_zero     = self.memory_zero_rewards.sample()
        batch = [torch.cat((positive, negative, zero), dim=0)
                 for positive, negative, zero in zip(experiences_positive, experiences_negative, experiences_zero)]
        return tuple(batch)

    def internal_buffers_length(self):
        """ Return tuple (buffer length of positive reward samples,
                            buffer length of negative reward samples,
                            buffer length of zero reward samples)"""
        return len(self.memory_positive_rewards), len(self.memory_negative_rewards), len(self.memory_zero_rewards)

    def internal_buffers_batch_size(self):
        """ Return tuple (batch size of positive reward samples buffer, batch size of negative reward samples buffer, ...)"""
        return self.memory_positive_rewards.batch_size, self.memory_negative_rewards.batch_size, self.memory_zero_rewards.batch_size

    def __len__(self):
        """
        Return the sum of current size of internal memory buffers.
        """
        return len(self.memory_positive_rewards) + len(self.memory_negative_rewards) + len(self.memory_zero_rewards)

    def __str__(self):
        format_str_positive = str(self.memory_positive_rewards)
        format_str_negative = str(self.memory_negative_rewards)
        format_str_zero     = str(self.memory_zero_rewards)
        format_str = ''.join(['\n  positive reward memory:\n' + format_str_positive,
                              '\n\nnegative reward memory:\n' + format_str_negative,
                              '\n\n    zero reward memory:\n' + format_str_zero])
        return format_str

    def save(self, store_folder=None):
        """
        Save sub-buffers to its paths with different suffixes.
        Parameters:
            store_folder: a folder to save data in. Store in the current folder if it is None
        Return: None
        """
        if store_folder is None:
            store_folder = './'
        # See
        # https://stackoverflow.com/questions/273192/how-do-i-create-a-directory-and-any-missing-parent-directories
        # Quote
        # pathlib.Path.mkdir as used above recursively creates the directory and does not raise an exception
        # if the directory already exists.
        # If you don't need or want the parents to be created, skip the parents argument.
        pathlib.Path(store_folder).mkdir(parents=True, exist_ok=True)
        path = os.path.join(store_folder, self.storage_prefix)
        with shelve.open(path) as db:
            db['batch_positive_part']       = self.batch_positive_part
            db['batch_negative_part']       = self.batch_negative_part
            db['replay_sub_buf_imbalance']  = self.replay_sub_buf_imbalance
        self.memory_positive_rewards.save(path=path + self.positive_suffix)
        self.memory_negative_rewards.save(path=path + self.negative_suffix)
        self.memory_negative_rewards.save(path=path + self.zero_suffix)

    def load(self, store_folder=None, restore_as_was=True):
        """
        Load sub-buffers from its paths with different suffixes.
        Parameters:
            store_folder: (str) a path to a folder which contains stored data.
            restore_as_was: (str) restore all metadata: seed, buffer sizes, batch sizes ...
        Return: None
        """
        if store_folder is None:
            store_folder = './'
        path = os.path.join(store_folder, self.storage_prefix)
        self.logger.info(f'Going to load Categorical reply buffer from the path =\n{path}')  # TODO:
        with shelve.open(path) as db:
            batch_positive_part        = db['batch_positive_part']
            batch_negative_part        = db['batch_negative_part']
            replay_sub_buf_imbalance   = db['replay_sub_buf_imbalance']
        if restore_as_was:
            self.batch_positive_part        = batch_positive_part
            self.batch_negative_part        = batch_negative_part
            self.replay_sub_buf_imbalance   = replay_sub_buf_imbalance
        self.memory_positive_rewards.load(path=path + self.positive_suffix, restore_as_was=restore_as_was)
        self.memory_negative_rewards.load(path=path + self.negative_suffix, restore_as_was=restore_as_was)
        self.memory_zero_rewards.load(path=path + self.zero_suffix, restore_as_was=restore_as_was)
        self.logger.info(f'Replay Buffer of Positive, Negative and Zero '
                         f'sub-buffers size = {self.internal_buffers_length()}')

# --------------------------- Checks ------------------------


def check_play():
    """
    To check an agent play functionality
    :return: None
    """
    task_name = 'check_play'
    task_logger = create_logger(root_dir='.', log_name=f'log_{task_name}')
    task_logger.info(f'{task_name}: -------------- Start ---------------')

    from Config.CC_MADDPG_config import config_agent

    env_file_name = '../Tennis_Windows_x86_64/Tennis.exe'
    env_seed = 92736
    env = UnityEnvironment(file_name=env_file_name, seed=env_seed)
    brain_name = env.brain_names[0]
    cfg = config_agent.copy()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env_agent = Environment_Agent(env, brain_name, env_seed, cfg, task_logger, device)
    env_agent.play()
    task_logger.info(f'{task_name}: -------------- End ---------------')


def check_train():
    """
    To check an agent train functionality.
    :return: None
    """
    task_name = 'check_train'
    task_logger = create_logger(root_dir='.', log_name=f'log_{task_name}')
    task_logger.info(f'{task_name}: -------------- Start ---------------')

    from Config.CC_MADDPG_config import config_agent

    env_file_name = '../Tennis_Windows_x86_64/Tennis.exe'
    env_seed = 92736
    env = UnityEnvironment(file_name=env_file_name, seed=env_seed)
    brain_name = env.brain_names[0]
    cfg = config_agent.copy()
    cfg['config_train']['buffer_size']      = int(1e5)  # int(1e5)  # replay buffer size
    cfg['config_train']['batch_size']       = 128       # 128       # minibatch size
    cfg['config_train']['gamma']            = 0.99      # 0.99      # discount factor
    cfg['config_train']['tau']              = 1e-3      # 1e-3      # for soft update of target parameters
    cfg['config_train']['lr_actor']         = 1e-4      # 1e-4      # learning rate of the actor
    cfg['config_train']['lr_critic']        = 1e-3      # 1e-3      # learning rate of the critic
    cfg['config_train']['weight_decay']     = 0         # 0         # L2 weight decay

    cfg['config_train']['replay_non_zero_rewards_only'] = True     # True  # Collect experiments with non zeros rewards only.
                                                                    # Otherwise, collect all experiments
    cfg['config_train']['add_other_agent_rewards_part'] = 0.5
    cfg['config_train']['use_this_agent_rewards_part']  = 1.

    cfg['config_train']['add_noise']        = True      # True      # Set True To add random noise during training
    cfg['config_train']['noise_mu']         = 0.        # 0.        # noise mean
    cfg['config_train']['noise_theta']      = 0.15      # 0.15      # noise scale factor of (mu - noise_state)
    cfg['config_train']['noise_sigma']      = 0.2       # 0.2       # noise scale factor of (mu - noise_state)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env_agent = Environment_Agent(env, brain_name, env_seed, cfg, task_logger, device)
    task_logger.info(f'{task_name}: Going to train agents ...')
    episode_scores, average_episode_scores = env_agent.train(n_episodes=1000)
    agent_name = config_agent['config_train']['model_name']
    scores_file_name = env_agent.save_scores(
        agent_type_name=type(env_agent).__name__, agent_name=agent_name + '_scores', scores=episode_scores)
    task_logger.info(f'{task_name}: scores saved in the file \'{scores_file_name}\'')
    average_scores_file_name = env_agent.save_scores(
        agent_type_name=type(env_agent).__name__, agent_name=agent_name + '_average_scores', scores=average_episode_scores)
    task_logger.info(f'{task_name}: average scores saved in the file \'{average_scores_file_name}\'')

    task_logger.info(f'{task_name}: -------------- End ---------------')


def check_centralized_buffer_serialization():
    """
    To check reply buffer save/load functionalities.
    :return: None
    """
    task_name = 'check_centralized_buffer_serialization'
    task_logger = create_logger(root_dir='.', log_name=f'log_{task_name}')
    task_logger.info(f'{task_name}: -------------- Start ---------------')

    # Initialize a memory buffer
    buffer_size = 10
    batch_size  = 2
    seed = 1
    device = 'cpu'
    memory = CentralizedReplayBuffer(buffer_size, batch_size, seed, device)

    # Fill memory with data
    state_sz    = 3
    action_sz   = 2
    reward_sz   = 1
    done_sz     = 1
    n_samples   = 10
    for i in range(1, n_samples+1):
        state   = i * np.ones(state_sz)
        action  = (i+1) * np.ones(action_sz)
        reward  = [i+2] * reward_sz
        next_state  = (i+3) * np.ones(state_sz)
        done    = [i+4] * done_sz
        memory.add(state, action, reward, next_state, done)

    print(f'\n---------- Buffer to save: ----------------\n')
    print(f'{str(memory)}')

    # save memory
    mem_file_name   = './dbg_memory_buffer.npy'
    memory.save(mem_file_name)

    # Create another memory buffer and load it with a data from the saved one
    memory_load = CentralizedReplayBuffer(buffer_size, batch_size, seed, device)
    memory_load.load(mem_file_name)

    print(f'\n---------- Buffer loaded: ----------------\n')
    print(f'{str(memory_load)}')
    task_logger.info(f'{task_name}: --------------  End  ---------------')


def check_centralized_buffer_sampling():
    """
    To check reply buffer sampling functionalities.
    :return: None
    """
    task_name = 'check_centralized_buffer_sampling'
    task_logger = create_logger(root_dir='.', log_name=f'log_{task_name}')
    task_logger.info(f'{task_name}: -------------- Start ---------------')

    # Initialize a memory buffer
    buffer_size = 30
    batch_size  = 5
    add_other_agent_rewards_part = 0.
    use_this_agent_rewards_part = 1.0
    positive_rewards_scale = 1.0
    negative_rewards_scale = 10.0
    zero_rewards_to_negative = -0.005
    sample_all_experiences_to_all_agents = False
    swap_agents_experience = False
    swap_agents_probability = 0.5
    seed = 1
    device = 'cpu'
    num_agents = 2
    memory = CentralizedReplayBuffer(
        buffer_size=buffer_size, batch_size=batch_size,
        add_other_agent_rewards_part=add_other_agent_rewards_part, use_this_agent_rewards_part=use_this_agent_rewards_part,
        positive_rewards_scale=positive_rewards_scale, negative_rewards_scale=negative_rewards_scale,
        zero_rewards_to_negative=zero_rewards_to_negative,
        sample_all_experiences_to_all_agents=sample_all_experiences_to_all_agents,
        swap_agents_experience=swap_agents_experience, swap_agents_probability=swap_agents_probability,
        seed=seed, device=device, logger=task_logger, num_agents=num_agents)

    # Fill memory with data
    state_sz    = (num_agents, 4)
    action_sz   = (num_agents, 2)
    reward_sz   = (num_agents, 1)
    done_sz     = (num_agents, 1)
    n_samples   = 20
    agent_id_to_negate = 0  # 0  # 1
    for i in range(1, n_samples+1):
        state   = i * np.ones(state_sz)
        state[1] *= 10
        action  = (i+1) * np.ones(action_sz)
        action[1] *= 100
        action[i % 2] *= -1
        # reward  = [i+2] * np.ones(reward_sz)
        reward  = (i+2) * np.ones(reward_sz)
        reward[1] *= 1000
        if i % 5 == 0:
            reward[i % 2] *= -1
        if i % 3 == 0:          # To test negate zero
            # reward[i % 2] = 0   # To change agents
            reward[agent_id_to_negate] = 0  # To change agents
        next_state  = (i+3) * np.ones(state_sz)
        next_state[1] *= 10000
        done    = [i+4] * np.ones(done_sz)
        done[1] *= 100000
        memory.add(state, action, reward, next_state, done)
    task_logger.info(f'memory = \n{memory}')

    for smpl in range(3):
        experience = memory.sample()
        task_logger.info(f'experience sample #{smpl} = \n{experience}')
    task_logger.info(f'{task_name}: --------------  End  ---------------')
    pass


def check_accumulate_reward_stat_centralized_buffer():
    """
    To check reply buffer reward statistic accumulation functionality.
    :return: None
    """
    task_name = 'check_accumulate_reward_stat_centralized_buffer'
    task_logger = create_logger(root_dir='.', log_name=f'log_{task_name}')
    task_logger.info(f'{task_name}: -------------- Start ---------------')

    # Initialize a memory buffer
    buffer_size = 10
    batch_size  = 3
    add_other_agent_rewards_part = 0
    use_this_agent_rewards_part = 1.0
    positive_rewards_scale = 1.0
    negative_rewards_scale = 1.0
    zero_rewards_to_negative = -0.005
    sample_all_experiences_to_all_agents = False
    swap_agents_experience = False
    swap_agents_probability = 0.0
    seed = 1
    device = 'cpu'
    num_agents = 2
    memory = CentralizedReplayBuffer(
        buffer_size=buffer_size, batch_size=batch_size,
        add_other_agent_rewards_part=add_other_agent_rewards_part,
        use_this_agent_rewards_part=use_this_agent_rewards_part,
        positive_rewards_scale=positive_rewards_scale, negative_rewards_scale=negative_rewards_scale,
        zero_rewards_to_negative=zero_rewards_to_negative,
        sample_all_experiences_to_all_agents=sample_all_experiences_to_all_agents,
        swap_agents_experience=swap_agents_experience, swap_agents_probability=swap_agents_probability,
        seed=seed, device=device, logger=task_logger, num_agents=num_agents)

    # Fill memory with data
    state_sz    = (num_agents, 4)
    action_sz   = (num_agents, 2)
    reward_sz   = (num_agents, 1)
    done_sz     = (num_agents, 1)
    n_samples   = 10
    for i in range(1, n_samples+1):
        state   = i * np.ones(state_sz)
        state[1] *= 10
        action  = (i+1) * np.ones(action_sz)
        action[1] *= 100
        reward  = [i+2] * np.ones(reward_sz)
        reward[0] *= np.random.choice([-1, 0, 1], size=reward_sz[1])           # add a signe randomly
        reward[1] *= 1000 * np.random.choice([-1, 0, 1], size=reward_sz[1])    # add a signe randomly
        next_state  = (i+3) * np.ones(state_sz)
        next_state[1] *= 10
        done    = [i+4] * np.ones(done_sz)
        done[1] *= 1000
        memory.add(state, action, reward, next_state, done)
    task_logger.info(f'Centralized Replay Buffer = \n{memory}')
    task_logger.info(f'\n ---------------------------------------------- \n')
    rewards = np.array([e.reward for e in memory.memory if e is not None])
    task_logger.info(f'Centralized Replay Buffer rewards = \n{rewards}')

    centralized_reward_stat = CentralizedReplayBuffers_statistics()
    reward_sign_stat = centralized_reward_stat.accumulate_reward_sign_stat(memory)
    task_logger.info(f'\n ---------------------------------------------- \n')
    task_logger.info(f'Centralized Replay Buffer reward sign statistics = \n{reward_sign_stat}')
    task_logger.info(f'{task_name}: --------------  End  ---------------')
    pass


def check_accumulate_reward_val_stat_centralized_buffer():
    """
    To check reply buffer reward statistic accumulation functionality.
    :return: None
    """
    task_name = 'check_accumulate_reward_val_stat_centralized_buffer'
    task_logger = create_logger(root_dir='.', log_name=f'log_{task_name}')
    task_logger.info(f'{task_name}: -------------- Start ---------------')

    # Initialize a memory buffer
    buffer_size = 10
    batch_size  = 3
    add_other_agent_rewards_part = 0
    use_this_agent_rewards_part = 1.0
    positive_rewards_scale = 1.0
    negative_rewards_scale = 1.0
    zero_rewards_to_negative = -0.005
    scale_sample_rewards = 1.0
    sample_all_experiences_to_all_agents = False
    swap_agents_experience = False
    swap_agents_probability = 0.0
    seed = 1
    device = 'cpu'
    num_agents = 2
    memory = CentralizedReplayBuffer(
        buffer_size=buffer_size, batch_size=batch_size, add_other_agent_rewards_part=add_other_agent_rewards_part,
        use_this_agent_rewards_part=use_this_agent_rewards_part,
        positive_rewards_scale=positive_rewards_scale, negative_rewards_scale=negative_rewards_scale,
        zero_rewards_to_negative=zero_rewards_to_negative,
        sample_all_experiences_to_all_agents=sample_all_experiences_to_all_agents,
        swap_agents_experience=swap_agents_experience, swap_agents_probability=swap_agents_probability,
        seed=seed, device=device, logger=task_logger, num_agents=num_agents)

    # Fill memory with data
    state_sz    = (num_agents, 4)
    action_sz   = (num_agents, 2)
    reward_sz   = (num_agents, 1)
    done_sz     = (num_agents, 1)
    n_samples   = 10
    for i in range(1, n_samples+1):
        state   = i * np.ones(state_sz)
        state[1] *= 10
        action  = (i+1) * np.ones(action_sz)
        action[1] *= 100
        reward  = [i+2] * np.ones(reward_sz)
        reward[0] *= np.random.choice([-1, 0, 1], size=reward_sz[1])           # add a signe randomly
        reward[1] *= 1000 * np.random.choice([-1, 0, 1], size=reward_sz[1])    # add a signe randomly
        next_state  = (i+3) * np.ones(state_sz)
        next_state[1] *= 10
        done    = [i+4] * np.ones(done_sz)
        done[1] *= 1000
        memory.add(state, action, reward, next_state, done)
    task_logger.info(f'Centralized Replay Buffer = \n{memory}')
    task_logger.info(f'\n ---------------------------------------------- \n')
    rewards = np.array([e.reward for e in memory.memory if e is not None])
    task_logger.info(f'Centralized Replay Buffer rewards = '
                     f'\n{rewards}')

    centralized_reward_stat = CentralizedReplayBuffers_statistics()
    centralized_reward_sign_stat = centralized_reward_stat.accumulate_reward_sign_stat(memory)
    task_logger.info(f'Centralized Replay Buffer reward sign statistics = \n{centralized_reward_sign_stat}')
    centralized_reward_val_stat = centralized_reward_stat.accumulate_reward_val_per_sign_stat(memory)
    task_logger.info(f'\n ---------------------------------------------- \n')
    task_logger.info(f'Centralized Replay Buffer reward value statistics of agent[0] = \n{centralized_reward_val_stat[:, :, 0]}')
    task_logger.info(f'Centralized Replay Buffer reward value statistics of agent[1] = \n{centralized_reward_val_stat[:, :, 1]}')
    task_logger.info(f'\n ---------------------------------------------- \n')
    task_logger.info(f'{task_name}: --------------  End  ---------------')
    pass


def check_centralized_buffer_agent_experience_swapping():
    """
    To check reply buffer save/load functionalities.
    :return: None
    """
    task_name = 'check_centralized_buffer_agent_experience_swapping'
    task_logger = create_logger(root_dir='.', log_name=f'log_{task_name}')
    task_logger.info(f'{task_name}: -------------- Start ---------------')

    # Initialize a memory buffer
    buffer_size = 100
    batch_size  = 10  # 3
    add_other_agent_rewards_part = 0  # 0.5
    use_this_agent_rewards_part = 1.0
    scale_sample_rewards = 1.0
    sample_all_experiences_to_all_agents = False
    seed = 1
    device = 'cpu'
    num_agents = 2

    def initialize_buffer():
        rbuf = CentralizedReplayBuffer(
            buffer_size=buffer_size,
            batch_size=batch_size,
            add_other_agent_rewards_part=add_other_agent_rewards_part,
            use_this_agent_rewards_part=use_this_agent_rewards_part,
            scale_sample_rewards=scale_sample_rewards,
            sample_all_experiences_to_all_agents=sample_all_experiences_to_all_agents,
            swap_agents_experience=swap_agents_experience,
            swap_agents_probability=swap_agents_probability,
            seed=seed, device=device, logger=task_logger, num_agents=num_agents)

        # Fill memory with data
        state_sz    = (num_agents, 4)
        action_sz   = (num_agents, 2)
        reward_sz   = (num_agents, 1)
        done_sz     = (num_agents, 1)
        n_samples   = 10
        for i in range(1, n_samples+1):
            state   = i * np.ones(state_sz)
            state[1] *= 10
            action  = (i+1) * np.ones(action_sz)
            action[1] *= 100
            reward  = [i+2] * np.ones(reward_sz)
            reward[1] *= 1000
            next_state  = (i+3) * np.ones(state_sz)
            next_state[1] *= 10
            done    = np.ones(done_sz, dtype=bool) if i % 2 == 1 else np.zeros(done_sz, dtype=bool)
            done[1] = not done[1]
            rbuf.add(state, action, reward, next_state, done)
        return rbuf

    for swap_agents_probability in [1.0, 0.5, 0.0]:
        swap_agents_experience = False
        set_global_seed_in_libs(seed)
        memory = initialize_buffer()
        experience = memory.experience(*memory.sample())

        swap_agents_experience = not swap_agents_experience
        set_global_seed_in_libs(seed)
        memory_swap = initialize_buffer()
        experience_swap = memory.experience(*memory_swap.sample())

        task_logger.info(f'---------------------------------------------------------------------------')
        task_logger.info(f'--------- Swap Agent probability = {swap_agents_probability} --------------')
        task_logger.info(f'---------------------------------------------------------------------------')
        task_logger.info(f'state of the sample = \n{experience.state},\nstate of the swap  sample = \n{experience_swap.state}')
        task_logger.info(f'action of the sample = \n{experience.action},\naction of the swap  sample = \n{experience_swap.action}')
        task_logger.info(f'reward of the sample = \n{experience.reward},\nreward of the swap  sample = \n{experience_swap.reward}')
        task_logger.info(f'next_state of the sample = \n{experience.next_state},\nnext_state of the swap  sample = \n{experience_swap.next_state}')
        task_logger.info(f'done of the sample = \n{experience.done},\ndone of the swap  sample = \n{experience_swap.done}')

    task_logger.info(f'{task_name}: --------------  End  ---------------')
    pass


def check_categorical_buffer_load(database_path='../train_agents_40000/pretrained_reply_buffer'):
    """
    To check categorical reply buffer save/load functionalities.
    :return: None
    """
    task_name = 'check_categorical_buffer_load'
    task_logger = create_logger(root_dir='.', log_name=f'log_{task_name}')
    task_logger.info(f'{task_name}: -------------- Start ---------------')

    # Initialize a memory buffer
    buffer_size = int(1e6)
    batch_size  = 2
    batch_positive_part = 0.8
    replay_sub_buf_imbalance = 0.001
    combine_this_rewards_part = 1.0
    sample_all_experiences_to_all_agents = True
    seed = 1
    device = 'cpu'
    memory = CategoricalReplayBuffers(buffer_size, batch_size, batch_positive_part, replay_sub_buf_imbalance, combine_this_rewards_part,
             sample_all_experiences_to_all_agents, seed, device, task_logger)

    memory.load(database_path, restore_as_was=False)
    task_logger.info(f'\nBuffer loaded: size = {len(memory)}\n')
    task_logger.info(f'{task_name}: --------------  End  ---------------')


def check_categorical_buffer_force_update(database_path, logger=None):
    # database_path='../debug/swap_agents_experience/RBuf-2048/replay_buffer'
    # database_path='../debug/swap_agents_experience/RBuf-2048/pretrained_replay_buffer'
    # database_path='../debug/pos-neg-buf/4_rbuf-512_ep-1500/pretrained_replay_buffer'
    # database_path = 'C:/Data/Study/Udacity/DeeoReinforcementLearning/Cource4/Code/Progect/p3_collaboration-competition_pos-neg-rbuf/pos-neg-buf/5_o-part-0.25_update-per-10-new_rbuf-3800_swap_ep-2500/replay_buffer'
    # database_path = 'C:/Data/Study/Udacity/DeeoReinforcementLearning/Cource4/Code/Progect/p3_collaboration-competition_pos-neg-rbuf/pos-neg-buf/9_o-part-0.25_update-per-100-new_rbuf-3800_swap_ep-150000/replay_buffer'
    """
    To check categorical reply buffer force_update functionalities.
    :return: None
    """
    task_name = 'check_categorical_buffer_force_update'
    if logger is None:
        task_logger = create_logger(root_dir='.', log_name=f'log_{task_name}')
    else:
        task_logger = logger
    task_logger.info(f'{task_name}: -------------- Start ---------------')

    # Initialize a memory buffer
    buffer_size = int(1e6)
    batch_size = 256 # 10  # 256
    batch_positive_part = 0.8
    replay_sub_buf_imbalance = 0.001
    add_other_agent_rewards_part = 0
    use_this_agent_rewards_part = 1
    positive_rewards_scale = 1
    negative_rewards_scale = 10
    zero_rewards_to_negative = 0
    combine_this_rewards_part = 1.0
    sample_all_experiences_to_all_agents = False
    swap_agents_experience = False
    swap_agents_probability = 0.0
    seed = 1
    device = 'cpu'
    num_agents = 2

    memory_src = CategoricalReplayBuffers(
        buffer_size=buffer_size, batch_size=batch_size,
        batch_positive_part=batch_positive_part, replay_sub_buf_imbalance=replay_sub_buf_imbalance,
        add_other_agent_rewards_part=add_other_agent_rewards_part,
        use_this_agent_rewards_part=use_this_agent_rewards_part, positive_rewards_scale=positive_rewards_scale,
        negative_rewards_scale=negative_rewards_scale, zero_rewards_to_negative=zero_rewards_to_negative,
        sample_all_experiences_to_all_agents=sample_all_experiences_to_all_agents,
        swap_agents_experience=swap_agents_experience, swap_agents_probability=swap_agents_probability,
        seed=seed, device=device, logger=task_logger, num_agents=num_agents)

    task_logger.info(f'\nGoing to load a source buffer')
    memory_src.load(database_path, restore_as_was=False)
    task_logger.info(f'\nThe source buffer loaded: size = {len(memory_src)}\n')

    memory_src_reward_statistics = CategoricalReplayBuffers_statistics()
    memory_src_reward_sign_statistics = memory_src_reward_statistics.accumulate_reward_sign_stat(memory_src)
    memory_src_reward_val_per_sign_statistics = memory_src_reward_statistics.accumulate_reward_val_per_sign_stat(memory_src)
    memory_src_reward_statistics.print_statistics(
        logger=task_logger, reward_sign_statistics=memory_src_reward_sign_statistics,
        reward_val_per_sign_statistics=memory_src_reward_val_per_sign_statistics, statistics_name='Categorical Source Replay Buffer Statistics')

    memory_new = CategoricalReplayBuffers(
        buffer_size=buffer_size, batch_size=batch_size,
        batch_positive_part=batch_positive_part, replay_sub_buf_imbalance=replay_sub_buf_imbalance,
        add_other_agent_rewards_part=add_other_agent_rewards_part,
        use_this_agent_rewards_part=use_this_agent_rewards_part, positive_rewards_scale=positive_rewards_scale,
        negative_rewards_scale=negative_rewards_scale, zero_rewards_to_negative=zero_rewards_to_negative,
        sample_all_experiences_to_all_agents=sample_all_experiences_to_all_agents,
        swap_agents_experience=swap_agents_experience, swap_agents_probability=swap_agents_probability,
        seed=seed, device=device, logger=task_logger, num_agents=num_agents)

    task_logger.info(f'\nGoing to copy from the source buffer to the new buffer')
    memory_new.force_to_append_from_buffer(memory_src)
    task_logger.info(f'\nBuffer memory_new: size = {len(memory_new)}\n')

    task_logger.info(f'\nCalculate the new buffer statistics.')
    memory_new_reward_statistics = CategoricalReplayBuffers_statistics()
    memory_new_reward_sign_statistics = memory_new_reward_statistics.accumulate_reward_sign_stat(memory_new)
    memory_new_reward_val_per_sign_statistics = memory_new_reward_statistics.accumulate_reward_val_per_sign_stat(memory_new)
    memory_new_reward_statistics.print_statistics(
        logger=task_logger, reward_sign_statistics=memory_new_reward_sign_statistics,
        reward_val_per_sign_statistics=memory_new_reward_val_per_sign_statistics, statistics_name='Categorical New Replay Buffer Statistics')

    task_logger.info(f'\nCreate update buffer and add new buffer experiences to it.')
    memory_update = CategoricalReplayBuffers(
        buffer_size=buffer_size, batch_size=batch_size,
        batch_positive_part=batch_positive_part, replay_sub_buf_imbalance=replay_sub_buf_imbalance,
        add_other_agent_rewards_part=add_other_agent_rewards_part,
        use_this_agent_rewards_part=use_this_agent_rewards_part, positive_rewards_scale=positive_rewards_scale,
        negative_rewards_scale=negative_rewards_scale, zero_rewards_to_negative=zero_rewards_to_negative,
        sample_all_experiences_to_all_agents=sample_all_experiences_to_all_agents,
        swap_agents_experience=swap_agents_experience, swap_agents_probability=swap_agents_probability,
        seed=seed, device=device, logger=task_logger, num_agents=num_agents)

    task_logger.info(f'\nSample a batch experience from the New buffer.')
    experiences = memory_new.sample()
    stt, act, rwrd, nxt_stt, dn = experiences
    task_logger.info(f'The Sample experience has batch size = {stt.shape[0]}.')

    task_logger.info(f'\nAdd these samples from the New buffer to the Update buffer.')

    def to_numpy(v):
        if v.type() is not np.array:
            v = np.array(v)
        return v

    stt, act, rwrd, nxt_stt, dn = to_numpy(stt), to_numpy(act), to_numpy(rwrd), to_numpy(nxt_stt), to_numpy(dn)
    samples = memory_update.force_to_add(stt, act, rwrd, nxt_stt, dn)

    task_logger.info(f'\nCalculate the update buffer statistics.')
    memory_update_reward_statistics = CategoricalReplayBuffers_statistics()
    memory_update_reward_sign_statistics = memory_update_reward_statistics.accumulate_reward_sign_stat(memory_update)
    memory_update_reward_val_per_sign_statistics = memory_update_reward_statistics.accumulate_reward_val_per_sign_stat(memory_update)
    memory_update_reward_statistics.print_statistics(
        logger=task_logger, reward_sign_statistics=memory_update_reward_sign_statistics,
        reward_val_per_sign_statistics=memory_update_reward_val_per_sign_statistics,
        statistics_name='Categorical Update')


    task_logger.info(f'{task_name}: --------------  End  ---------------')


def check_categorical_buffer_serialization():
    """
    To check categorical reply buffer save/load functionalities.
    :return: None
    """
    task_name = 'check_categorical_buffer_serialization'
    task_logger = create_logger(root_dir='.', log_name=f'log_{task_name}')
    task_logger.info(f'{task_name}: -------------- Start ---------------')

    # Initialize a memory buffer
    batch_positive_part = 0.5
    buffer_size = 10
    batch_size  = 2
    seed = 1
    device = 'cpu'
    memory = CategoricalReplayBuffers(buffer_size, batch_size, batch_positive_part, seed, device)

    # Fill memory with an experience samples
    state_sz    = 3
    action_sz   = 2
    reward_sz   = 1
    done_sz     = 1
    n_samples   = 10
    for i in range(1, n_samples+1):
        state   = i * np.ones(state_sz)
        action  = (i+1) * np.ones(action_sz)
        reward  = [(-1)**i * (i+2)] * reward_sz
        next_state  = (i+3) * np.ones(state_sz)
        done    = [i+4] * done_sz
        memory.add(state, action, reward, next_state, done)
    print(f'\n---------- Buffer to save: ----------------\n')
    print(f'{str(memory)}')

    # save memory
    mem_file_name   = './dbg_memory_buffer.npy'
    memory.save(mem_file_name)

    # Create another memory buffer and load it with a data from the saved one
    memory_load = CategoricalReplayBuffers(buffer_size, batch_size, batch_positive_part, seed, device)
    memory_load.load(mem_file_name)

    print(f'\n---------- Buffer loaded: ----------------\n')
    print(f'{str(memory_load)}')
    task_logger.info(f'{task_name}: --------------  End  ---------------')


def check_categorical_buffer_sampling():
    """
    To check Categorical reply buffer sampling functionalities.
    :return: None
    """
    task_name = 'check_categorical_buffer_sampling'
    task_logger = create_logger(root_dir='.', log_name=f'log_{task_name}')
    task_logger.info(f'{task_name}: -------------- Start ---------------')

    # Initialize a memory buffer
    buffer_size = 10
    batch_size  = 10
    batch_positive_part = 0.80
    replay_sub_buf_imbalance = 0.001
    add_other_agent_rewards_part = 0.5
    use_this_agent_rewards_part = 1.0
    positive_rewards_scale = 1.0
    negative_rewards_scale = 1.0
    zero_rewards_to_negative = -0.005
    sample_all_experiences_to_all_agents = False
    swap_agents_experience = False
    swap_agents_probability = 0.5  # 0.1  # 0.0
    seed = 1
    device = 'cpu'
    num_agents = 2
    memory = CategoricalReplayBuffers(
        buffer_size=buffer_size, batch_size=batch_size, batch_positive_part=batch_positive_part,
        replay_sub_buf_imbalance=replay_sub_buf_imbalance, add_other_agent_rewards_part=add_other_agent_rewards_part,
        use_this_agent_rewards_part=use_this_agent_rewards_part,
        positive_rewards_scale=positive_rewards_scale, negative_rewards_scale=negative_rewards_scale,
        zero_rewards_to_negative=zero_rewards_to_negative,
        sample_all_experiences_to_all_agents=sample_all_experiences_to_all_agents,
        swap_agents_experience=swap_agents_experience, swap_agents_probability=swap_agents_probability,
        seed=seed, device=device, logger=task_logger, num_agents=num_agents)

    # Fill memory with data
    state_sz    = (num_agents, 4)
    action_sz   = (num_agents, 2)
    reward_sz   = (num_agents, 1)
    done_sz     = (num_agents, 1)
    n_samples   = 10
    n_samples_positive_reward = int(n_samples * batch_positive_part)
    n_samples_negative_reward = int(n_samples - n_samples_positive_reward)
    for i in range(1, n_samples+1):
        state   = i * np.ones(state_sz)
        state[1] *= 10
        action  = (i+1) * np.ones(action_sz)
        action[1] *= 100
        if i <= n_samples_positive_reward:
            reward_sign = 1
        elif i <= n_samples_positive_reward + n_samples_negative_reward:
            reward_sign = -1
        else:
            reward_sign = 0
        reward  = [i+2] * np.ones(reward_sz) * reward_sign
        reward[1] *= 1000
        next_state  = (i+3) * np.ones(state_sz)
        next_state[1] *= 10
        done    = [i+4] * np.ones(done_sz)
        done[1] *= 1000
        memory.add(state, action, reward, next_state, done)
    task_logger.info(f'memory = \n{memory}')

    experience = memory.sample()
    task_logger.info(f'experience fields are {memory.memory_positive_rewards.experience._fields}')
    task_logger.info(f'experience sample = \n{experience}')
    task_logger.info(f'{task_name}: --------------  End  ---------------')
    pass


def check_categorical_buffer_swap_sampling():
    """
    To check Categorical reply buffer sampling functionalities.
    :return: None
    """
    task_name = 'check_categorical_buffer_swap_sampling'
    task_logger = create_logger(root_dir='.', log_name=f'log_{task_name}')
    task_logger.info(f'{task_name}: -------------- Start ---------------')

    # Initialize a memory buffer
    buffer_size = 10
    batch_size  = 10
    batch_positive_part = 0.80
    replay_sub_buf_imbalance = 0.001
    add_other_agent_rewards_part = 0.
    use_this_agent_rewards_part = 1.0
    positive_rewards_scale = 1.0
    negative_rewards_scale = 1.0
    sample_all_experiences_to_all_agents = False
    swap_agents_experience = False
    swap_agents_probability = 0.5  # 1.0  # 0.0
    seed = 1
    device = 'cpu'
    num_agents = 2

    def initialize_reply_buffer():
        rbuf = CategoricalReplayBuffers(
            buffer_size, batch_size, batch_positive_part,
            replay_sub_buf_imbalance, add_other_agent_rewards_part, use_this_agent_rewards_part,
            positive_rewards_scale, negative_rewards_scale,
            sample_all_experiences_to_all_agents, swap_agents_experience, swap_agents_probability,
            seed, device, task_logger, num_agents)

        # Fill Reply Buffer with data
        state_sz    = (num_agents, 4)
        action_sz   = (num_agents, 2)
        reward_sz   = (num_agents, 1)
        done_sz     = (num_agents, 1)
        n_samples   = 10
        n_samples_positive_reward = int(n_samples * batch_positive_part)
        n_samples_negative_reward = int(n_samples - n_samples_positive_reward)
        for i in range(1, n_samples+1):
            state   = i * np.ones(state_sz)
            state[1] *= 10
            action  = (i+1) * np.ones(action_sz)
            action[1] *= 100
            if i <= n_samples_positive_reward:
                reward_sign = 1
            elif i <= n_samples_positive_reward + n_samples_negative_reward:
                reward_sign = -1
            else:
                reward_sign = 0
            reward  = [i+2] * np.ones(reward_sz) * reward_sign
            reward[1] *= 1000
            next_state  = (i+3) * np.ones(state_sz)
            next_state[1] *= 10
            done    = [i+4] * np.ones(done_sz)
            done[1] *= 1000
            rbuf.add(state, action, reward, next_state, done)
        return rbuf

    memory      = initialize_reply_buffer()
    experience  = memory.sample()

    swap_agents_experience = not swap_agents_experience
    set_global_seed_in_libs(seed)
    memory_swap = initialize_reply_buffer()
    experience_swap  = memory_swap.sample()

    task_logger.info(f'state of the sample = \n{experience[0]},\nstate of the swap  sample = \n{experience_swap[0]}')
    task_logger.info(f'action of the sample = \n{experience[1]},\naction of the swap  sample = \n{experience_swap[1]}')
    task_logger.info(f'reward of the sample = \n{experience[2]},\nreward of the swap  sample = \n{experience_swap[2]}')
    task_logger.info(f'next_state of the sample = \n{experience[3]},\nnext_state of the swap  sample = \n{experience_swap[3]}')
    task_logger.info(f'done of the sample = \n{experience[4]},\ndone of the swap  sample = \n{experience_swap[4]}')

    task_logger.info(f'{task_name}: --------------  End  ---------------')
    pass


def check_replay_buffer_reward_statistics():
    """ Load and check pretrained Replay Buffer experience reward statistics """

    def calc_centralized_buffer_stats(db_memory: CentralizedReplayBuffer, rwp_t, rwp_o):
        rewards = np.array([[rwp_t * e.reward[0] + rwp_o * e.reward[1],  # combine rewards from this(0) and another(1) agent
                             rwp_t * e.reward[1] + rwp_o * e.reward[0]]  # combine rewards from this(1) and another(0) agent
                            for e in db_memory.memory if e is not None])
        stat_axis = 0
        r_max       = rewards.max(axis=stat_axis)
        r_min       = rewards.min(axis=stat_axis)
        r_mean      = rewards.mean(axis=stat_axis)
        return r_max, r_min, r_mean

    task_name = 'check_replay_buffer_rewards'
    task_logger = create_logger(root_dir='.', log_name=f'log_{task_name}')
    task_logger.info(f'{task_name}: -------------- Start ---------------')

    # db_dir = '../debug/dbg_replay_buffer'
    db_dir = '../debug/no-zero-RBuf/db_528'
    load_reply_buffer_path = os.path.join(db_dir, 'replay_buffer')

    buffer_size                     = 1e5
    batch_size                      = 1024
    batch_positive_part             = 0.8
    replay_sub_buf_imbalance        = 0.00005
    add_other_agent_rewards_part    = 0.25
    use_this_agent_rewards_part     = 1.
    sample_all_experiences_to_all_agents = False
    seed                            = 92736
    device                          = 'cpu'
    num_agents                      = 2

    memory = CategoricalReplayBuffers(
        buffer_size, batch_size, batch_positive_part,
        replay_sub_buf_imbalance,
        add_other_agent_rewards_part, use_this_agent_rewards_part,
        sample_all_experiences_to_all_agents, seed, device, task_logger, num_agents
    )

    as_is = False
    memory.load(load_reply_buffer_path, restore_as_was=as_is)  # load reply buffer to continue training

    positive_len, negative_len = memory.internal_buffers_length()
    full_len = len(memory)
    task_logger.info(f'RBuf len = {full_len}, positive len = {positive_len}, negative length = {negative_len}')

    positive_part = positive_len / full_len
    negative_part = negative_len / full_len
    task_logger.info(f'positive part = {positive_part:.3f}, negative_part = {negative_part:.3f}')

    rwp_ts = [1, 1]
    rwp_os = [0, 0.25]
    for rwp_t, rwp_o in zip(rwp_ts, rwp_os):
        r_max, r_min, r_mean = calc_centralized_buffer_stats(memory.memory_positive_rewards, rwp_t, rwp_o)
        task_logger.info(f'Positive samples buffer reward with rwp_t = {rwp_t} rwp_o = {rwp_o}:'
                         f'\n max = \n {r_max}, \n min = \n {r_min}, \n mean = \n {r_mean}')

        r_max, r_min, r_mean = calc_centralized_buffer_stats(memory.memory_negative_rewards, rwp_t, rwp_o)
        task_logger.info(f'Negative samples buffer reward with rwp_t = {rwp_t} rwp_o = {rwp_o}:'
                         f'\n max = \n {r_max}, \n min = \n {r_min}, \n mean = \n {r_mean}')

    task_logger.info(f'{task_name}: --------------  End  ---------------')


def REMOVE_check_replay_buffer_reward_statistics():
    """ Load and check pretrained Replay Buffer experience reward statistics """

    def calc_centralized_buffer_stats(db_memory: CentralizedReplayBuffer, rwp_t, rwp_o):
        rewards = np.array([[rwp_t * e.reward[0] + rwp_o * e.reward[1],  # combine rewards from this(0) and another(1) agent
                             rwp_t * e.reward[1] + rwp_o * e.reward[0]]  # combine rewards from this(1) and another(0) agent
                            for e in db_memory.memory if e is not None])
        stat_axis = 0
        r_max       = rewards.max(axis=stat_axis)
        r_min       = rewards.min(axis=stat_axis)
        r_mean      = rewards.mean(axis=stat_axis)
        return r_max, r_min, r_mean

    task_name = 'check_replay_buffer_rewards'
    task_logger = create_logger(root_dir='.', log_name=f'log_{task_name}')
    task_logger.info(f'{task_name}: -------------- Start ---------------')

    db_dir = '../debug/dbg_replay_buffer'
    load_reply_buffer_path = os.path.join(db_dir, 'replay_buffer')

    buffer_size                     = 1e5
    batch_size                      = 1024
    batch_positive_part             = 0.8
    batch_negative_part             = 0.15
    replay_sub_buf_imbalance        = 0.00005
    add_other_agent_rewards_part    = 0.25
    use_this_agent_rewards_part     = 1.
    sample_all_experiences_to_all_agents = False
    seed                            = 92736
    device                          = 'cpu'
    num_agents                      = 2

    memory = CategoricalReplayBuffers(
        buffer_size, batch_size, batch_positive_part, batch_negative_part,
        replay_sub_buf_imbalance,
        add_other_agent_rewards_part, use_this_agent_rewards_part,
        sample_all_experiences_to_all_agents, seed, device, task_logger, num_agents
    )

    as_is = False
    memory.load(load_reply_buffer_path, restore_as_was=as_is)  # load reply buffer to continue training

    rwp_ts = [1, 1]
    rwp_os = [0, 0.25]
    for rwp_t, rwp_o in zip(rwp_ts, rwp_os):
        r_max, r_min, r_mean = calc_centralized_buffer_stats(memory.memory_positive_rewards, rwp_t, rwp_o)
        task_logger.info(f'Positive samples buffer reward with rwp_t = {rwp_t} rwp_o = {rwp_o}:'
                         f'\n max = \n {r_max}, \n min = \n {r_min}, \n mean = \n {r_mean}')

        r_max, r_min, r_mean = calc_centralized_buffer_stats(memory.memory_negative_rewards, rwp_t, rwp_o)
        task_logger.info(f'Negative samples buffer reward with rwp_t = {rwp_t} rwp_o = {rwp_o}:'
                         f'\n max = \n {r_max}, \n min = \n {r_min}, \n mean = \n {r_mean}')

        r_max, r_min, r_mean = calc_centralized_buffer_stats(memory.memory_zero_rewards, rwp_t, rwp_o)
        task_logger.info(f'Zero samples buffer reward with rwp_t = {rwp_t} rwp_o = {rwp_o}:'
                         f'\n max = \n {r_max}, \n min = \n {r_min}, \n mean = \n {r_mean}')


    task_logger.info(f'{task_name}: --------------  End  ---------------')


def check_accumulate_categorical_buffer_reward_statistics(
        database_path='../debug/swap_agents_experience/RBuf-2048/pretrained_replay_buffer/'):
    """
    To check categorical reply buffer save/load functionalities.
    :return: None
    """
    task_name = 'check_accumulate_categorical_buffer_reward_statistics'
    task_logger = create_logger(root_dir='.', log_name=f'log_{task_name}')
    task_logger.info(f'{task_name}: -------------- Start ---------------')

    # Initialize a memory buffer
    buffer_size = int(1e6)
    batch_size  = 2
    batch_positive_part = 0.8
    replay_sub_buf_imbalance = 0.001
    add_other_agent_rewards_part = 0
    use_this_agent_rewards_part = 1
    positive_rewards_scale = 1
    negative_rewards_scale = 1
    zero_rewards_to_negative = -0.005
    sample_all_experiences_to_all_agents = True
    swap_agents_experience = False
    swap_agents_probability = 0.0
    seed = 1
    device = 'cpu'
    num_agents = 2

    memory = CategoricalReplayBuffers(
        buffer_size=buffer_size, batch_size=batch_size,
        batch_positive_part=batch_positive_part, replay_sub_buf_imbalance=replay_sub_buf_imbalance,
        add_other_agent_rewards_part=add_other_agent_rewards_part,
        use_this_agent_rewards_part=use_this_agent_rewards_part, positive_rewards_scale=positive_rewards_scale,
        negative_rewards_scale=negative_rewards_scale, zero_rewards_to_negative=zero_rewards_to_negative,
        sample_all_experiences_to_all_agents=sample_all_experiences_to_all_agents,
        swap_agents_experience=swap_agents_experience, swap_agents_probability=swap_agents_probability,
        seed=seed, device=device, logger=task_logger, num_agents=num_agents)

    memory.load(database_path, restore_as_was=False)
    task_logger.info(f'\nBuffer loaded: size = {len(memory)}\n')
    reward_statistics = CategoricalReplayBuffers_statistics()
    reward_statistics = reward_statistics.accumulate_reward_sign_stat(memory)
    task_logger.info(f'\nPositive Replay Buffer reward sign statistics = \n{reward_statistics[0]}\n')
    task_logger.info(f'--------------------------------------------------')
    task_logger.info(f'\nPositive - Negative Replay Buffer reward sign statistics = \n{reward_statistics[1]}\n')
    task_logger.info(f'--------------------------------------------------')
    task_logger.info(f'\nNegative Replay Buffer reward sign statistics = \n{reward_statistics[2]}\n')
    task_logger.info(f'{task_name}: --------------  End  ---------------')


def check_accumulate_reward_val_stat_categorical_buffer(
        database_path='../debug/swap_agents_experience/RBuf-2048/replay_buffer/'):
        # database_path='../debug/swap_agents_experience/RBuf-2048/pretrained_replay_buffer/'):
    """
    To check categorical reply buffer save/load functionalities.
    :return: None
    """
    task_name = 'check_accumulate_reward_val_stat_categorical_buffer'
    task_logger = create_logger(root_dir='.', log_name=f'log_{task_name}')
    task_logger.info(f'{task_name}: -------------- Start ---------------')

    # Initialize a memory buffer
    buffer_size = int(1e6)
    batch_size  = 2
    batch_positive_part = 0.8
    replay_sub_buf_imbalance = 0.001
    add_other_agent_rewards_part = 0
    use_this_agent_rewards_part = 1
    positive_rewards_scale = 1
    negative_rewards_scale = 1
    zero_rewards_to_negative = -0.005
    combine_this_rewards_part = 1.0
    sample_all_experiences_to_all_agents = True
    swap_agents_experience = False
    swap_agents_probability = 0.0
    seed = 1
    device = 'cpu'
    num_agents = 2

    # os.path.join('../debug/swap_agents_experience/RBuf-2048/pretrained_replay_buffer/')

    memory = CategoricalReplayBuffers(
        buffer_size=buffer_size, batch_size=batch_size,
        batch_positive_part=batch_positive_part, replay_sub_buf_imbalance=replay_sub_buf_imbalance,
        add_other_agent_rewards_part=add_other_agent_rewards_part,
        use_this_agent_rewards_part=use_this_agent_rewards_part, positive_rewards_scale=positive_rewards_scale,
        negative_rewards_scale=negative_rewards_scale, zero_rewards_to_negative=zero_rewards_to_negative,
        sample_all_experiences_to_all_agents=sample_all_experiences_to_all_agents,
        swap_agents_experience=swap_agents_experience, swap_agents_probability=swap_agents_probability,
        seed=seed, device=device, logger=task_logger, num_agents=num_agents)

    memory.load(database_path, restore_as_was=False)
    task_logger.info(f'\nBuffer loaded: size = {len(memory)}\n')
    reward_statistics = CategoricalReplayBuffers_statistics()
    reward_sign_statistics = reward_statistics.accumulate_reward_sign_stat(memory)
    reward_val_per_sign_statistics = reward_statistics.accumulate_reward_val_per_sign_stat(memory)
    reward_statistics.print_statistics(
        logger=task_logger, reward_sign_statistics=reward_sign_statistics,
        reward_val_per_sign_statistics=reward_val_per_sign_statistics, statistics_name='Categorical Replay Buffer Statistics')
    task_logger.info(f'{task_name}: --------------  End  ---------------')


def check_load_categorical_statistics(stat_path: str='../debug/negate_zero_rwrd/5_prepare_statistics\critic_stat'):
    stat = CategoricalReplayBuffers_statistics()
    stat.load_statistics(path_to_load=stat_path)
    pass


def check_plot_categorical_statistics(stat_path: str='../debug/negate_zero_rwrd/5_prepare_statistics\critic_stat'):
    stat = CategoricalReplayBuffers_statistics()
    stat.plot_statistics(path_to_load=stat_path)
    pass


def check_plot_statistics():
    save_dir = '../debug/episode_reward_statistics/5_db-512_neg-rwrd-scale-10_ep-15000'
    r_buf_stat = CategoricalReplayBuffers_statistics()
    r_buf_stat.plot_statistics(os.path.join(save_dir, CategoricalReplayBuffers_statistics.f_actor_prefix))
    r_buf_stat.plot_statistics(os.path.join(save_dir, CategoricalReplayBuffers_statistics.f_critic_prefix))


def check_model_copying():
    task_name = 'check_model_copying'
    print(f'{task_name}: --------------  Start  ---------------')
    import torch.nn as nn
    in_feature = 1
    out_feature = 1
    a_linear = nn.Linear(in_feature,out_feature)
    print(f'a_linear - created')
    print(f'a_linear.weight.data[0, 0] = {a_linear.weight.data[0, 0]}')

    a_linear_type = a_linear.__class__
    b_linear = a_linear_type(in_feature, out_feature)
    print(f'b_linear - created')
    print(f'b_linear.weight.data[0, 0] = {b_linear.weight.data[0, 0]}')

    b_linear.load_state_dict(a_linear.state_dict())
    print(f'b_linear - copied')
    print(f'b_linear.weight.data[0, 0] = {b_linear.weight.data[0, 0]}')

    a_linear.weight.data[0, 0] = 1.
    print(f'a_linear - changed')
    print(f'a_linear.weight.data[0, 0] = {a_linear.weight.data[0, 0]}')
    print(f'b_linear.weight.data[0, 0] = {b_linear.weight.data[0, 0]}')

    print(f'{task_name}: --------------  End  ---------------')


if __name__ == '__main__':
    # check_play()
    # check_train()
    # check_centralized_buffer_serialization()
    # check_centralized_buffer_sampling()
    # check_accumulate_reward_stat_centralized_buffer()
    # check_accumulate_reward_val_stat_centralized_buffer()
    # check_centralized_buffer_agent_experience_swapping()
    # check_categorical_buffer_load()

    # check_categorical_buffer_force_update(database_path=
    # 'C:/Data/Study/Udacity/DeeoReinforcementLearning/Cource4/Code/Progect'
    # '/p3_collaboration-competition_pos-neg-rbuf/pos-neg-buf/9_o-part-0.25_update-per-100-new_rbuf-3800_swap_ep-150000/'
    # 'replay_buffer')

    # check_categorical_buffer_serialization()
    # check_categorical_buffer_sampling()
    # check_categorical_buffer_swap_sampling()
    # check_replay_buffer_reward_statistics()
    # check_accumulate_categorical_buffer_reward_statistics()
    # check_accumulate_reward_val_stat_categorical_buffer()
    # check_load_categorical_statistics(stat_path='../debug/negate_zero_rwrd/5_prepare_statistics/critic_stat')
    check_plot_categorical_statistics(stat_path='../debug/negate_zero_rwrd/5_prepare_statistics/critic_stat')
    # check_plot_statistics()
    # check_model_copying()
    pass
