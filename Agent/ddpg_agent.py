"""
DDPG Agent
Main parts are from the Udacity DDPG code which I refactored to work with multy agent environment.
[3] The article Lowe at all. Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
"""
import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from Model.model_mlp import Actor, Critic
from Model.model_conv import ActorConv
from Utils.ddpg_utils import dbg_get_gradient_norm

from Model.model_mlp import dclamp

# ---------------- Agent -------------------------------


class Agent:
    """Environment agnostic agent."""

    critic_loss_mse         = 'mse'
    critic_loss_sqrt_mse    = 'sqrt_mse'

    actor_loss_critic       = 'critic'
    actor_loss_inverse_critic   = 'inverse_critic'

    maddpg_mlp_name     = 'maddpg_mlp'
    maddpg_cnn_actor    = 'maddpg_cnn_actor'

    def __init__(self, state_size, action_size, agent_id, n_agents, cfg, logger, device):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            agent_id (int): an agent ID
            n_agents (int): number of parallel agents in an environment
            cfg (dict): configuration
            logger: to log info and debug data
            device: GPU or CPU
        """
        self.state_size = state_size
        self.action_size = action_size

        self.cfg = cfg['config_train']
        self.logger = logger
        self.device = device

        self.agent_id = agent_id
        self.n_agents = n_agents

        self.lr_actor = self.cfg['lr_actor']
        self.lr_critic = self.cfg['lr_critic']
        self.buffer_size = self.cfg['buffer_size']
        self.batch_size = self.cfg['batch_size']
        self.gamma = self.cfg['gamma']
        self.tau = self.cfg['tau']
        self.weight_decay = self.cfg['weight_decay']

        self.is_uniform_noise_sample = self.cfg['noise_sampling_uniformly']
        self.add_noise = self.cfg['add_noise']
        self.noise_mu = self.cfg['noise_mu']
        self.noise_theta = self.cfg['noise_theta']
        self.noise_sigma = self.cfg['noise_sigma']
        self.noise_sigma_reduction = self.cfg['noise_sigma_reduction']

        self.logger.info(f'{type(self).__name__}.__init__(): agent_{self.agent_id} '
                         f'lr_actor = {self.lr_actor}, lr_critic = {self.lr_critic}, '
                         f'buffer_size = {self.buffer_size}, batch_size = {self.batch_size}, gamma = {self.gamma}, '
                         f'tau = {self.tau}, weight_decay = {self.weight_decay}')

        self.critic_loss    = self.cfg['critic_loss']
        self.actor_loss     = self.cfg['actor_loss']
        self.model_name     = self.cfg['model_name']
        self.fc1_units      = self.cfg['model_fc1_units']
        self.fc2_units      = self.cfg['model_fc2_units']
        self.actor_regularization = self.cfg['actor_regularization']
        self.critic_regularization = self.cfg['critic_regularization']
        self.drop_out_val = self.cfg['drop_out_val']

        # Actor Network (w/ Target Network) for an agent
        if self.model_name == self.maddpg_mlp_name:
            self.actor_local = Actor(state_size, action_size, fc1_units=self.fc1_units, fc2_units=self.fc2_units,
                                     regularization=self.actor_regularization, drop_out_val=self.drop_out_val).to(self.device)
            self.logger.info(f'{type(self).__name__}.__init__(): agent_{self.agent_id} actor:\n{self.actor_local}')

            self.actor_target = Actor(state_size, action_size, fc1_units=self.fc1_units, fc2_units=self.fc2_units,
                                     regularization=self.actor_regularization, drop_out_val=self.drop_out_val).to(self.device)
        elif self.model_name == self.maddpg_cnn_actor:
            self.actor_local = ActorConv(state_size, action_size).to(self.device)
            self.logger.info(f'{type(self).__name__}.__init__(): agent_{self.agent_id} actor:\n{self.actor_local}')

            self.actor_target = ActorConv(state_size, action_size).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network) for an agent which get states and actions from all agents
        self.critic_local = Critic(state_size*self.n_agents, action_size*self.n_agents,
                                   fcs1_units=self.fc1_units, fc2_units=self.fc2_units,
                                   regularization=self.critic_regularization, drop_out_val=self.drop_out_val).to(self.device)
        self.logger.info(f'{type(self).__name__}.__init__(): agent_{self.agent_id} critic:\n{self.critic_local}')

        self.critic_target = Critic(state_size*self.n_agents, action_size*self.n_agents,
                                    fcs1_units=self.fc1_units, fc2_units=self.fc2_units,
                                    regularization=self.critic_regularization, drop_out_val=self.drop_out_val).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic,
                                           weight_decay=self.weight_decay)

        self.noise = OUNoise(action_size, self.noise_mu, self.noise_theta, self.noise_sigma,
                             self.is_uniform_noise_sample, self.noise_sigma_reduction)

        self.clip_actor_grad_norm   = self.cfg['clip_actor_grad_norm']
        self.actor_grad_clip_norm   = self.cfg['actor_grad_clip_norm']
        self.clip_critic_grad_norm  = self.cfg['clip_critic_grad_norm']
        self.critic_grad_clip_norm  = self.cfg['critic_grad_clip_norm']
        self.clip_critic_loss       = self.cfg['clip_critic_loss']
        self.clip_actor_loss        = self.cfg['clip_actor_loss']

        # -------------- Debugging ---------------
        self.dbg_actor_gradient_norm = []
        self.dbg_actor_loss = []

        self.dbg_critic_gradient_norm = []
        self.dbg_critic_loss = []

    def act(self, state, add_noise=True):
        """
        Returns actions for given state as per current (local) policy.
        :param state: this agent observation
        :param add_noise: set True to add a noise to the action
        """
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            noise = self.noise.sample()
            action += noise
        return action

    def local_action(self, state):
        """
        Returns actions for given observation as per local policy.
        :param state: this agent observation tensor
        """
        action = self.actor_local(state)
        return action

    def target_action(self, state):
        """
        Returns actions for given observation as per target policy.
        :param state: this agent observation tensor
        """
        self.actor_target.eval()
        with torch.no_grad():
            action = self.actor_target(state).cpu().data.numpy()
        return action

    def reset(self):
        self.noise.reset()

    def learn_critic(self, experiences, target_actions_on_next_state):
        """Update state-action value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            target_actions_on_next_state (Tensor): Next state actions from all agents target models.
                                                   See Environment_Agent.create_critic_learn_tensors method for details
        """
        states, actions, rewards, next_states, dones = experiences
        dbg_learning = self.cfg['debug_learning']

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        # Policy target (e.g. actions_next)  calculation moved to ddpg_env_agent.py

        self.critic_target.eval()
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_states.reshape((next_states.shape[0], -1)),
                                                target_actions_on_next_state)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards[:, self.agent_id, :] + (self.gamma * Q_targets_next * (1 - dones[:, self.agent_id, :]))

        Q_expected = self.critic_local(states.reshape((states.shape[0], -1)),
                                       actions.reshape((actions.shape[0], -1)))
        if self.critic_loss == self.critic_loss_mse:
            critic_loss = F.mse_loss(Q_expected, Q_targets)
        elif self.critic_loss == self.critic_loss_sqrt_mse:
            critic_loss = torch.sqrt(F.mse_loss(Q_expected, Q_targets))
        else:
            assert False

        if self.clip_critic_loss is not None:
            critic_loss = dclamp(critic_loss, self.clip_critic_loss[0], self.clip_critic_loss[1])
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # critic_loss.backward(retain_graph=True)
        if dbg_learning:
            dbg_critic_grad_norm = dbg_get_gradient_norm(self.critic_local.parameters(), max_norm=1, norm_type=2)
            self.logger.debug(f'\ndbg_critic_grad_norm_{self.agent_id} = {dbg_critic_grad_norm}')
            self.dbg_critic_loss.append(critic_loss.detach().to('cpu').item())
            self.dbg_critic_gradient_norm.append(dbg_critic_grad_norm)
        if self.clip_critic_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), max_norm=self.critic_grad_clip_norm, norm_type=2)
            if dbg_learning:  # recalculate the gradient after clipping
                dbg_critic_grad_norm = dbg_get_gradient_norm(self.critic_local.parameters(), max_norm=1, norm_type=2)
                self.logger.debug(f'\ndbg_critic_grad_norm_{self.agent_id}  = {dbg_critic_grad_norm} after clipping '
                                 f'critic_grad_clip_norm = {self.critic_grad_clip_norm}')
        self.critic_optimizer.step()
        self.soft_update(self.critic_local, self.critic_target, self.tau)  # update critic target network

    def learn_actor(self, experiences, local_actions_on_current_state):
        """Update policy parameters using given batch of experience tuples.
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            local_action_on_current_state (Tensor): Current state actions from all agents local models.
                                                    See Environment_Agent.create_actor_learn_tensors method for details
            update_actor_nn (bool): True to update actor parameters
        """
        states, actions, rewards, next_states, dones = experiences
        dbg_learning = self.cfg['debug_learning']
        # Compute actor loss
        pre_loss = self.critic_local(states.reshape((states.shape[0], -1)), local_actions_on_current_state).mean()
        if self.actor_loss == self.actor_loss_critic:
            actor_loss = -pre_loss
        elif self.actor_loss == self.actor_loss_inverse_critic:
            actor_loss = 1. / (pre_loss + np.sign(pre_loss.item())*1e-9)
        if self.clip_actor_loss is not None:
            actor_loss = dclamp(actor_loss, self.clip_actor_loss[0], self.clip_actor_loss[1])

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if dbg_learning:
            dbg_actor_grad_norm = dbg_get_gradient_norm(self.actor_local.parameters(), max_norm=1, norm_type=2)
            self.logger.debug(f'\ndbg_actor_grad_norm_{self.agent_id}  = {dbg_actor_grad_norm}')
            self.dbg_actor_loss.append(actor_loss.detach().to('cpu').item())
            self.dbg_actor_gradient_norm.append(dbg_actor_grad_norm)
        if self.clip_actor_grad_norm:
            # Clip grads of self.actor_local
            # torch.nn.utils.clip_grad_norm_(self.actor_local, max_norm=1.0, norm_type=2)
            # See https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
            # https://www.geeksforgeeks.org/gradient-clipping-in-pytorch-methods-implementation-and-best-practices/
            # torch.nn.utils.clip_grad_value_(self.actor_local.parameters(), clip_value=0.1)
            # torch.nn.utils.clip_grad_value_(self.actor_local.parameters(), clip_value=grad_clip_val)
            torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), max_norm=self.actor_grad_clip_norm, norm_type=2)
            if dbg_learning:  # recalculate the gradient after clipping
                dbg_actor_grad_norm = dbg_get_gradient_norm(self.actor_local.parameters(), max_norm=1, norm_type=2)
                self.logger.debug(f'\ndbg_actor_grad_norm_{self.agent_id} = {dbg_actor_grad_norm} after clip_grad_norm_; '
                                 f'actor_grad_clip_norm = {self.actor_grad_clip_norm}')
        self.actor_optimizer.step()
        self.soft_update(self.actor_local, self.actor_target, self.tau)  # update actor target networks

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def load_models(self, local_actor_path='checkpoint_actor.pth', local_critic_path='checkpoint_critic.pth'
                    , target_actor_path=None, target_critic_path=None):
        """Load local and target models. Target equals to local if target paths are None."""
        self.logger.info(f'Going to load: '
                         f'\nlocal_actor_path_{self.agent_id} = \n{local_actor_path} '
                         f'\nlocal_critic_path_{self.agent_id} = \n{local_critic_path}')
        try:
            if local_critic_path is None:   # playing
                self.actor_local.load_state_dict(torch.load(local_actor_path))
            else:                           # training
                self.actor_local.load_state_dict(torch.load(local_actor_path))
                self.actor_target.load_state_dict(torch.load(local_actor_path if target_actor_path is None
                                                             else target_actor_path))
                self.critic_local.load_state_dict(torch.load(local_critic_path))
                self.critic_target.load_state_dict(torch.load(local_critic_path if target_critic_path is None
                                                              else target_critic_path))
        except FileNotFoundError:
            self.logger.info(f'File not found: '
                             f'\nlocal_actor_path_{self.agent_id} = \n{local_actor_path} '
                             f'\nlocal_critic_path_{self.agent_id} = \n{local_critic_path}')
            raise ValueError('Cannot load weights to actor or critic model - a critic or actor weights file not found.')
        except IsADirectoryError:
            self.logger.info(f'A directory error: '
                             f'\nlocal_actor_path_{self.agent_id} = \n{local_actor_path} '
                             f'\nlocal_critic_path_{self.agent_id} = \n{local_critic_path}')
            raise ValueError('Cannot load weights to actor or critic model - a directory error.')
        except:
            self.logger.info(f'Cannot load: '
                             f'\nlocal_actor_path_{self.agent_id} = \n{local_actor_path} '
                             f'\nlocal_critic_path_{self.agent_id} = \n{local_critic_path}')
            raise ValueError('Cannot load weights to actor or critic model.')

        self.logger.info(f'Models loaded: from'
                         f'\nlocal_actor_path_{self.agent_id} = \n{local_actor_path} '
                         f'\nlocal_critic_path_{self.agent_id} = \n{local_critic_path}')

    def get_actor_type(self):
        return self.actor_local.__class__

    def get_critic_type(self):
        return self.critic_local.__class__


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2, noise_sampling_uniformly=False, noise_sigma_reduction=0.97):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_init = copy.copy(sigma)
        self.noise_sigma_reduction = noise_sigma_reduction
        self.noise_sampling_uniformly = noise_sampling_uniformly
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        self.sigma = copy.copy(self.sigma_init)

    def scale_sigma(self):
        self.sigma *= self.noise_sigma_reduction

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = np.zeros_like(x)

        if self.noise_sampling_uniformly:  # Uniform
            dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() - 0.5 for i in range(len(x))])
        else:  # Normal
            dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)

        self.state = x + dx
        return self.state
