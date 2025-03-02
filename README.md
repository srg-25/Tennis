# Tennis - M A D D P G code 

## Algorithm - Main Parts

[1] Main parts are from the U D A C I T Y  D D P G code

[2] The article Lowe at all. Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments

This is the main parts description only. See more details in the python source and configuration files. 

### Classes

```jupyterpython
class Environment_Agent
    'This is a class which connect between environment and an environment agnostic agent algorithm'
    # .... Some methods ....
```

```jupyterpython
class Agent:
    """Environment agnostic agent."""

    # .... Some methods ....
```

```jupyterpython
class Actor(nn.Module):
    """Actor (Deterministic Policy Action) Model."""
```

```jupyterpython
class Critic(nn.Module):
    """Critic (State-Action Value) Model."""

```

### Main training method of Environment_Agent class

```jupyterpython
import numpy as np
from collections import namedtuple, deque, OrderedDict

class Environment_Agent:
    # .... Some methods ....
    
    def train(self, n_episodes=1000, max_t=10000, print_every=100,
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
        
        # .... Some code ....
        ep_scores = []                                      # to collect scores of episodes
        ep_max_scores = []                                  # maximum episode score of two agents
        ep_max_scores_deque = deque(maxlen=print_every)     # maximum of agent episode scores "list"
                                                            # of 100 last episodes
            
        i_episode = 0                                       # Episode '0' is used
                                                            # to accumulate samples BEFORE learning.
        i_iteration = 0                                     # Number of time steps
                                                            # begin from episode '1'.
                                                            # It is used to decide
                                                            # which network to update on an iteration
        while i_episode < n_episodes:
            # Accumulate enough samples to start learning and then learn n_episodes
            # if all(np.array(self.memory.internal_buffers_length()) > self.min_samples_to_start_learn()):
            if len(self.memory) > self.min_samples_to_start_learn():
                i_episode += 1
            else:  # Reset training counters on experience accumulation stage
                i_iteration = 0
                n_next_actor_update = 1
                n_next_critic_update = 1
            
            # .... Some code ....
            
            env_info = self.env.reset(train_mode=True)[self.brain_name]         # To train set train_mode=True.
            state = env_info.vector_observations                                # get first state (for each agent)
            self.reset()
            ep_score = np.zeros(self.num_agents)                                # Initialize a score of this episode
            for t in range(max_t):                                              # update overall number of iterations
                if i_episode > 0:                                               # over training episodes
                    i_iteration += 1                                            # e.g., when i_episode > 0
                action = self.act(state, add_noise=True)
                action = np.clip(action, -1, 1)                                 # all actions between -1 and 1
                env_info = self.env.step(action)[self.brain_name]               # send all actions to tne environment
                next_state = env_info.vector_observations                       # get next states of all parallel agents
                reward = env_info.rewards                                       # get reward (for each agent)
                done = env_info.local_done                                      # see if episode finished
    
                # .... Some code ....
            
                n_next_critic_update, n_next_actor_update = self.step(state, action, reward, next_state, done,
                          i_iteration, n_next_critic_update, n_next_actor_update, save_dir=save_dir)
                state = next_state
                ep_score += np.array(reward)                                    # update the score of this episode
                self.scale_agent_noise()                                        # scale an agent noise range
                                                                                # by noise_sigma_reduction
                if np.any(done):                                                # Break the episode loop if it is done
                    # .... Some code ....
                    break
            
            # -------------- After each episode -------------
            ep_scores.append(ep_score)                                          # add this episode score to the list
            episode_max_score = np.max(ep_score)                                # calculate maximum between two agents
            ep_max_scores.append(episode_max_score)                             # collect agents max episode score
            ep_max_scores_deque.append(episode_max_score)                       # add the maximum score to the window
        
            # .... Some code ....
            
         return ep_scores, ep_max_scores
```

#### A part of the Environment_Agent.__init__ method which initializes Agent objects

```jupyterpython
class Environment_Agent:
    def __init__(self, env, brain_name, env_seed, cfg, logger, device):
        # .... Some Code ....
        self.agent = [Agent(state_size=self.state_size, action_size=self.action_size,
                            agent_id=agent_id, n_agents=self.num_agents,
                            cfg=self.cfg, logger=self.logger, device=self.device)
                      for agent_id in range(self.num_agents)]
        # .... Some Code ....
        pass
```

### Calculate action in the train method above

#### Environment_Agent.act methode

```jupyterpython
import numpy as np

class Environment_Agent:
    # .... Some methods ....
    
    def act(self, state, add_noise=True):
        """ Perform action by each agent """
        actions = np.array([self.agent[i].act(state[i], add_noise=add_noise) for i in range(self.num_agents)])
        return actions
```

####  Agent.act method

```jupyterpython
class Agent:
    # .... Some methods ....

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
```

### Environment_Agent.step used in the Environment_Agent.train method above

```jupyterpython
import numpy as np

class Environment_Agent:
    # .... Some methods ....

    def step(self, state,  action, reward, next_state, done, i_iteration, n_next_critic_update, n_next_actor_update, save_dir):
        """
        Store experience in replay memory, and use random sample from buffer to learn each agent.
        Params
        ======
            state: observations from all agents
            action: actions from all agents
            reward: rewards from all agents
            next_state: next observations from all agents
            done: is an agent done?
            i_iteration (int) : an iteration counter (begin from i_episode == 1)
            n_next_critic_update (int): next critic update number
            n_actor_update (int); next actor update number
            save_dir (str): A folder to save data in
        """
        # ... Some code ...

        # Store experience / reward to the a Replay Buffer self.memory.
        # The self.memory is an object of class CategoricalReplayBuffers
        # Which contains three CentralizedReplayBuffer class objects:
        # memory_positive_rewards, memory_negative_rewards and memory_zero_rewards. 
        # CentralizedReplayBuffer class is a changed version of Udacity's ReplayBuffer class.  
        # CategoricalReplayBuffers.add method able to collect  
        # positive, negative and zero reward samples with predefined proportions described by in a configuration dictionary. 
        # For example: 75% positive, 20% negative and 5% zero reward samples.
        
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.min_samples_to_start_learn():
            if self.save_pretrained_reply_buffer:
                self.save_pretrained_reply_buffer = False  # Save on time per training
                self.memory.save(os.path.join(save_dir, self.pretrained_reply_buffer_path))
            episode_done = np.any(done)  # True if it is end of the episode
            n_next_critic_update, n_next_actor_update = \
                self.learn(i_iteration, n_next_critic_update, n_next_actor_update, episode_done)
        return n_next_critic_update, n_next_actor_update
```

### The Environment_Agent.learn method used in the step method above

```jupyterpython
import numpy as np

class Environment_Agent:
    # .... Some methods ....

    def learn(self, i_iteration, n_next_critic_update, n_next_actor_update, episode_done):
        """
        This method used To call agent's actual learning procedures:
            First, this method determines by bellow conditions if critic or actor should be optimized  :
                Update(optimize) a critic weights every `critic_update_frequency` per time step (when i_episode>=1).
                Update(optimize) an actor weights every `actor_update_frequency` per time step (when i_episode>=1).
            Then, if an update needed:
                --> It samples randomly a batch of  experiences from the Replay Buffer.
                --> Prepare (modify) experiences specifically for each agent critic's or actor's model 
                as described in the article [2].
                --> Calls actual learning procedure method of Agent class with these experiences.
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
            i_iteration (int): a number of iterations from i_episode==1.
            n_next_critic_update (int): next critic update number.
            n_next_actor_update (int): next actor update number.
            episode_done (bool): True if the 'i_episode' finished. NOT USED CURRENTLY
        Return: n_next_critic_update, n_next_actor_update
        """
        update_critic_nn = n_next_critic_update <= self.critic_update_frequency * i_iteration
        update_actor_nn = n_next_actor_update <= self.actor_update_frequency * i_iteration
        dbg_max_updates = 10000  # It is used To prevent endless looping
        while update_critic_nn or update_actor_nn:
            dbg_max_updates -= 1
            if dbg_max_updates <= 0:
                self.logger.warning(f'\n\n ------------ !!! Learning: Endless loop detected !!! ------------- \n\n')
            experiences = self.memory.sample()
            if update_critic_nn:
                target_actions_on_next_state = self.create_critic_learn_tensors(experiences, self.agent, self.device)
                for i in range(self.num_agents):
                    self.agent[i].learn_critic(experiences, target_actions_on_next_state=target_actions_on_next_state)
                n_next_critic_update += 1  # update global training counter
            if update_actor_nn:
                local_actions_on_current_state = self.create_actor_learn_tensors(experiences, self.agent, self.device)
                for i in range(self.num_agents):
                    self.agent[i].learn_actor(experiences, local_actions_on_current_state=local_actions_on_current_state[i])
                n_next_actor_update += 1  # update global training counter
            # Update the loop end/continue expressions
            update_critic_nn = n_next_critic_update <= self.critic_update_frequency * i_iteration
            update_actor_nn = n_next_actor_update <= self.actor_update_frequency * i_iteration

        return n_next_critic_update, n_next_actor_update
```

### Agent class learn_critic and learn_actor methods used by the Environment_Agent.learn method above

#### Agent.learn_critic method 

```jupyterpython
class Agent:
    # .... Some methods ....

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

        self.critic_target.eval()
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_states.reshape((next_states.shape[0], -1)),
                                                target_actions_on_next_state)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards[:, self.agent_id, :] + (self.gamma * Q_targets_next * (1 - dones[:, self.agent_id, :]))

        Q_expected = self.critic_local(states.reshape((states.shape[0], -1)),
                                       actions.reshape((actions.shape[0], -1)))
        # ... Some Code ...

        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # ... Some Code ...    

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # ... Some Code ...    

        self.critic_optimizer.step()
        self.soft_update(self.critic_local, self.critic_target, self.tau)  # update critic target network
```

#### Agent.learn_actor

```jupyterpython
class Agent:
    # .... Some methods ....

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

        # ... Some Code ...    

        actor_loss = -pre_loss

        # ... Some Code ...    

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        # ... Some Code ...    

        self.actor_optimizer.step()
        self.soft_update(self.actor_local, self.actor_target, self.tau)  # update actor target networks

```

### Helper methods which prepare tensors used by learn methods above

#### Environment_Agent.create_critic_learn_tensors

```jupyterpython
import numpy as np

class Environment_Agent:
    # .... Some methods ....

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
```

#### Agent.target_action method   
 
```jupyterpython
class Agent:
    # .... Some methods ....

    def target_action(self, state):
        """
        Returns actions for given observation as per target policy.
        :param state: this agent observation tensor
        """
        self.actor_target.eval()
        with torch.no_grad():
            action = self.actor_target(state).cpu().data.numpy()
        return action

```

#### Environment_Agent.create_actor_learn_tensors method

```jupyterpython
import numpy as np

class Environment_Agent:
    # .... Some methods ....

    @staticmethod
    def create_actor_learn_tensors(experiences: object, agents: object, device: object):
        """
        Used to create tensors for learn function as in the article [2] algorithm. 
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

        # A code below has a technical problem. It is also non valid algoritmicaly. See the article [2].
        #
        # local_action_on_current_state_list = [agents[i].local_action(current_states[:, i, :])
        #                                       for i in range(num_agents)]
        # local_action_on_current_state = torch.cat(local_action_on_current_state_list, dim=1)
        #
        # Techical problem descussion: 
        # I think that problem is that the for j != i we calculate the j'-clone-tensor gradient at 'i' iteration,
        #  But the j'-clone-tensor still connected to 'j'-optimizer.
        #  Therefore, when we use it in 'i' iteration on 'i'-loss calculation
        #  it calculates gradient, which registered in 'j'-optimizer.
        #  Therefore, when there is the 'j'-iteration, the 'j'-optimizer has deleted gradient from this 'j'-clone tensor
        #  Therefore if j != i then convert the tensor to numpy and then reconvert it to tensor.
        #  This will break it's connection to the optimizer J.
        
        # Below is a code which I think matches the article [2]
        local_action_on_current_state = []
        for i in range(num_agents):
            las_i = []
            for j in range(num_agents):
                a_j = actions[:, j, :] if j != i else agents[i].local_action(current_states[:, i, :])
                las_i.append(a_j)
            local_action_on_current_state.append(torch.cat(las_i, dim=1))

        return local_action_on_current_state

```

#### Agent.local_action method

```jupyterpython
class Agent:
    # .... Some methods ....

    def local_action(self, state):
        """
        Returns actions for given observation as per local policy.
        :param state: this agent observation tensor
        """
        action = self.actor_local(state)
        return action
```

#### Environment_Agent.soft_update method (from [1])

```jupyterpython
import numpy as np

class Environment_Agent:
    # .... Some methods ....

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
```

### Networks

#### Actor

```jupyterpython
Actor(
  (fc1): Linear(in_features=24, out_features=400, bias=True)
  (regular1): Sequential(
    (0): ReLU()
    (1): Dropout(p=0.25, inplace=False)
  )
  (fc2): Linear(in_features=400, out_features=300, bias=True)
  (regular2): Sequential(
    (0): ReLU()
    (1): Dropout(p=0.25, inplace=False)
  )
  (fc3): Linear(in_features=300, out_features=2, bias=True)
  (fc3_activation): Tanh()
)
```

#### Critic

```jupyterpython

Critic(
  (fcs1): Linear(in_features=48, out_features=400, bias=True)
  (regular1): Sequential(
    (0): ReLU()
    (1): Dropout(p=0.25, inplace=False)
  )
  (fc2): Linear(in_features=404, out_features=300, bias=True)
  (regular2): Sequential(
    (0): ReLU()
    (1): Dropout(p=0.25, inplace=False)
  )
  (fc3): Linear(in_features=300, out_features=1, bias=True)
)
```

## To configure and run the code

### To configure the algorithm flow

#### Change default configuration parameters in main_train.py train_env_agents(...) function

```jupyterpython
def train_env_agents(...):
    
    # ... Some Code ...

    cfg['config_train']['buffer_size']      = int(1e6)  # replay buffer size
    cfg['config_train']['batch_size']       = 128       # minibatch size
    cfg['config_train']['gamma']            = 0.95      # discount factor
    cfg['config_train']['tau']              = 1e-2      # scale factor for soft update of target parameters
    cfg['config_train']['lr_actor']         = 1e-4      # learning rate of the actor
    cfg['config_train']['lr_critic']        = 3e-4      # learning rate of the critic
    cfg['config_train']['weight_decay']     = 0         # L2 weight decay
    cfg['config_train']['actor_update_frequency']  = 0.75  # Udacity GPT: update an actor NN weights frequency per iteration.
    cfg['config_train']['critic_update_frequency'] = 1.25  # Udacity GPT: update an critic NN weights frequency per iteration

    cfg['config_train']['replay_batch_positive_rewards_part'] = 0.75  # Udacity GPT: It is the part of positive reward experiences in a batch
    cfg['config_train']['replay_batch_negative_rewards_part'] = 0.20  # Udacity GPT: It is the part of negative reward experiences in a batch.
                                                                      # zero reward experiences part is 0.05 = 1. - 0.75 - 0.20
    cfg['config_train']['sample_all_experiences_to_all_agents'] = False  # If True: agent[0] may receive experiences
                                                                    # from agents a[0] and a[1] at different times.
                                                                    # The same is regard agent[1].
    cfg['config_train']['save_reply_buffer']                = True  # Set True To save reply buffer
    cfg['config_train']['load_restore_reply_buffer_asis']   = False # Set True To restore reply buffer
                                                                    # with seeds and other metadata
    cfg['config_train']['replay_sub_buf_imbalance']         = 0.01  # It is a maximum unbalance between
                                                                    # positive and negative sub-buffers.
                                                                    # Which means that a positive sample will not be added
                                                                    # to the sub_positive buffer if length of sub_positive
                                                                    # greater than
                                                                    # (positive_part + imbalance) * (len(sub_positive) + len(sub_negative) + len(sub_zero))
                                                                    # Which also means that a negative sample
                                                                    # will not be added to the sub_negative buffer
                                                                    # If len(sub_negative)  greater than
                                                                    # (negative_part +imbalance) * (len(sub_positive) + len(sub_negative) + len(sub_zero))
    cfg['config_train']['add_other_agent_rewards_part'] = 0.25      # Another agent reward part to add to this agent reward
    cfg['config_train']['use_this_agent_rewards_part'] = 1.         # A factor to scale this agent reward.

    cfg['config_train']['number_samples_to_start_learning'] = 2000  # Number samples to collect before learning
    cfg['config_train']['noise_sampling_uniformly'] = False         # Set True To add random noise sampled uniformly.
                                                                    # Otherwise, Use Normal distribution
    cfg['config_train']['noise_mu']         = 0.                    # noise mean
    cfg['config_train']['noise_theta']      = 0.15                  # noise scale factor of (mu - noise_state)
    cfg['config_train']['noise_sigma']      = 0.44                  # noise sigma
    cfg['config_train']['noise_sigma_reduction'] = 0.95             # To reduce sigma during an episode
                                                                    # to encourage less noise in longer episodes.
                                                                    # from Udacity GPT,
                                                                    # Jonas from https://knowledge.udacity.com/questions/65068, and others
                                                                    
                                                                    # Values below are from https://knowledge.udacity :    
    cfg['config_train']['model_fc1_units']       = 400              # Number neurones in the first hidden layer of actor and critic NN
    cfg['config_train']['model_fc2_units']       = 300              # Number neurones in the second hidden layer of actor and critic NN
    cfg['config_train']['actor_regularization']  = 'DropOut'        # 'DropOut', 'BatchNormalization' Actor Regularization method
    cfg['config_train']['critic_regularization'] = 'DropOut'        # 'No', 'DropOut' 'BatchNormalization' Citic Regularization method
    cfg['config_train']['drop_out_val']          = 0.25             # A percent to 'DropOut'. Udacity knowledge
```

Config/CC_MADDPG_config.py file contains all configuration parameters.

n_episodes is a number of episodes to train. You may change it in the main_train.py file.
For Example
```jupyterpython
main_train.py

if __name__ == '__main__':

    # ... Some Code ...

    n_episodes = 500

    # .. Some Code ...
```

### To run the code in drlnd conda environment
    type the command bellow from the code root folder:
```jupyterpython
(drlnd) python.exe main_train.py
```
