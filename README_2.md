Bellow is a short description where the code handles experience storage, noise sampling, and training logic

Experience accumulated by a class CategoricalReplayBuffers object in the Agent/ddpg_env_agent.py file. This object is in the Environment_Agent class:
self.memory = CategoricalReplayBuffers(...)

self.memory is filled in the Environment_Agent. step(...) method in the previous version:
```python
self.memory.add(state, action, reward, next_state, done)  
```

But I added in the current version self.epoch_memory  to the Environment_Agent  class which collects experiencies during an episode in the same step(...) method instead of the self.memory class. When the episode done the self.epoch_memory exports experiencies to the self.memory in the Environment_Agent.train(...) method  if it collected enough rewards during this episode:

```python
   if np.any(done):                                                # Break the episode loop if it is done
		# Fill the "update" memory by the episode experience if the episode reward is reach enough:
		if np.max(ep_score) > self.min_epoch_reward_to_collect_experiences:
			n_new_samples += self.export_epoch_memory_to_update_memory()
		else:
			self.epoch_memory.reset()

```
The CategoricalReplayBuffers class contains 3 sub-buffers of the class CentralizedReplayBuffer (updated version):
self.memory_positive_rewards which contains experiences where an one agent has positive reward and the other has zero reward.
self.memory_pos_neg_rewards	which contains experiences where both agents have positive or negative rewards.
self.memory_negative_rewards which contains experiences where an one agent has negative reward and the other has zero reward.

When networks should be optimised self.memory samples a batch of experiences from these 3 sub-bufers in the CategoricalReplayBuffers.sample() method:

```python
experiences_positive = self.memory_positive_rewards.sample()
...
experiences_pos_neg  = self.memory_pos_neg_rewards.sample()
...
experiences_negative = self.memory_negative_rewards.sample()

```

It occures in the Environment_Agent.learn(...) method which is called from the Environment_Agent.step(...) method:

```python

	def learn(...):
		
		# ... Other code ...
		
        while update_critic_nn or update_actor_nn:
            dbg_max_updates -= 1
            if dbg_max_updates <= 0:
                self.logger.warning(f'\n\n ------------ !!! Learning: Endless loop detected !!! ------------- \n\n')
            experiences = self.memory.sample()

		# ... Other code ...

```	

Then Environment_Agent.learn(...) method calls critic or actor learn methods to optimize its networks:

```python

	def learn(...):
		
		# ... Other code ...

			if update_critic_nn:
				# ...
				target_actions_on_next_state = self.create_critic_learn_tensors(experiences, self.agent, self.device)
				# ...
				for i in self.agents_to_train:
					self.agent[i].learn_critic(experiences, target_actions_on_next_state=target_actions_on_next_state)
			

		# ... Other code ...

```	
```python

	def learn(...):
		
		# ... Other code ...

            if update_actor_nn:
				# ...
                local_actions_on_current_state = self.create_actor_learn_tensors(experiences, self.agent, self.device)
				# ...
                for i in self.agents_to_train:
                    self.agent[i].learn_actor(experiences, local_actions_on_current_state=local_actions_on_current_state[i])
			

		# ... Other code ...

```	

Noise handled by the Agent.noise member of the class OUNoise. I use standard Ornstein-Uhlenbeck process e.g. noise sampling by Normal destribution:

```python

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state

		# ... Other code ...

        else:  # Normal
            dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)

        self.state = x + dx
		
		return self.state

``` 

The nois OUNoise.seample(...) method called by the Agent.act(...) method:

```python

def act(self, state, add_noise=True):

		# ... Other code ...
        if add_noise:
            noise = self.noise.sample()
            action += noise
		# ... Other code ...

```

The Agent.act(...) method colled by Environment_Agent.act(...) method in the Environment_Agent.train(...) method:

```python

    def train(...)
	
		# ... Other code ...

            for t in range(max_t):                                              # update overall number of iterations
                if i_episode > 0:                                               # over training episodes
                    i_iteration += 1                                            # e.g., when i_episode > 0
                action = self.act(state, add_noise=self.add_noise)              # Train with or without noise
                action = np.clip(action, -1, 1)                                 # all actions between -1 and 1

		# ... Other code ...

```

```python

    def act(self, state, add_noise=True):
        """ Perform action by each agent """
                                               # It is batched
        actions = np.array([self.agent[i].act(state[i], add_noise=add_noise[i]) for i in range(self.num_agents)])
        return actions

```

Training logic performed in Environment_Agent and Agent classes. 
Where the Environment_Agent class handles interaction with the Tennis environment, loops throw episodes and iteration during an episode in the Environment_Agent.train method and managed Reply Buffer objects. 

The Environment_Agent contains also two agent objects of the class Agent in the Environment_Agent.agent list. The Environment_Agent used as a manager to call the Agent class methods in the Environment_Agent methods learn(...), act(...), hard_update(...), reset(...), scale_agent_noise(...)

The Environment_Agent has also two static mentods to create experiences tensors for Agent.learn_critic(...) and Agent.learn_actor(...) methods which perform agents critic and actor networks optimisation.