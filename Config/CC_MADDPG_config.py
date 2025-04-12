"""
MADDPG
[1] Udacity code
[2] Sutton, Barto book: "Reinforcement Learning An Introduction", Chepter 13, Actor-Critic Episodic Continuous Control
[3] Lowe at all. Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
"""
from collections import OrderedDict

config_train = OrderedDict()
config_train['buffer_size']     = int(1e5)      # replay buffer size
config_train['batch_size']      = 256           # minibatch size
config_train['gamma']           = [0.95, 0.95]  # discount factor
config_train['tau']             = [1e-2, 1e-2]  # for soft update of target parameters
config_train['lr_actor']        = [1e-4, 1e-4]  # learning rate of the actor
config_train['lr_critic']       = [3e-4, 3e-4]  # learning rate of the critic
config_train['weight_decay']    = [0, 0]        # L2 weight decay
config_train['actor_update_frequency'] = 0.75   # Udacity GPT: update an actor NN weights frequency per iteration.
config_train['critic_update_frequency'] = 1.25  # Udacity GPT: update a critic NN weights frequency per iteration.

config_train['add_noise']       = [True, True]  # Set True To add random noise during training
config_train['noise_sampling_uniformly'] = [False, False]  # Set True To add random noise sampled uniformly
                                                # Otherwise, Use Normal distribution
config_train['noise_mu']        = [0., 0.]      # noise mean
config_train['noise_theta']     = [0.15, 0.15]  # noise scale factor of (theta - noise_state)
config_train['noise_sigma']     = [0.44, 0.44]  # noise scale factor of (sigma - noise_state)
config_train['noise_sigma_reduction'] = [1.0, 1.0]  # A scale factor to reduce/enlarge sigma during an episode
                                                # to encourage less/more noise in longer episodes.
                                                # Udacity GPT,
                                                # Jonas https://knowledge.udacity.com/questions/65068, and others

config_train['min_epoch_reward_to_collect_experiences']  = 0.06  # Collect an epoch experiments
                                                                 # if the epoch received such reward or more.
# config_train['replay_non_zero_rewards_only']        = False      # Collect experiments with non zeros rewards only.
#                                                                 # Otherwise, collect all experiments
config_train['save_reply_buffer']                   = True      # Set True To save reply buffer
config_train['load_restore_reply_buffer_asis']      = False      # Set True To restore reply buffer
                                                                # with seeds and other metadata
config_train['sample_all_experiences_to_all_agents'] = True     # For example: agent[0] may receive experiences
                                                                # from agents a[0] and a[1] at different times
config_train['replay_sub_buf_imbalance']            = 0.01      # It is maximum unbalance between
                                                                # positive and negative sub-buffers.
                                                                # Which means that a positive sample will not be added
                                                                # to the sub_positive buffer len(sub_positive)
                                                                # greater than
                                                                # (positive_part + imbalance) * (len(sub_negative) + len(sub_negative))
                                                                # Which also means that a negative sample
                                                                # will not be added to the sub_negative buffer
                                                                # If len(sub_negative)  greater than
                                                                # (negative_part +imbalance) * (len(sub_negative) + len(sub_negative))
config_train['save_reply_buffer_frequency']         = 1000      # a frequency (in episodes) to save reply buffer
config_train['replay_batch_positive_rewards_part']  = 0.80       # It is the part of positive reward experiences
                                                                # in a batch
config_train['swap_agents_experience']              = False     # Set True to Swap agent[0] experience
                                                                # with agent[1] experience
config_train['swap_agents_probability']             = 0         # Set 1.0 to swap agents in each sample of an experience
                                                                # Set to a value from 0.0 to 1.0 to swap agents
                                                                # in samples with the probability equals to this value.
config_train['use_this_agent_rewards_part']         = 1.0       # a scale to multiply this agent reward.
config_train['add_other_agent_rewards_part']        = 0.5       # a reward part to add from another agent
                                                                # to this agent reward.
                                                                # If it is 0 then this agent reward is its own reward only.
config_train['negative_rewards_scale']              = 1.0       # To scale negative ReplyBuffer rewards.
config_train['positive_rewards_scale']              = 1.0       # To scale positive ReplyBuffer rewards.
config_train['zero_rewards_to_negative']            = 0         # To change zero reward to negative
                                                                # if another agent receive positive reward
config_train['number_samples_to_start_learning']    = 20000     # Number samples to collect before first learning
config_train['number_samples_before_each_update']   = 100       # Number samples to collect before each call to learning

# ------------------- Debugging ----------------------------
config_train['debug_save_frequency']                = 1.0       # Frequency to save debug info as part from number
                                                                # of all training episodes. For example,
                                                                # if there is scheduled to train during 100 episodes
                                                                # with debug_save_frequency = 0.1,
                                                                # then debug info saved every 100*0.1 = 10 episodes
                                                                # and after last episode
config_train['debug_learning']                      = True     # True to accumulate and print learning debug info.
config_train['clip_critic_grad_norm']               = False     # True to clip critic model gradient norm
config_train['critic_grad_clip_norm']               = 0.01      # Critic is clipping to this maximum of gradient norm
config_train['clip_actor_grad_norm']                = False     # True to clip actor model gradient norm
config_train['actor_grad_clip_norm']                = 0.01      # Actor is clipping to this maximum of gradient norm
config_train['clip_critic_loss']                    = None      # Set [min_val, max_val] to clip critic loss
                                                                # in this range. None otherwise
config_train['clip_actor_loss']                     = None     # Set [min_val, max_val] to clip actor loss
                                                                # in this range. None otherwise

config_train['critic_loss']             = 'sqrt_mse'    # 'mse', 'sqrt_mse' - a critic loss function
config_train['actor_loss']              = 'critic'      # 'critic', 'inverse_critic' - an actor loss function
config_train['model_name']              = 'maddpg_mlp'  # 'maddpg_mlp', 'maddpg_cnn_actor' - model name suffix
config_train['model_fc1_units']         = 400           # Udacity mentors. Number neurones in the first hidden layer of actor and critic NN
config_train['model_fc2_units']         = 300           # Udacity mentors. Number neurones in the second hidden layer of actor and critic NN
config_train['actor_regularization']    = 'DropOut'     # 'DropOut' 'BatchNormalization' Actor Regularization method
config_train['critic_regularization']   = 'DropOut'     # 'No', 'DropOut' 'BatchNormalization' Citic Regularization method
config_train['drop_out_val']            = [0.25, 0.25]  # A percent to 'DropOut' per layer
config_train['init_target_by_local_nn'] = False         # Set True to initialize target NN models by local models
                                                        # at the beginning of any training session.

config_train['agents_to_train']                     = [0, 1]    # [0, 1] - An array of agent IDs to train.
                                                                # It may be [0, 1], [1, 0], [0] or [1]

config_agent = OrderedDict()
config_agent['agent_name']      = 'MADDPG_config'
config_agent['seed']            = 0     # Use Utils.config_utils.set_seed() changes it

config_agent['config_train']    = config_train


# -------------- Checks ------------------


def check_agent_configuration():
    txt = str('{}: Going to train the agent with config. parameters {} with model {}'.
              format('task_name', config_agent['agent_name'], config_agent['config_train']['model_name']))
    print(txt)
    print(f'config_agent=\n{config_agent}')
    print(f'config_train=\n{config_train}')


if __name__ == '__main__':
    check_agent_configuration()

