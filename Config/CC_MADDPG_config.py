"""
MADDPG
[1] Udacity code
[2] Sutton, Barto book: "Reinforcement Learning An Introduction", Chepter 13, Actor-Critic Episodic Continuous Control
[3] Lowe at all. Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
"""
from collections import OrderedDict

config_train = OrderedDict()
config_train['buffer_size']     = int(1e6)      # replay buffer size
config_train['batch_size']      = 128           # minibatch size
config_train['gamma']           = 0.95          # discount factor
config_train['tau']             = 1e-2          # for soft update of target parameters
config_train['lr_actor']        = 1e-4          # learning rate of the actor
config_train['lr_critic']       = 3e-4          # learning rate of the critic
config_train['weight_decay']    = 0             # L2 weight decay
# config_train['weight_update_freq'] = 10         # update target NN weights every 'weight_update_freq'
#                                                 # samples added to the replay buffer
config_train['actor_update_frequency'] = 1.     # Udacity GPT: update an actor NN weights frequency per iteration.
config_train['critic_update_frequency'] = 5     # Udacity GPT: update a critic NN weights frequency per iteration.

config_train['add_noise']       = True          # Set True To add random noise during training
config_train['noise_sampling_uniformly'] = True # Set True To add random noise sampled uniformly
                                                # Otherwise, Use Normal distribution
config_train['noise_mu']        = 0.            # noise mean
config_train['noise_theta']     = 0.15          # noise scale factor of (mu - noise_state)
config_train['noise_sigma']     = 20            # noise scale factor of (mu - noise_state)
config_train['noise_sigma_reduction'] = 1.0     # A scale factor to reduce/enlarge sigma during an episode
                                                # to encourage less/more noise in longer episodes.
                                                # Udacity GPT,
                                                # Jonas https://knowledge.udacity.com/questions/65068, and others

config_train['replay_non_zero_rewards_only']        = False      # Collect experiments with non zeros rewards only.
                                                                # Otherwise, collect all experiments
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
config_train['replay_batch_negative_rewards_part']  = 0.10      # It is the part of negative reward experiences
                                                                # in a batch
                                                                # Other part contains zero reward experiences
config_train['use_this_agent_rewards_part']         = 1.0       # a scale to multiply this agent reward.
config_train['add_other_agent_rewards_part']        = 0.5       # a reward part to add from another agent
                                                                # to this agent reward.
                                                                # If it is 0 then this agent reward is its own reward only.
config_train['number_samples_to_start_learning']    = 60000     # Number samples to collect before learning
# config_train['amount_batches_to_start_learning']    = 1.        # Minimum samples to collect (in buffer size)
#                                                                 # to start learning

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
config_train['actor_regularization']    = 'DropOut'  # 'DropOut' 'BatchNormalization' Actor Regularization method
config_train['critic_regularization']   = 'DropOut'          # 'No', 'DropOut' 'BatchNormalization' Citic Regularization method
config_train['drop_out_val']            = 0.25          # A percent to 'DropOut'


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

