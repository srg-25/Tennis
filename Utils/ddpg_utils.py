"""
---------------- DDPG Utils -----------------
"""

import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

import os
# import os_io

from Utils.logger_utils import create_logger
# ------------- Debug Utils -----------------------


def dbg_get_gradient_norm(parameters, max_norm=1, norm_type=2):
    """
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=1, norm_type=2)
    :param parameters:
    :return:
    """
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return torch.tensor(0.)
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    device = parameters[0].grad.device

    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm

# ----------------- Plots -------------------------


def plot_window_average_episode_score(max_scores, win_size=100, episode_score_max=0.5, target_average_score=0.5):
    """
    Plot average of average scores per 'win_size' consecutive episodes.
    max_scores ([float]): list of maximum scores of parallel agents  per episode
    win_size (int): number of consecutive max_scores to calculate the average per episodes in the window.
    episode_score_max (float): a minimum of maximum average agent episode scores to be reached to solve the task.
    target_average_score (float): target average over consecutive average agent episode scores in a window
    """
    if len(max_scores) < win_size:
        assert False
    n_scores = len(max_scores)
    average = [np.mean(max_scores[i-win_size:i]) if i >= win_size else 0 for i in range(n_scores)]
    target = [episode_score_max] * n_scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    max_average_score_episode = np.argmax(average)
    max_average_score = average[max_average_score_episode]
    max_average_score = np.max(average)
    print(f'Ep_{max_average_score_episode} mean score = {max_average_score:.4f} '
          f'which is maximum scores average over last {win_size} episodes')
    max_average_score_score = np.max(max_scores)
    if max_average_score_score > target_average_score:
        target_score_episode = np.argwhere(max_scores > target_average_score)[0]
    else:
        target_score_episode = None
    plt.plot(np.arange(len(max_scores)), max_scores, label='episode average score')
    plt.plot(np.arange(len(average)), average, label=f'Average score of {win_size} previous episodes')
    plt.plot(max_average_score_episode, average[max_average_score_episode],'o', label='max average')
    plt.plot(np.arange(len(target)), target, label='maximum score of episodes')
    if target_score_episode is None:
        plt.title(
            f'Scores are less then {target_average_score}. \nEpisode {max_average_score_episode} has {average[max_average_score_episode]:.4f} average score.')
    else:
        plt.title(f'Episode {target_score_episode} is the first one with the score > {target_average_score}. /nEpisode {max_average_score_episode} has {average[max_average_score_episode]:.2f} average score.')
    plt.ylabel(f'average scores per {win_size} episodes')
    plt.xlabel('Episode #')
    plt.legend(loc="center right")

    # plt.show()
    return fig, max_average_score_episode, max_average_score


def plot_episode_score(scores):
    """
    plot parallel agent average, maximum and minimum scores per episode
    scores ([float]): list of scores per episode in shape 'number of episodes' x 'number of parallel agents'
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(scores.max(axis=1), color='orange', ls=':')
    plt.plot(scores.mean(axis=1), color='brown', ls='--')
    plt.plot(scores.min(axis=1), color='blue', ls=':')
    plt.title(f'Average scores of parallel agents per episode')
    plt.legend(['max', 'average', 'min'])
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


def plot_episode_score_details(scores):
    """
    plot all parallel agent scores per episode
    scores ([int]): list of scores per episode in shape 'number of episodes' x 'number of parallel agents'
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = list(mcolors.TABLEAU_COLORS.keys())
    colors = [c[4:] for c in colors]  # Get color names e.g. drop 'tab:' prefix
    n_colors = len(colors)
    n_markers = min(len(Line2D.markers), scores.shape[1])
    markers = list(Line2D.markers.keys())[:n_markers]
    lines = [' ']  # ['-', '--', '-.', ':']
    n_lines = len(lines)
    for s_id in range(scores.shape[1]):
        c = colors[s_id % n_colors]
        m = markers[s_id % n_markers]
        l = lines[s_id % n_lines]
        plt.plot(scores[:, s_id], color=c, marker=m, ls=l)
    plt.title(f'Scores of parallel agents per episode')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


def plot_score_1_dim(score_path=None, title='scores', x_label='Iteration #',
                     file_name_suffix='', show_figure=False, start_pos=0):
    # task_name = 'plot_score_1_dim'
    # print(f'{task_name}: -------------- Start ---------------')
    with open(score_path, 'rb'):
        scores = np.load(score_path)
    scores = scores[start_pos:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(scores, color='blue', ls=':')
    plt.title(title)
    # plt.legend(['max', 'average', 'min'])
    plt.ylabel('Score')
    plt.xlabel(x_label)

    dirname = os.path.dirname(score_path)
    basename_without_ext = os.path.splitext(os.path.basename(score_path))[0]
    plt.savefig(os.path.join(dirname, title + file_name_suffix + '.jpg'))

    if show_figure:
        plt.show()
    # print(f'{task_name}: -------------- End ---------------')


def plot_training_sessions(set_session_ids, set_max_average_score, set_max_average_score_episode, file_name, show_figure=False):
    """
    Plot set training sessions history: it scores and episodes distribution
    Store to a disk.
    Show the plot(optional)
    Params:
    ======
    set_session_ids: (numpy 1d array) the training set sessions
    set_max_average_score: (numpy 1d array) the training set scores
    set_max_average_score_episode: (numpy 1d array) the training set score episodes
    """
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Set Training History')

    ax1.plot(set_session_ids, set_max_average_score)
    # ax1.set_title('Scores')
    # ax1.set_xlabel('# Session')
    ax1.set_ylabel('Scores')

    ax2.plot(set_session_ids, set_max_average_score_episode)
    # ax2.set_title('Episodes of these scores')
    ax2.set_xlabel('# Session')
    ax2.set_ylabel('Episodes')

    plt.savefig(file_name)
    if show_figure:
        fig.show()
    return fig


def plot_training_sessions_history(load_dir, save_dir, logger, show_figure=False):
    session_ids_file_name = os.path.join(load_dir, 'train_session_ids.npy')
    session_scores_file_name = os.path.join(load_dir, 'train_set_max_average_scores.npy')
    session_episodes_file_name = os.path.join(load_dir, 'train_set_max_average_score_episodes.npy')
    try:
        session_ids = np.load(session_ids_file_name)
        session_scores = np.load(session_scores_file_name)
        session_episodes = np.load(session_episodes_file_name)
    except FileNotFoundError:
        logger.info(f'One of a file not found: '
                    f'\n{session_ids_file_name} '
                    f'\n{session_scores_file_name}'
                    f'\n{session_episodes_file_name}')
    except IsADirectoryError:
        logger.info(f'A directory error: \n\'{load_dir}\' ')
    except:
        logger.info(f'Cannot load: '
                    f'\n{session_ids_file_name} '
                    f'\n{session_scores_file_name}'
                    f'\n{session_episodes_file_name}')

    file_name = os.path.join(save_dir, 'set_training_history.jpg')
    plot_training_sessions(session_ids, session_scores, session_episodes, file_name=file_name, show_figure=show_figure)

    pass

# ------------------- Checks ------------------------


def check_plot_average_episode_score(average_scores_path=None, max_win_sz=100, target_average_score=0.5, tail_sz=None,
                                     file_name_suffix='', show_figure=False):
    """
    Plot average score over a predefined number of episodes of a parallel agents average scores.
    :param average_scores_path: path to an '.npy' file with parallel agents average scores per episode.
    :param max_win_sz: maximum number consecutive episodes to average on
    target_average_score: (float) target average over consecutive average agent episode scores in a window.
    :param tail_sz: a length of a tail of the array  to plot. It is used to drop down results on preload episodes
    :param file_name_suffix: (str) additional part to file name to store results
    :param show_figure: (bool) True to show the figure. Otherwise, save it only.
    :return:
    """
    task_name = 'check_plot_average_episode_score'
    print(f'{task_name}: -------------- Start ---------------')
    if average_scores_path is None:
        average_scores_path = 'check_data/average_scores.npy'
    with open(average_scores_path, 'rb'):
        episode_scores = np.load(average_scores_path)
        start = int(0)
        if tail_sz is not None:
            start = int(len(episode_scores) - tail_sz)
        if start < 0:
            raise ValueError(f'Tail size should be nor more then scores array length. '
                             f'Tail size is {tail_sz}, scores array size is {episode_scores}.')
        # print(f'Debug: episode_scores length = {len(episode_scores)}, start={start}')
        # print(f'Debug: episode_scores=\n{episode_scores}')
        scores_to_plot = episode_scores[start:]
        print(f'{np.mean(scores_to_plot):.4f} is average of scores size {len(scores_to_plot)}')
        fig, max_average_score_episode, max_average_score = plot_window_average_episode_score(
            scores_to_plot, win_size=min(max_win_sz, int(len(scores_to_plot))),
            episode_score_max=episode_scores.max(), target_average_score=target_average_score)

    dirname = os.path.dirname(average_scores_path)
    # basename_without_ext = os.path.splitext(os.path.basename(average_scores_path))[0]
    plt.savefig(os.path.join(dirname, "avrg_max_scores" + file_name_suffix + '.jpg'))
    print(f'{task_name}: -------------- End ---------------')
    return max_average_score_episode, max_average_score


def check_plot_episode_score(score_path=None):
    task_name = 'check_plot_episode_score'
    print(f'{task_name}: -------------- Start ---------------')
    if score_path is None:
        score_path = 'check_data/scores.npy'
    with open(score_path, 'rb'):
        episode_scores = np.load(score_path)
        plot_episode_score(episode_scores)
    print(f'{task_name}: -------------- End ---------------')


def check_plot_episode_score_details(score_path=None):
    task_name = 'check_plot_episode_score'
    print(f'{task_name}: -------------- Start ---------------')
    if score_path is None:
        score_path = 'check_data/scores.npy'
    with open(score_path, 'rb'):
        episode_scores = np.load(score_path)
        plot_episode_score_details(episode_scores)
    print(f'{task_name}: -------------- End ---------------')


def check_gradient_loss(dir_name='../', file_name_suffix='', show_figure=False,
                        start_pos=0, model_name='Agent.maddpg_mlp'):
    task_name = 'check_gradient_loss'
    print(f'{task_name}: -------------- Start ---------------')

    for agent_id in range(2):
        # scores_path = f'../result/Parameters_search/alr-1e-2_clr-1e-1_NO-clip-grad/Agent.maddpg_mlp_dbg_actor_loss_{agent_id}.npy'
        scores_path = os.path.join(dir_name, f'Agent.{model_name}_dbg_actor_loss_{agent_id}.npy')
        title = f'Actor_{agent_id}_Loss_via_critic'
        plot_score_1_dim(scores_path, title=title, file_name_suffix=file_name_suffix, show_figure=show_figure)

        scores_path = os.path.join(dir_name, f'Agent.{model_name}_dbg_actor_gradient_norm_{agent_id}.npy')
        title = f'Actor_{agent_id}_gradient_norm'
        plot_score_1_dim(scores_path, title=title, file_name_suffix=file_name_suffix, show_figure=show_figure)

        scores_path = os.path.join(dir_name, f'Agent.{model_name}_dbg_critic_loss_{agent_id}.npy')
        title = f'Critic_{agent_id}_Loss'
        plot_score_1_dim(scores_path, title=title, file_name_suffix=file_name_suffix, show_figure=show_figure)

        scores_path = os.path.join(dir_name, f'Agent.{model_name}_dbg_critic_gradient_norm_{agent_id}.npy')
        title = f'Critic_{agent_id}_gradient_norm'
        plot_score_1_dim(scores_path, title=title, file_name_suffix=file_name_suffix, show_figure=show_figure)

    print(f'{task_name}: -------------- End ---------------')


# def check_plot_training_sessions_history(load_dir, save_dir, logger, show_figure=False):


if __name__ == '__main__':
    # average_scores_path = '../result/Parameters_search/alr-1e-2_clr-1e-1_NO-clip-grad/Environment_Agent.maddpg_mlp_average_max_scores.npy'
    # # average_scores_path = '../../p3_collaboration-competition_MADDPG_bestParameters/Environment_Agent.maddpg_mlp_dbg_average_scores.npy'
    # # average_scores_path = '../Environment_Agent.maddpg_mlp_average_max_scores.npy'
    # check_plot_average_episode_score(average_scores_path, tail_sz=None)
    # # Use below to remove scores during replay buffer preloading.
    # number_trained_epochs = 200
    # check_plot_average_episode_score(average_scores_path, tail_sz=number_trained_epochs)

    # scores_path = '../result/Parameters_search/alr-1e-2_clr-1e-1_a-c-clip-grad/Agent.maddpg_mlp_dbg_actor_gradient_norm.npy'
    # title = 'Actor Gradient'
    # check_plot_episode_score_details(score_path=scores_path)
    # check_plot_episode_score(score_path=scores_path)

    # test_name = '_dbg_save_plot'
    # check_gradient_loss(file_name_suffix=test_name)
    #
    # number_trained_epochs = 200
    # average_scores_path = '../Environment_Agent.maddpg_mlp_average_max_scores.npy'
    # check_plot_average_episode_score(average_scores_path=average_scores_path, tail_sz=number_trained_epochs,
    #                                  file_name_suffix=test_name)

    # start_pos = 1000
    # dir = '../database_replay/(1302, 1316)_0-6_16'
    # # score_path = os.path.join(dir, 'Agent.maddpg_mlp_dbg_critic_gradient_norm_0.npy')
    # # title = f'Loss_via_critic_begin_id{start_pos}'
    # score_path = os.path.join(dir, 'Agent.maddpg_mlp_dbg_critic_loss_1.npy')
    # title = f'Critic_Loss_begin_id{start_pos}'
    # plot_score_1_dim(score_path, title=title, file_name_suffix=f'_dbg_{start_pos}', show_figure=True, start_pos=start_pos)

    load_dir = 'C:/Data/Study/Udacity/DeeoReinforcementLearning/Cource4/Code/Progect/p3_collaboration-competition/debug/dbg_critic_sqrt-mse'
    s_id = 11
    save_dir = os.path.join(load_dir, f'session_{s_id}')
    logger = create_logger(root_dir='.', log_name=f'log_plot.log')

    plot_training_sessions_history(load_dir=load_dir, save_dir=save_dir, logger=logger, show_figure=False)