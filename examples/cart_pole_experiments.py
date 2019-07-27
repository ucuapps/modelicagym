import logging
import numpy as np
import math
from examples import run_experiments


def run_experiment_with_result_files(folder,
                                     n_experiments,
                                     n_episodes,
                                     visualize,
                                     m_trolley,
                                     m_load,
                                     phi1_start,
                                     w1_start,
                                     time_step,
                                     positive_reward,
                                     negative_reward,
                                     force,
                                     log_level):
    """
    Runs experiments with the given configuration and writes episodes length of all experiment as one file
    and execution times of experiments as another.
    File names are composed from numerical experiment parameters
    in the same order as in function definition.
    Episodes length are written as 2d-array of shape (n_episodes, n_experiments):
    i-th row - i-th episode, j-th column - j-th experiment.

    Execution times are written as 1d-array of shape (n_experiments, ): j-th element - j-th experiment


    :param folder: folder for experiment result files
    :return: None
    """
    experiment_file_name_prefix = "{}/experiment_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_".format(
        folder,
        n_experiments,
        n_episodes,
        m_trolley,
        m_load,
        phi1_start,
        w1_start,
        time_step,
        positive_reward,
        negative_reward,
        force,
    )
    n_episodes = 200
    _, episodes_lengths, exec_times = run_experiments(n_experiments=n_experiments,
                                                      n_episodes=n_episodes,
                                                      visualize=visualize,
                                                      m_trolley=m_trolley,
                                                      m_load=m_load,
                                                      phi1_start=phi1_start,
                                                      w1_start=w1_start,
                                                      time_step=time_step,
                                                      positive_reward=positive_reward,
                                                      negative_reward=negative_reward,
                                                      force=force,
                                                      log_level=log_level)

    np.transpose(episodes_lengths).tofile(experiment_file_name_prefix + "episodes_lengths.csv", )
    exec_times.tofile(experiment_file_name_prefix + "exec_times.csv", )


if __name__ == "__main__":

    folder = "experiments_results"

    for f in [5, 11, 17]:
        run_experiment_with_result_files(folder,
                                         n_experiments=10,
                                         n_episodes=200,
                                         visualize=False,
                                         m_trolley=10,
                                         m_load=1,
                                         phi1_start=85/180*math.pi,
                                         w1_start=0,
                                         time_step=0.05,
                                         positive_reward=1,
                                         negative_reward=-100,
                                         force=f,
                                         log_level=logging.DEBUG)
