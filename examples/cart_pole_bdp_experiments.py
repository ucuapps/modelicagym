import logging
import numpy as np
import math
from examples import run_bdp_experiments


def run_experiment_with_result_files(folder,
                                     n_experiments,
                                     n_episodes,
                                     visualize,
                                     m_cart,
                                     m_pole,
                                     theta_0,
                                     theta_dot_0,
                                     time_step,
                                     positive_reward,
                                     negative_reward,
                                     force,
                                     discount_factor,
                                     policy_update_interval,
                                     dirichlet_smoothing_const,
                                     default_reward,
                                     log_level
                                     ):
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
    experiment_file_name_prefix = "{}/experiment_bdp_{}_{}_{}_{}_{:.0f}_{}_{}_{}_{}_{}_{}_{}_{}_{}_".format(
        folder,
        n_experiments,
        n_episodes,
        m_cart,
        m_pole,
        theta_0 * 180 / math.pi,
        theta_dot_0,
        time_step,
        positive_reward,
        negative_reward,
        force,
        discount_factor,
        policy_update_interval,
        dirichlet_smoothing_const,
        default_reward
    )
    _, episodes_lengths, exec_times = run_bdp_experiments(n_experiments=n_experiments,
                                                          n_episodes=n_episodes,
                                                          visualize=visualize,
                                                          m_cart=m_cart,
                                                          m_pole=m_pole,
                                                          theta_0=theta_0,
                                                          theta_dot_0=theta_dot_0,
                                                          time_step=time_step,
                                                          positive_reward=positive_reward,
                                                          negative_reward=negative_reward,
                                                          force=force,
                                                          discount_factor=discount_factor,
                                                          policy_update_interval=policy_update_interval,
                                                          dirichlet_smoothing_const=dirichlet_smoothing_const,
                                                          default_reward=default_reward,
                                                          log_level=log_level)

    np.savetxt(fname=experiment_file_name_prefix + "episodes_lengths.csv",
               X=np.transpose(episodes_lengths),
               delimiter=",",
               fmt="%d")
    np.savetxt(fname=experiment_file_name_prefix + "exec_times.csv",
               X=np.array(exec_times),
               delimiter=",",
               fmt="%.4f")


def policy_update_interval_experiment(puis):
    for pui in puis:
        run_experiment_with_result_files(folder,
                                         n_experiments=3,
                                         n_episodes=100,
                                         visualize=False,
                                         m_cart=10,
                                         m_pole=1,
                                         theta_0=85 / 180 * math.pi,
                                         theta_dot_0=0,
                                         time_step=0.05,
                                         positive_reward=1,
                                         negative_reward=-100,
                                         force=15,
                                         discount_factor=0.95,
                                         policy_update_interval=pui,
                                         dirichlet_smoothing_const=1,
                                         default_reward=1,
                                         log_level=logging.INFO)


def default_reward_experiment(drs):
    for dr in drs:
        run_experiment_with_result_files(folder,
                                         n_experiments=3,
                                         n_episodes=100,
                                         visualize=False,
                                         m_cart=10,
                                         m_pole=1,
                                         theta_0=85 / 180 * math.pi,
                                         theta_dot_0=0,
                                         time_step=0.05,
                                         positive_reward=1,
                                         negative_reward=-100,
                                         force=15,
                                         discount_factor=0.95,
                                         policy_update_interval=50,
                                         dirichlet_smoothing_const=1,
                                         default_reward=dr,
                                         log_level=logging.INFO)


if __name__ == "__main__":
    import time
    start = time.time()
    folder = "experiments_results"
    # policy_update_interval_experiment([50])
    default_reward_experiment([0, 2])
    # following experiments rake significant amount of time, so it is advised to run only one of them at once
    #
    end = time.time()
    print("Total execution time {:.2f} seconds".format(end-start))
