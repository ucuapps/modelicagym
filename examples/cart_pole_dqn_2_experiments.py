import logging
import numpy as np
import math
from examples.cart_pole_dqn_2 import run_dqn_experiments


def run_experiment_with_result_files(folder,
                                     agent_config,
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
                                     log_level,
                                     binning,
                                     mode,
                                     exp_id=None):
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
    experiment_file_name_prefix = f"{folder}/experiment_dqn2_{exp_id}_"
    _, episodes_lengths, exec_times = run_dqn_experiments(agent_config=agent_config,
                                                          n_experiments=n_experiments,
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
                                                         log_level=log_level,
                                                          mode=mode)

    np.savetxt(fname=experiment_file_name_prefix + "episodes_lengths.csv",
               X=np.transpose(episodes_lengths),
               delimiter=",",
               fmt="%d")
    np.savetxt(fname=experiment_file_name_prefix + "exec_times.csv",
               X=np.array(exec_times),
               delimiter=",",
               fmt="%.4f")


if __name__ == "__main__":
    import time
    start = time.time()
    folder = "experiments_results/dqn_2/"
    agent_config = {
        'input_dim': 4,
        'output_dim': 2,
        'hidden_layers': 16,
        'buffer_capacity': 50000,
        'max_episode': 50,
        'min_eps': 0.01,
        'batch_size': 64
    }
    run_experiment_with_result_files(folder,
                                     agent_config=agent_config,
                                     n_experiments=5,
                                     n_episodes=1000,
                                     visualize=False,
                                     m_cart=10,
                                     m_pole=1,
                                     theta_0=85 / 180 * math.pi,
                                     theta_dot_0=0,
                                     time_step=0.05,
                                     positive_reward=1,
                                     negative_reward=-1,
                                     force=15,
                                     log_level=logging.INFO,
                                     binning=False,
                                     mode="CS",
                                     exp_id="n1")
    run_experiment_with_result_files(folder,
                                     agent_config=agent_config,
                                     n_experiments=5,
                                     n_episodes=1000,
                                     visualize=False,
                                     m_cart=10,
                                     m_pole=1,
                                     theta_0=85 / 180 * math.pi,
                                     theta_dot_0=0,
                                     time_step=0.05,
                                     positive_reward=1,
                                     negative_reward=-100,
                                     force=15,
                                     log_level=logging.INFO,
                                     binning=False,
                                     mode="CS",
                                     exp_id="n2")
    end = time.time()
    print("Total execution time {:.2f} seconds".format(end-start))
