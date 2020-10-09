import logging
from collections import deque

import gym
import numpy as np
import math
import time

from gymalgs.rl.dqn_2 import DQN2Agent, ReplayMemory, epsilon_annealing, play_episode


def cart_pole_train_dqn(cart_pole_env, agent_config, max_number_of_steps=500, n_episodes=4, visualize=True):
    """
    Runs one experiment of DQN training on cart pole environment
    :param cart_pole_env: environment RL agent will learn on.
    :param max_number_of_steps: maximum episode length.
    :param n_episodes: number of episodes of training to perform.
    :param visualize: flag if experiments should be rendered.
    :return: trained DQN agent, array of actual episodes length, execution time in s
    """

    start = time.time()
    input_dim = agent_config.get('input_dim', 4)
    output_dim = agent_config.get('output_dim', 2)
    hidden_layers = agent_config.get('hidden_layers', 16)
    buffer_capacity = agent_config.get('buffer_capacity', 50000)
    max_episode = agent_config.get('max_episode', 50)
    min_eps = agent_config.get('min_eps', 0.01)
    batch_size = agent_config.get('batch_size', 64)

    episode_lengths = np.array([])
    agent = DQN2Agent(input_dim, output_dim, hidden_layers)
    replay_memory = ReplayMemory(buffer_capacity)
    rewards = deque(maxlen=100)

    for i in range(n_episodes):
        eps = epsilon_annealing(i, max_episode, min_eps)
        r = play_episode(cart_pole_env, agent, replay_memory, eps, batch_size)
        print("[Episode: {:5}] Reward: {:5} 𝜺-greedy: {:5.2f}".format(i + 1, r, eps))

        episode_lengths = np.append(episode_lengths, r)
        if len(rewards) == rewards.maxlen:

            if np.mean(rewards) >= 200:
                print("Game cleared in {} games with {}".format(i + 1, np.mean(rewards)))
                break

    end = time.time()
    execution_time = end - start
    return agent, episode_lengths, execution_time


def run_dqn_experiments(agent_config,
                        n_experiments=1,
                       n_episodes=10,
                       visualize=False,
                       m_cart=10,
                       m_pole=1,
                       theta_0=85/180*math.pi,
                       theta_dot_0=0,
                       time_step=0.05,
                       positive_reward=1,
                       negative_reward=-100,
                       force=12,
                       log_level=logging.DEBUG,
                        mode="CS"):
    """
    Wrapper for running experiment of DQN training on cart pole environment.
    Is responsible for environment creation and closing, sets all necessary parameters of environment.
    Runs n experiments, where each experiment is training DQN agent on the same environment.
    After one agent finished training, environment is reset to the initial state.
    Parameters of the experiment:
    :param n_episodes: number of episodes to perform in each experiment run
    :param visualize: boolean flag if experiments should be rendered
    :param n_experiments: number of experiments to perform.
    :param log_level: level of logging that should be used by environment during experiments.
    :param force: magnitude to be applied during experiment at each time step.

    Parameters of the cart pole environment:
    :param m_cart: mass of cart.
    :param m_pole: mass of a pole.
    :param theta_0: angle of the pole. Is counted from the positive direction of X-axis. Specified in radians.
    1/2*pi means pole standing straight on the cast.
    :param theta_dot_0: angle speed of the poles mass center. I.e. how fast pole angle is changing.
    :param time_step: time difference between simulation steps.
    :param positive_reward: positive reward for RL agent.
    :param negative_reward: negative reward for RL agent.
    :return: trained DQN agent, array of actual episodes length
    that were returned from cart_pole_train_dqn()
    """
    if mode == "ME":
        path = "../resources/jmodelica/linux/ModelicaGym_CartPole_ME.fmu"
        env_entry_point = 'examples:JModelicaMECartPoleEnv'
    else:
        path = "../resources/jmodelica/linux/ModelicaGym_CartPole_CS.fmu"
        env_entry_point = 'examples:JModelicaCSCartPoleEnv'

    config = {
        'path': path,
        'm_cart': m_cart,
        'm_pole': m_pole,
        'theta_0': theta_0,
        'theta_dot_0': theta_dot_0,
        'time_step': time_step,
        'positive_reward': positive_reward,
        'negative_reward': negative_reward,
        'force': force,
        'log_level': log_level
    }

    from gym.envs.registration import register
    env_name = "JModelicaCSCartPoleEnv-v0"

    register(
        id=env_name,
        entry_point=env_entry_point,
        kwargs=config
    )
    trained_agent_s = []
    episodes_length_s = []
    exec_time_s = []
    env = gym.make(env_name)
    for i in range(n_experiments):
        trained_agent, episodes_length, exec_time = cart_pole_train_dqn(env, agent_config,
                                                                        n_episodes=n_episodes,
                                                                        visualize=visualize)
        trained_agent_s.append(trained_agent)
        episodes_length_s.append(episodes_length)
        exec_time_s.append(exec_time)
        env.reset()

    env.close()
    # delete registered environment to avoid errors in future runs.
    del gym.envs.registry.env_specs[env_name]
    return trained_agent_s, episodes_length_s, exec_time_s


if __name__ == "__main__":
    agent_config = {
        'actions': [0, 1],
        'n_state_variables': 4,
        'n_hidden_1': 64,
        'n_hidden_2': 64,
        'buffer_size': 512,
        'batch_size': 64,
        'exploration_rate': 0.5,
        'expl_rate_decay': 0.999,
        'expl_rate_final': 0.05,
        'discount_factor': 0.99,
        'target_update': 1000,
        'expl_decay_step': 1
    }
    _, episodes_lengths, exec_times = run_dqn_experiments(visualize=True, log_level=logging.INFO,
                                                          agent_config=agent_config)
    print("Experiment length {} s".format(exec_times[0]))
    print(u"Avg episode performance {} {} {}".format(episodes_lengths[0].mean(),
                                                     chr(177),  # plus minus sign
                                                     episodes_lengths[0].std()))
    print(u"Max episode performance {}".format(episodes_lengths[0].max()))
    print(u"All episodes performance {}".format(episodes_lengths))
