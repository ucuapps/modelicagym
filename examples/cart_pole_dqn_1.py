import logging
from gymalgs.rl import DqnAgent
import gym
import numpy as np
import math
import time


def cart_pole_train_dqn(cart_pole_env, agent_config, max_number_of_steps=500, n_episodes=4, visualize=True,
                        binning=False):
    """
    Runs one experiment of DQN training on cart pole environment
    :param cart_pole_env: environment RL agent will learn on.
    :param max_number_of_steps: maximum episode length.
    :param n_episodes: number of episodes of training to perform.
    :param visualize: flag if experiments should be rendered.
    :return: trained DQN agent, array of actual episodes length, execution time in s
    """

    start = time.time()

    episode_lengths = np.array([])

    if binning:
        x_bins = _get_bins(-2.4, 2.4, 10)
        x_dot_bins = _get_bins(-1, 1, 10)
        phi_bins = _get_bins(78/180*math.pi, 102/180*math.pi, 10)
        phi_dot_bins = _get_bins(-2, 2, 10)

    learner = DqnAgent(**agent_config)

    for episode in range(n_episodes):
        x, x_dot, phi, phi_dot = cart_pole_env.reset()

        if binning:
            state = [_to_bin(x, x_bins),
                      _to_bin(phi, phi_bins),
                      _to_bin(x_dot, x_dot_bins),
                      _to_bin(phi_dot, phi_dot_bins)]
        else:
            state = [x, x_dot, phi, phi_dot]

        action = learner.use(state)

        for step in range(max_number_of_steps):
            if visualize:
                cart_pole_env.render()

            observation, reward, done, _ = cart_pole_env.step(action)

            x, x_dot, phi, phi_dot = observation

            if binning:
                next_state = [_to_bin(x, x_bins),
                         _to_bin(phi, phi_bins),
                         _to_bin(x_dot, x_dot_bins),
                         _to_bin(phi_dot, phi_dot_bins)]
            else:
                next_state = [x, x_dot, phi, phi_dot]

            action = learner.learn(state, action, reward, next_state)
            state = next_state
            if done or step == max_number_of_steps - 1:
                episode_lengths = np.append(episode_lengths, int(step + 1))
                break

    end = time.time()
    execution_time = end - start
    return learner, episode_lengths, execution_time


# Internal logic for state discretization
def _get_bins(lower_bound, upper_bound, n_bins):
    """
    Given bounds for environment state variable splits it into n_bins number of bins,
    taking into account possible values outside the bounds.

    :param lower_bound: lower bound for variable describing state space
    :param upper_bound: upper bound for variable describing state space
    :param n_bins: number of bins to receive
    :return: n_bins-1 values separating bins. I.e. the most left bin is opened from the left,
    the most right bin is open from the right.
    """
    return np.linspace(lower_bound, upper_bound, n_bins + 1)[1:-1]


def _to_bin(value, bins):
    """
    Transforms actual state variable value into discretized one,
    by choosing the bin in variable space, where it belongs to.

    :param value: variable value
    :param bins: sequence of values separating variable space
    :return: number of bin variable belongs to. If it is smaller than lower_bound - 0.
    If it is bigger than the upper bound
    """
    return np.digitize(x=[value], bins=bins)[0]


def _get_state_index(state_bins):
    """
    Transforms discretized environment state (represented as sequence of bin indexes) into an integer value.
    Value is composed by concatenating string representations of a state_bins.
    Received string is a valid integer, so it is converted to int.

    :param state_bins: sequence of integers that represents discretized environment state.
    Each integer is an index of bin, where corresponding variable belongs.
    :return: integer value corresponding to the environment state
    """
    state = int("".join(map(lambda state_bin: str(state_bin), state_bins)))
    return state


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
                        binning=False,
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
                                                                        visualize=visualize,
                                                                        binning=binning)
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
    _, episodes_lengths, exec_times = run_dqn_experiments(agent_config=agent_config, visualize=True, log_level=logging.INFO,
                                                          binning=True)
    print("Experiment length {} s".format(exec_times[0]))
    print(u"Avg episode performance {} {} {}".format(episodes_lengths[0].mean(),
                                                     chr(177),  # plus minus sign
                                                     episodes_lengths[0].std()))
    print(u"Max episode performance {}".format(episodes_lengths[0].max()))
    print(u"All episodes performance {}".format(episodes_lengths))

    _, episodes_lengths, exec_times = run_dqn_experiments(agent_config=agent_config,
                                                          visualize=True, log_level=logging.INFO,
                                                          binning=False)
    print("Experiment length {} s".format(exec_times[0]))
    print(u"Avg episode performance {} {} {}".format(episodes_lengths[0].mean(),
                                                     chr(177),  # plus minus sign
                                                     episodes_lengths[0].std()))
    print(u"Max episode performance {}".format(episodes_lengths[0].max()))
    print(u"All episodes performance {}".format(episodes_lengths))

    _, episodes_lengths, exec_times = run_dqn_experiments(agent_config=agent_config,
                                                          visualize=False, log_level=logging.INFO,
                                                          binning=False, mode="ME")
