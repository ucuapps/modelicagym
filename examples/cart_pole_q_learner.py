import logging
import gym
from gymalgs.rl import QLearner
import numpy as np
import math


def cart_pole_train_qlearning(cart_pole_env, max_number_of_steps=200, n_episodes=4, visualize=True):
    """
    Runs one experiment of Q-learning training on cart pole environment
    :param cart_pole_env: environment RL agent will learn on.
    :param max_number_of_steps: maximum episode length.
    :param n_episodes: number of episodes to perform.
    :param visualize: flag if experiments should be rendered.
    :return: trained Q-learning agent, array of actual episodes length.
    """
    n_outputs = cart_pole_env.observation_space.shape[0]

    episode_lengths = np.array([])

    x_bins = _get_bins(-2.4, 2.4, 10)
    x_dot_bins = _get_bins(-1, 1, 10)
    phi_bins = _get_bins(78/180*math.pi, 102/180*math.pi, 10)
    phi_dot_bins = _get_bins(-2, 2, 10)

    learner = QLearner(n_states=10 ** n_outputs,
                       n_actions=cart_pole_env.action_space.n,
                       learning_rate=0.2,
                       discount_factor=1,
                       exploration_rate=0.5,
                       exploration_decay_rate=0.99)

    for episode in range(n_episodes):
        x, x_dot, phi, phi_dot = cart_pole_env.reset()

        state = _get_state_index([_to_bin(x, x_bins),
                                  _to_bin(phi, phi_bins),
                                  _to_bin(x_dot, x_dot_bins),
                                  _to_bin(phi_dot, phi_dot_bins)])

        action = learner.set_initial_state(state)

        for step in range(max_number_of_steps):
            if visualize:
                cart_pole_env.render()

            observation, reward, done, _ = cart_pole_env.step(action)

            x, x_dot, phi, phi_dot = observation
            state_prime = _get_state_index([_to_bin(x, x_bins),
                                            _to_bin(phi, phi_bins),
                                            _to_bin(x_dot, x_dot_bins),
                                            _to_bin(phi_dot, phi_dot_bins)])

            action = learner.learn_observation(state_prime, reward)

            if done or step == max_number_of_steps - 1:
                episode_lengths = np.append(episode_lengths, int(step + 1))
                break

    return learner, episode_lengths


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


def run_experiment(m_trolley=10, m_load=1,
                   phi1_start=85/180*math.pi,
                   w1_start=0,
                   time_step=0.05,
                   positive_reward=1,
                   negative_reward=-100,
                   force=12,
                   log_level=logging.DEBUG):
    """
    Wrapper for running experiment of q-learning training on cart pole environment.
    Is responsible for environment creation and closing, sets all necessary parameters of environment.
    Parameters of the experiment:
    :param log_level: level of logging that should be used by environment during experiments.
    :param force: magnitude to be applied during experiment at each time step.

    Parameters of the cart pole environment:
    :param m_trolley: mass of cart.
    :param m_load: mass of a pole.
    :param phi1_start: angle of the pole. Is counted from the positive direction of X-axis. Specified in radians.
    1/2*pi means pole standing straight on the cast.
    :param w1_start: angle speed of the poles mass center. I.e. how fast pole angle is changing.
    :param time_step: time difference between simulation steps.
    :param positive_reward: positive reward for RL agent.
    :param negative_reward: negative reward for RL agent.
    :return: trained Q-learning agent, array of actual episodes length
    that were returned from cart_pole_train_qlearning()
    """
    config = {
        'm_trolley': m_trolley,
        'm_load': m_load,
        'phi1_start': phi1_start,
        'w1_start': w1_start,
        'time_step': time_step,
        'positive_reward': positive_reward,
        'negative_reward': negative_reward,
        'force': force,
        'log_level': log_level
    }

    from gym.envs.registration import register
    register(
        id="JModelicaCSCartPoleEnv-v0",
        entry_point='examples:JModelicaCSCartPoleEnv',
        kwargs=config
    )
    env = gym.make("JModelicaCSCartPoleEnv-v0")
    res = cart_pole_train_qlearning(env, visualize=True)
    env.close()
    return res


if __name__ == "__main__":
    _, last_time_steps = run_experiment()
    print(u"Avg episode performance {} {} {}".format(last_time_steps.mean(),
                                                     chr(177),  # plus minus sign
                                                     last_time_steps.std()))
    print(u"Max episode performance {}".format(last_time_steps.max()))
    print(u"All episodes performance {}".format(last_time_steps))
