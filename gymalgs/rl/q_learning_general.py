import numpy as np
from gym.wrappers import Monitor
import pandas as pd
import os.path


class QLearner(object):
    def __init__(self,
                 num_states=100,
                 num_actions=4,
                 alpha=0.2,
                 gamma=0.9,
                 random_action_rate=0.5,
                 random_action_decay_rate=0.99):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.random_action_rate = random_action_rate
        self.random_action_decay_rate = random_action_decay_rate
        self.state = 0
        self.action = 0
        self.qtable = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))

    def use(self, state):
        return self.qtable[state].argsort()[-1]

    def set_initial_state(self, state):
        """
        @summary: Sets the initial state and returns an action
        @param state: The initial state
        @returns: The selected action
        """
        self.state = state
        self.action = self.qtable[state].argsort()[-1]
        return self.action

    def move(self, state_prime, reward):
        """
        @summary: Moves to the given state with given reward and returns action
        @param state_prime: The new state
        @param reward: The reward
        @returns: The selected action
        """

        choose_random_action = (1 - self.random_action_rate) <= np.random.uniform(0, 1)

        if choose_random_action:
            action_prime = np.random.randint(0, self.num_actions - 1)
        else:
            action_prime = self.qtable[state_prime].argsort()[-1]

        self.random_action_rate *= self.random_action_decay_rate

        self.qtable[self.state, self.action] = (1 - self.alpha) * self.qtable[self.state, self.action] + self.alpha * (
            reward + self.gamma * self.qtable[state_prime, action_prime])

        self.state = state_prime
        self.action = action_prime

        return self.action


class GymQLearner:
    def __init__(self, env):
        self.env = env
        self.learner = None

    def inv_pendulum_experiment(self, visualize=True):
        if visualize:
            experiment_filename = './animation_50'
            monitor = Monitor(self.env, experiment_filename, force=True)

        goal_average_steps = 100
        max_number_of_steps = 200
        number_of_iterations_to_average = 100

        number_of_features = self.env.observation_space.shape[0]
        last_time_steps = np.ndarray(0)

        cart_position_bins = pd.cut([-2.4, 2.4], bins=10, retbins=True)[1][1:-1]
        pole_angle_bins = pd.cut([-2, 2], bins=10, retbins=True)[1][1:-1]
        cart_velocity_bins = pd.cut([-1, 1], bins=10, retbins=True)[1][1:-1]
        angle_rate_bins = pd.cut([-3.5, 3.5], bins=10, retbins=True)[1][1:-1]

        self.learner = QLearner(num_states=10 ** number_of_features,
                                num_actions=self.env.action_space.n,
                                alpha=0.2,
                                gamma=1,
                                random_action_rate=0.5,
                                random_action_decay_rate=0.99)

        for episode in range(4):
            print("episode {}".format(episode))
            observation = self.env.reset()
            print("observation : {}".format(observation))
            cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
            state = _build_state([_to_bin(cart_position, cart_position_bins),
                                 _to_bin(pole_angle, pole_angle_bins),
                                 _to_bin(cart_velocity, cart_velocity_bins),
                                 _to_bin(angle_rate_of_change, angle_rate_bins)])

            action = self.learner.set_initial_state(state)

            file_nr = episode
            filename = "Model_internal" + str(file_nr) + ".mat"
            if os.path.exists(filename):
                os.remove(filename)

            for step in range(max_number_of_steps - 1):
                if visualize:
                    self.env.render()

                observation, reward, done, info = self.env.step(action)

                cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation

                state_prime = _build_state([_to_bin(cart_position, cart_position_bins),
                                           _to_bin(pole_angle, pole_angle_bins),
                                           _to_bin(cart_velocity, cart_velocity_bins),
                                           _to_bin(angle_rate_of_change, angle_rate_bins)])

                if done:
                    reward = -300

                action = self.learner.move(state_prime, reward)

                if done or step == max_number_of_steps - 2:
                    last_time_steps = np.append(last_time_steps, int(step + 1))
                    if len(last_time_steps) > number_of_iterations_to_average:
                        last_time_steps = np.delete(last_time_steps, 0)
                    break

            if last_time_steps.mean() > goal_average_steps:
                print("Goal reached!")
                print("Episodes before solve: ", episode + 1)
                print(u"Best 100-episode performance {} {} {}".format(max(last_time_steps),
                                                                      chr(177),  # plus minus sign
                                                                      last_time_steps.std()))
                break

        print("Goal was not reached!")
        print(u"Avg episode performance {} {} {}".format(last_time_steps.mean(),
                                                         chr(177),  # plus minus sign
                                                         last_time_steps.std()))
        print(u"Max episode performance {}".format(last_time_steps.max()))
        print(u"All episodes performance {}".format(last_time_steps))

        if visualize:
            monitor.close()
            self.env.render(close=True)
        return self.learner

    def run_experiments(self, n_episodes, visualize=True):

        monitor = Monitor(self.env, None, force=True)
        goal_average_steps = 100
        max_number_of_steps = 200
        number_of_iterations_to_average = 100

        number_of_features = self.env.observation_space.shape[0]
        last_time_steps = np.ndarray(0)

        cart_position_bins = pd.cut([-2.4, 2.4], bins=10, retbins=True)[1][1:-1]
        pole_angle_bins = pd.cut([-2, 2], bins=10, retbins=True)[1][1:-1]
        cart_velocity_bins = pd.cut([-1, 1], bins=10, retbins=True)[1][1:-1]
        angle_rate_bins = pd.cut([-3.5, 3.5], bins=10, retbins=True)[1][1:-1]

        self.learner = QLearner(num_states=10 ** number_of_features,
                                num_actions=self.env.action_space.n,
                                alpha=0.2,
                                gamma=1,
                                random_action_rate=0.5,
                                random_action_decay_rate=0.99)

        for episode in range(n_episodes):
            print("episode {}".format(episode))
            observation = self.env.reset()
            print("observation : {}".format(observation))
            cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
            state = _build_state([_to_bin(cart_position, cart_position_bins),
                                 _to_bin(pole_angle, pole_angle_bins),
                                 _to_bin(cart_velocity, cart_velocity_bins),
                                 _to_bin(angle_rate_of_change, angle_rate_bins)])

            action = self.learner.set_initial_state(state)

            file_nr = episode
            filename = "Model_internal" + str(file_nr) + ".mat"
            if os.path.exists(filename):
                os.remove(filename)

            for step in range(max_number_of_steps - 1):
                if visualize:
                    self.env.render()

                observation, reward, done, info = self.env.step(action)

                cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation

                state_prime = _build_state([_to_bin(cart_position, cart_position_bins),
                                           _to_bin(pole_angle, pole_angle_bins),
                                           _to_bin(cart_velocity, cart_velocity_bins),
                                           _to_bin(angle_rate_of_change, angle_rate_bins)])

                if done:
                    reward = -300

                action = self.learner.move(state_prime, reward)

                if done or step == max_number_of_steps - 2:
                    last_time_steps = np.append(last_time_steps, int(step + 1))
                    if len(last_time_steps) > number_of_iterations_to_average:
                        last_time_steps = np.delete(last_time_steps, 0)
                    break

            if last_time_steps.mean() > goal_average_steps:
                print("Goal reached!")
                print("Episodes before solve: ", episode + 1)
                print(u"Best 100-episode performance {} {} {}".format(max(last_time_steps),
                                                                      chr(177),  # plus minus sign
                                                                      last_time_steps.std()))
                break

        print("Goal was not reached!")
        print(u"Avg episode performance {} {} {}".format(last_time_steps.mean(),
                                                         chr(177),  # plus minus sign
                                                         last_time_steps.std()))
        print(u"Max episode performance {}".format(last_time_steps.max()))
        print(u"All episodes performance {}".format(last_time_steps))

        monitor.close()
        if visualize:
            self.env.render(close=True)
        return self.learner


def _build_state(features):
    state = int("".join(map(lambda feature: str(int(feature)), features)))
    return state


def _to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]
