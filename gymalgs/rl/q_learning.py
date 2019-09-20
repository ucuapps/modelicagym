import numpy as np


# Is extension of the original version:
# https://gist.github.com/carlos-aguayo/3df32b1f5f39353afa58fbc29f9227a2#file-cartpole-v0-py
class QLearner(object):
    """
    QLearning algorithm implementation:
    dynamic programming approach to estimation of Q-values for state-action pairs
    Attributes:
        learning_rate (float): a.k.a. alpha or update rate. Advised to be in interval [0;1].
        Determines how much Q-value is updated with each new observation. 0 - no update, 1 - forget everything.

        discount_factor (float): a.k.a. gamma.
        Determines the importance of future reward. Advised to be in interval [0;1].
        0 - consider immediate reward, 1 - only long-term reward (encourages infinitely long scenarios).

        exploration_rate (float): determines exploration (random action choice) probability in exploration-exploitation
        trade-off. Should be in interval [0;1].

        exploration_decay_rate (float): determines exploration probability decay each step. Should be in [0;1].

        num_states (int): number of states environment may be in. If you use continuous state environment,
        its states space should be discretized.

        num_actions (float): number of action that can be performed on environment
        If you use continuous action environment, its states space should be discretized.

        action (int): last performed action

        state (int): last environment state

        qtable (ndarray): Q-table of Q-learning algorithm. qtable[i][j] contains Q-values for i-th state, j-th action


    """
    def __init__(self,
                 learning_rate,
                 discount_factor,
                 exploration_rate,
                 exploration_decay_rate,
                 n_states,
                 n_actions,
                 rand_qtab=False):

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.random_action_rate = exploration_rate
        self.random_action_decay_rate = exploration_decay_rate

        self.num_states = n_states
        self.num_actions = n_actions

        self.state = 0
        self.action = 0

        if rand_qtab:
            self.qtable = np.random.uniform(low=-1, high=1, size=(n_states, n_actions))
        else:
            self.qtable = np.zeros((n_states, n_actions))

    def use(self, state):
        """
        Exploitation phase of the q-learning algorithm. I.e. no learning.
        Should not be used from outside before learning.
        :param state: current environment state
        :return: best action in this state according to the learnt Q-table
        """
        return _choose_max_index(self.qtable[state])

    def set_initial_state(self, state):
        """
        Choose action for initial state and save it
        :param state: state to be set as initial
        :return: best action (based on randomly initialized Q-table)
        """
        self.state = state
        self.action = self.use(state)
        return self.action

    def learn_observation(self, next_state, reward):
        """
        Updates Q-table based on the new observation. Action performed to get environment into next_state is stored
        in action attribute of the class instance.
        :param next_state: state where environment is after previous action.
        :param reward: reward for environment being in the next_state
        :return: advised (best possible) action for the next_state
        """
        if self.random_action_rate >= np.random.uniform(0, 1):
            next_action = np.random.randint(0, self.num_actions - 1)
        else:
            next_action = self.use(next_state)

        self.qtable[self.state, self.action] = (1 - self.learning_rate) * self.qtable[self.state, self.action] + \
            self.learning_rate * (reward + self.discount_factor * self.qtable[next_state, next_action])

        self.random_action_rate *= self.random_action_decay_rate

        self.state = next_state
        self.action = next_action

        return self.action


# inspired by approach to action selection agents in bayesrl library
def _choose_max_index(x):
    max_value = np.max(x)
    indices_with_max_value = np.flatnonzero(x == max_value)
    return np.random.choice(indices_with_max_value)
