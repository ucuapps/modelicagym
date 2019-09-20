import numpy as np


# is based on the ThompsonSampling agent from bayesrl.agents package
# https://github.com/dustinvtran/bayesrl
class BDP(object):
    def __init__(self,
                 num_states,
                 num_actions,
                 discount_factor,
                 policy_update_interval,
                 dirichlet_smoothing_const,
                 default_reward):

        self.num_states = num_states
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.dirichlet_smoothing_const = dirichlet_smoothing_const
        self.default_reward = default_reward
        self.policy_update_interval = policy_update_interval

        self.last_state = None
        self.last_action = None

        # To keep track of where in T-step policy the agent is in; initialized to recompute policy
        self.policy_step = 0

        self.transition_observations = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.value_table = np.zeros((self.num_states, self.num_actions))

        self.reward = np.full((self.num_states, self.num_actions, self.num_states), self.default_reward)

    def interact(self, reward, next_state, next_state_is_terminal):
        # Handle start of episode.
        if reward is None:
            # Return random action since there is no information.
            next_action = np.random.randint(self.num_actions)
            self.last_state = next_state
            self.last_action = next_action
            return self.last_action

        # Update the reward associated with (s,a,s') if first time.
        # if self.reward[self.last_state, self.last_action, next_state] == self.default_reward:
        #    self.reward[self.last_state, self.last_action, next_state] = reward

        self.reward[self.last_state, self.last_action, next_state] = reward

        # Update set of states reached by playing a.
        self.transition_observations[self.last_state, self.last_action, next_state] += 1

        # Handle completion of episode.
        if self.policy_update_interval == 0 and next_state_is_terminal:
            print("recompute policy on episode end")
            self.__compute_policy()

        # Update transition probabilities after every policy_update_interval steps
        if self.policy_update_interval > 0 and self.policy_step == self.policy_update_interval:
            print("recompute policy every T {} steps".format(self.policy_update_interval))
            self.__compute_policy()

        # Choose next action according to policy.
        next_action = self._argmax_breaking_ties_randomly(self.value_table[next_state])

        self.policy_step += 1
        self.last_state = next_state
        self.last_action = next_action

        return self.last_action

    def _value_iteration(self, transition_probs):
        """
        Run value iteration, using procedure described in Sutton and Barto
        (2012). The end result is an updated value_table, from which one can
        deduce the policy for state s by taking the argmax (breaking ties
        randomly).
        """
        value_dim = transition_probs.shape[0]
        value = np.zeros(value_dim)
        k = 0
        while True:
            diff = 0
            for s in range(value_dim):
                old = value[s]
                value[s] = np.max(np.sum(transition_probs[s]*(self.reward[s] +
                                  self.discount_factor*np.array([value, ]*self.num_actions)),
                                  axis=1))
                diff = max(0, abs(old - value[s]))
            k += 1
            if diff < 1e-1:
                break
            if k > 1e4:
                raise Exception("Value iteration not converging. Stopped at 1e4 iterations.")
        for s in range(value_dim):
            self.value_table[s] = np.sum(transition_probs[s]*(self.reward[s] +
                                         self.discount_factor*np.array([value, ]*self.num_actions)),
                                         axis=1)

    def __compute_policy(self):
        """Compute an optimal T-step policy for the current state."""
        print("""
        ---
        recomputing the policy
        ---""")
        self.policy_step = 0
        transition_probs = np.zeros((self.num_states, self.num_actions, self.num_states))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                transition_probs[s, a] = np.random.dirichlet(self.transition_observations[s, a] +
                                                             self.dirichlet_smoothing_const, size=1)
        self._value_iteration(transition_probs)

    def _argmax_breaking_ties_randomly(self, x):
        max_value = np.max(x)
        indices_with_max_value = np.flatnonzero(x == max_value)
        return np.random.choice(indices_with_max_value)
