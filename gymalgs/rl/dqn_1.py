# implementation based on tutorials:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# https://medium.com/@unnatsingh/deep-q-network-with-pytorch-d1ca6f40bfda

import random
from collections import namedtuple, deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Observation = namedtuple('Observation', ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Save an observation."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Observation(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class MlpDqn(nn.Module):
    def __init__(self, n_state_input, n_hidden_1, n_hidden_2, n_actions):
        super(MlpDqn, self).__init__()
        self.fc1 = nn.Linear(n_state_input, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.out = nn.Linear(n_hidden_2, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class DqnAgent:
    def __init__(self, actions, n_state_variables, n_hidden_1, n_hidden_2,
                 buffer_size, batch_size, exploration_rate, expl_rate_decay, expl_rate_final,
                 discount_factor, target_update, expl_decay_step, dummy=False):
        self.expl_decay_step = expl_decay_step
        n_actions = len(actions)
        self.actions = actions
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.expl_rate_decay = expl_rate_decay
        self.expl_rate_final = expl_rate_final
        self.step_counter = 0
        self.batch_size = batch_size
        self.dummy = dummy

        self.model = MlpDqn(n_state_variables, n_hidden_1, n_hidden_2, n_actions)

        self.target_update = target_update
        self.model_target = self.model

        if self.target_update is not None:
            self.model_target = MlpDqn(n_state_variables, n_hidden_1, n_hidden_2, n_actions)
            self.model_target.load_state_dict(self.model.state_dict())
            self.model_target.eval()

        self.buffer = ReplayBuffer(buffer_size)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def learn(self, state, action, reward, next_state):
        self.step_counter += 1

        action = self.actions.index(action)
        self.buffer.push(state, action, next_state, reward)

        if len(self.buffer) < self.batch_size:
            return self._choose_random_action()

        batch = self.buffer.sample(self.batch_size)

        self.train_dqn(batch)

        if self.target_update is not None and self.step_counter % self.target_update == 0:
            self.model_target.load_state_dict(self.model.state_dict())

        return self.use(next_state)

    def use(self, state):
        if random.random() > self.get_current_expl_rate():
            state = torch.Tensor(state)
            with torch.no_grad():
                optimal_action_index = self.model(state).max(0).indices
            return self.actions[optimal_action_index]
        else:
            return self._choose_random_action()

    def train_dqn(self, batch):
        self.optimizer.zero_grad()
        loss = self.calc_loss(batch)
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def calc_loss(self, batch):
        batch = Observation(*zip(*batch))

        non_final_next_states = torch.Tensor(batch.next_state)
        state_batch = torch.Tensor(batch.state)
        action_batch = torch.LongTensor(batch.action).view(-1, 1)
        reward_batch = torch.Tensor(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        all_qvs = self.model(state_batch)
        state_action_values = all_qvs.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        next_state_values = self.model_target(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        # loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        return loss

    def save(self, path):
        torch.save(self.model.state_dict(), path+".pt")

    def get_current_expl_rate(self):
        to_return = self.exploration_rate
        # print("Current exploration rate was {}".format(to_return))
        if self.step_counter % self.expl_decay_step == 0:
            self.exploration_rate *= self.expl_rate_decay
        if self.exploration_rate < self.expl_rate_final:
            self.exploration_rate = self.expl_rate_final
        return to_return

    def _choose_random_action(self):
        return random.sample(self.actions, k=1)[0]


def play_episode(env, agent):
    x, x_dot, phi, phi_dot = env.reset()
    state = [x, x_dot, phi, phi_dot]
    action = 1
    total_reward = 0
    done = False

    while not done:
        next_state, r, done, info = env.step(action)
        total_reward += r
        if done:
            r = -1
        action = agent.learn(state, action, r, next_state)
        state = next_state

    return total_reward


if __name__ == "__main__":

    net = MlpDqn(4, 32, 32, 2)
    print(net((torch.Tensor(
        [1, 1, 1, 1]

    ))))
    index = net((torch.Tensor(
        [1, 1, 1, 1]

    ))).max(0).indices
    print(index)
    ar = [20, 30, 40, 50, 60]
    print(ar[index])
    agent = DqnAgent(actions=[1, 2, 3], n_state_variables=4, n_hidden_1=32, n_hidden_2=32,
                     buffer_size=100, batch_size=16, exploration_rate=0.5, expl_rate_decay=0.999,
                     expl_rate_final=0.05,
                     discount_factor=0.6, target_update=100,
                     expl_decay_step=10)

    print(agent.use(torch.Tensor([10, 20, 30, 40])))

    buf = ReplayBuffer(5)
    buf.push([10, 20, 30, 40], 0, [10, 20, 30, 10], 0)
    buf.push([10, 20, 30, 40], 1, [10, 20, 30, 10], 0)
    buf.push([10, 20, 30, 40], 2, [10, 20, 30, 10], 0)
    buf.push([10, 20, 30, 40], 0, [10, 20, 30, 10], 1)
    buf.push([10, 20, 30, 40], 1, [10, 20, 30, 10], 1)
    buf.push([10, 20, 30, 40], 2, [10, 20, 30, 10], 1)

    sample = buf.sample(3)
    print(sample)
    agent.train_dqn(sample)

    env = gym.make("CartPole-v0")
    agent = DqnAgent(actions=[0, 1], n_state_variables=4, n_hidden_1=16,
                     n_hidden_2=16, buffer_size=512, batch_size=64,
                     exploration_rate=1, expl_rate_decay=0.99, expl_rate_final=0.01,
                     discount_factor=0.99, target_update=100, expl_decay_step=1)

    rewards = deque(maxlen=100)
    episodes_length = []

    for i in range(1000):
        r = play_episode(env, agent)
        print("[Episode: {:5}] Reward: {:5} 𝜺-greedy: {:5.2f}".format(i + 1, r, agent.exploration_rate))

        rewards.append(r)
        episodes_length.append(r)
        if len(rewards) == rewards.maxlen:

            if np.mean(rewards) >= 200:
                print("Game cleared in {} games with {}".format(i + 1, np.mean(rewards)))
                print(episodes_length)
                break

