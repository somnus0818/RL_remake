import numpy as np
import torch
import torch.nn as nn

import numpy

import gym


env = gym.make('CartPole-v0')
s_size = env.observation_space.shape[0]
a_size = env.action_space


class Net(nn.Module):
    def __init__(self, s_size):
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(s_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, a_size)
        )

    def forward(self, s):
        q = self.layer(s)
        return q


class DQN():
    def __init__(self, memory_len, s_size, a_size, gamma):
        self.memory = np.zeros((Memory_capacity, s_size*2 + a_size))
        self.memory_len = memory_len
        self.eval, self.target = net(s_size), net(s_size)
        self.index = 0
        self.s_size = s_size
        self.a_size = a_size

        self.gamma = gamma
        self.opti, self.loss = torch.optim.Adam(self.eval.parameters(), lr=0.001), nn.MSELoss()

    def insert(self, s, a, r, s_):
        self.memory[self.index, :] = np.hstack(s, a, r, s_)
        self.index += 1
        if self.index > self.memory_len:
            self.index = 0

    def sample(self, batch_size):
        sample_index = np.random.choice(self.memory_len, batch_size)
        sample_memory = self.memory[sample_index, :]

        sample_s = torch.Tensor(sample_memory[:, self.s_size])
        sample_a = torch.LongTensor(sample_memory[:, self.s_size:(self.s_size+self.a_size)])
        sample_r = torch.Tensor(sample_memory[:, (self.s_size+self.a_size):(self.s_size+self.a_size+1)])
        sample_s_ = torch.Tensor(sample_memory[:, (self.s_size+self.a_size+1):])

        return sample_s, sample_a, sample_r, sample_s_

    def train(self, s, a, r, s_):
        q_eval = self.eval(s).gather(1, a)
        q_target = self.target(s_).detach().max(1)[0].view(-1, 1)
        self.opti.zero_grad()
        loss_q = self.loss(q_eval, self.gamma*q_target+r)
        loss_q.backward()
        self.opti.step()

    def choose_action(self, s):
        a = torch.max(q_eval(s), 1)[1].data.numpy()



gamma = 0.9
batch_size = 64
memory_len = 1024
epochs = 400

agent = DQN(memory_len, s_size, a_size, gamma)

for epoch in range(epochs):
    s = env.reset()
    while True:
        # loop......

        if done:
            break








