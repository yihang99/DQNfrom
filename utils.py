import random
import time

import numpy as np
import torch
import cv2

import os
import gym
import matplotlib.pyplot as plt
from IPython.display import clear_output

from parameters import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_stats(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title(f'Total frames {frame_idx}. Avg reward over last 10 episodes: {np.mean(rewards[-10:])}')
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()


def compute_loss(dqn, target_dqn, states, actions, next_states, rewards, dones):
    current_q = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q = target_dqn(next_states).max(1)[0]
    expected_current_q = rewards + gamma * next_q * (1 - dones)
    loss = (current_q - expected_current_q.data).pow(2).mean()
    return loss


class Buffer(object):
    def __init__(self, capacity=buffer_capacity):
        super(Buffer, self).__init__()
        self.num_stacked_frames = num_stacked_frames
        self.current_state = np.zeros((k, *compressed_size), dtype=np.float)
        self.buffer_list = []
        self.capacity = capacity
        self.buffer_index = 0

    def stack_frames(self, frame, start_frame=False):
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(grey_frame, compressed_size[::-1]).reshape(1, *compressed_size) / 256.

        if start_frame:
            self.current_state = np.zeros((self.num_stacked_frames, *compressed_size), dtype=np.float)
        self.current_state = np.concatenate((resized_frame, self.current_state[:self.num_stacked_frames - 1]))

        return self.current_state.copy()

    def push_transition(self, state, action, next_state, reward, done):
        if len(self.buffer_list) < self.capacity:
            self.buffer_list.append((state, action, next_state, reward, done))
        else:
            self.buffer_list[self.buffer_index] = (state, action, next_state, reward, done)
            self.buffer_index = (self.buffer_index + 1) % self.capacity

    def sample_transition(self, batch_size):
        sampled_transitions = random.sample(self.buffer_list, batch_size)
        states, actions, next_states, rewards, dones = zip(*sampled_transitions)
        return np.stack(states), actions, np.stack(next_states), rewards, dones

    def len(self):
        return len(self.buffer_list)


def main():
    buffer = Buffer()
    frame = np.random.randn(210, 160, 3) * 0.01 + 0.5
    frame = np.ones((210, 160, 3), dtype=np.uint8)
    print(buffer.stack_frames(frame)[:, 1, 1])
    print(buffer.stack_frames(frame * 4)[:, 1, 1])
    print(buffer.stack_frames(frame * 9)[:, 1, 1])


if __name__ == '__main__':
    main()
