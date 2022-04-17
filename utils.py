import random

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
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title(f'Total frames {frame_idx}. Avg reward over last 10 episodes: {np.mean(rewards[-10:])}')
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()



class Buffer(object):
    def __init__(self, capacity=1000000):
        super(Buffer, self).__init__()
        self.num_stacked_frames = num_stacked_frames
        self.current_state = np.zeros((k, 84, 84), dtype=np.float)
        self.buffer_list = []
        self.capacity = capacity

    def stack_frames(self, frame, start_frame=False):
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(grey_frame, (110, 84))[0:84, 18:102].reshape(1, 84, 84) / 256.
        if start_frame:
            self.current_state = np.zeros((self.num_stacked_frames, 84, 84), dtype=np.float)
        self.current_state = np.concatenate((resized_frame, self.current_state[:self.num_stacked_frames - 1]))

        return self.current_state.copy()

    def push_transition(self, state, action, next_state, reward, done):
        if len(self.buffer_list) <= self.capacity:
            self.buffer_list.append((state, action, next_state, reward, done))
        else:
            self.buffer_list[random.randint(0, self.capacity - 1)] = (state, action, next_state, reward, done)

    def sample_transition(self, batch_size):
        sampled_transitions = random.sample(self.buffer_list, batch_size)
        states, actions, next_states, rewards, dones = zip(*sampled_transitions)
        # print(states, actions, next_states, rewards, dones)
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
