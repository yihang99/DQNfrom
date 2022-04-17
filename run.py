import argparse
import time

import torch
import gym
import matplotlib.pyplot as plt
import cv2

import model
import utils
from parameters import *


def main():
    parser = argparse.ArgumentParser(description='Run DQN')
    parser.add_argument('-e', '--env', default='Breakout-v0', help='Atari env name')
    parser.add_argument('-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('-s', '--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()
    # args.input_shape = tuple(args.input_shape)

    # args.output = get_output_folder(args.output, args.env)

    env = gym.make(args.env, render_mode='human')
    # env = gym.make(args.env)
    dqn = model.DQN((num_stacked_frames, 84, 84), env.action_space.n)
    dqn.load_state_dict(torch.load('checkpoint.pth', map_location=torch.device('cpu')))

    buffer = utils.Buffer()
    frame = env.reset()
    state = buffer.stack_frames(frame, start_frame=True)
    print(type(frame), frame.shape)

    done = True
    step_idx = 0
    for t in range(50000):
        if done:  # if the last trajectory ends, start a new one
            print('Trajectory length: ', step_idx)
            step_idx = 0
            frame = env.reset()
            state = buffer.stack_frames(frame, start_frame=True)
        step_idx += 1

        action = dqn.act(state)
        for j in range(k):
            next_frame, reward, done, _ = env.step(action)
            if done:
                break
        next_state = buffer.stack_frames(next_frame)
        state = next_state

        # cv2.imshow('anim', state[0])
        # exit(1)

        # print(t, action)

    env.close()


if __name__ == '__main__':
    main()
