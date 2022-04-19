import argparse
import random
import time
import numpy as np
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
    parser.add_argument('-v', '--vision', action='store_true', default=False,
                        help='Display video if True')

    args = parser.parse_args()
    # args.input_shape = tuple(args.input_shape)

    # args.output = get_output_folder(args.output, args.env)

    if args.vision:
        env = gym.make(args.env, render_mode='human')
    else:
        env = gym.make(args.env)
        
    env = Monitor(env, "recording", force=True)
    dqn = model.DQN((num_stacked_frames, 108, 84), env.action_space.n)
    dqn.load_state_dict(torch.load('ckpts/dqn2_ckpt.pth', map_location=torch.device('cpu')))

    buffer = utils.Buffer()
    frame = env.reset()
    state = buffer.stack_frames(frame, start_frame=True)
    print(type(frame), frame.shape)

    done = True
    step_idx = 0
    episode_reward = 0
    episode_rewards = []
    while len(episode_rewards) < 20 + 1:
        if done:  # if the last trajectory ends, start a new one
            print('Trajectory length: ', step_idx, '  Episode reward: ', episode_reward)
            step_idx = 0
            episode_rewards.append(episode_reward)
            episode_reward = 0
            frame = env.reset()
            state = buffer.stack_frames(frame, start_frame=True)
        step_idx += 1

        action = dqn.act(state)
        # action = env.action_space.sample()

        reward_sum = 0
        for j in range(k):
            next_frame, reward, done, _ = env.step(action)
            reward_sum += 1. if reward > 0 else 0.
            # reward_sum += reward
            if done:
                break
        next_state = buffer.stack_frames(next_frame)
        state = next_state
        episode_reward += reward_sum

        cv2.imshow('anim', abs(state[0] - state[num_stacked_frames - 1]))
        if 1:#reward_sum != 0:
            pass
            # print(t, action, reward_sum, dqn(torch.tensor(np.float32(state)).unsqueeze(0)))

    env.close()
    print("Avg Epi Rwd: ", sum(episode_rewards) / 20)


if __name__ == '__main__':
    main()
