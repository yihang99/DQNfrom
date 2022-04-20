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

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)


def main():
    parser = argparse.ArgumentParser(description='Run DQN')
    parser.add_argument('-e', '--env', default='Breakout-v0', help='Atari env name')
    parser.add_argument('-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('-s', '--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()

    env = gym.make(args.env)

    dqn = model.DQN((num_stacked_frames, 108, 84), env.action_space.n).to(device)

    points = []
    for ckpt_ind in range(int(NUMBER_OF_TRAINING_STEPS / Q_NETWORK_RESET_INTERVAL)):
        #dqn.load_state_dict(torch.load('ckpts_single/dqn_single_ckpt_59.pth', map_location=torch.device('cpu')))
        dqn.load_state_dict(torch.load('ckpts_single/dqn_single_ckpt_{:0>2d}.pth'.format(ckpt_ind), map_location=torch.device('cpu')))

        buffer = utils.Buffer()
        frame = env.reset()
        state = buffer.stack_frames(frame, start_frame=True)
        print(type(frame), frame.shape)

        done = True
        step_idx = 0
        truncated_episode_reward = 0
        episode_reward = 0
        truncated_episode_rewards = []
        episode_rewards = []
        while len(episode_rewards) < 20 + 1:
            if done:  # if the last trajectory ends, start a new one
                print('Trajectory length: ', step_idx,
                      '  Truncated Episode reward: ', truncated_episode_reward,
                      '  Episode reward: ', episode_reward)
                step_idx = 0
                truncated_episode_rewards.append(truncated_episode_reward)
                truncated_episode_reward = 0
                episode_rewards.append(episode_reward)
                episode_reward = 0
                frame = env.reset()
                state = buffer.stack_frames(frame, start_frame=True)
            step_idx += 1

            action = dqn.act(state)

            truncated_reward_sum = 0
            reward_sum = 0

            for j in range(k):
                next_frame, reward, done, _ = env.step(action)
                truncated_reward_sum += 1. if reward > 0 else 0.
                reward_sum += reward
                if done:
                    break

            next_state = buffer.stack_frames(next_frame)
            state = next_state
            truncated_episode_reward += truncated_reward_sum
            episode_reward += reward_sum

        avg_truncated_episode_rewards = sum(truncated_episode_rewards) / 20
        avg_episode_rewards = sum(episode_rewards) / 20
        print("Avg Trun Epi Rwd: ", avg_truncated_episode_rewards,
              "  Avg Epi Rwd: ", avg_episode_rewards)
        with open('Sim_Result.txt', 'a') as f:
            print(str(avg_truncated_episode_rewards)+', '+str(avg_episode_rewards), file=f)
    env.close()


if __name__ == '__main__':
    main()
