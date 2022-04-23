import argparse
import random
import time
import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
import cv2

import utils
import model
from parameters import *
from utils import plot_stats

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)


def main():
    parser = argparse.ArgumentParser(description='Run DQN')
    parser.add_argument('-e', '--env', default='Breakout-v0', help='Atari env name')
    parser.add_argument('-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('-s', '--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()
    log_file_name = 'logs/' + time.ctime().replace(' ', '_')[4:20] + args.env + '.txt'

    env = gym.make(args.env)
    dqn = model.DQN((num_stacked_frames, 108, 84), env.action_space.n).to(device)
    target_dqn = model.DQN((num_stacked_frames, 108, 84), env.action_space.n).to(device)

    target_dqn.load_state_dict(dqn.state_dict())

    optimizer = torch.optim.Adam(dqn.parameters(), lr=learning_rate)
    buffer = utils.Buffer()
    frame = env.reset()
    next_frame = np.zeros(frame.shape)
    state = buffer.stack_frames(frame, start_frame=True)
    print(type(frame), frame.shape)

    done = True
    step_idx = 0
    episode_reward = 0
    episode_rewards = []
    loss = 0
    losses = []
    for t in range(NUMBER_OF_TRAINING_STEPS):
        if done:  # if the last trajectory ends, start a new one
            frame = env.reset()
            state = buffer.stack_frames(frame, start_frame=True)
            print('Trajectory length: ', step_idx, '  Episode reward: ', episode_reward)
            step_idx = 0

            episode_rewards.append(episode_reward)
            episode_reward = 0
        step_idx += 1

        if random.random() < eps_max - t / NUMBER_OF_TRAINING_STEPS * (eps_max - eps_min):
            action = env.action_space.sample()
        else:
            action = dqn.act(state).item()

        reward_sum = 0.
        for j in range(k):
            next_frame, reward, done, _ = env.step(action)
            reward_sum += 1. if reward > 0 else 0.
            # reward_sum += reward
            if done:
                break
        next_state = buffer.stack_frames(next_frame)
        buffer.push_transition(state, action, next_state, reward_sum, done)
        state = next_state
        episode_reward += reward_sum

        if buffer.len() > batch_size:
            states, actions, next_states, rewards, dones = buffer.sample_transition(batch_size)

            states = torch.tensor(np.float32(states)).to(device)
            next_states = torch.tensor(np.float32(next_states)).to(device)
            actions = torch.tensor(actions).to(device)
            rewards = torch.tensor(rewards).to(device)
            dones = torch.FloatTensor(dones).to(device)

            loss = utils.compute_loss(dqn, target_dqn, states, actions, next_states, rewards, dones)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if t % Q_NETWORK_RESET_INTERVAL == 0:
            target_dqn.load_state_dict(dqn.state_dict())
            print('t =', t, 'Net reset here')

        if t % SAVE_CKPT_INTERVAL == 0:
            ckpt_ind = int(t / SAVE_CKPT_INTERVAL)
            torch.save(dqn.state_dict(), 'ckpts_single_new2/dqn_single_ckpt_{:0>2d}.pth'.format(ckpt_ind))

        env.close()


if __name__ == '__main__':
    main()
