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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description='Run DQN')
    parser.add_argument('-e', '--env', default='Breakout-v0', help='Atari env name')
    parser.add_argument('-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('-s', '--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()
    # args.input_shape = tuple(args.input_shape)

    # args.output = get_output_folder(args.output, args.env)

    env = gym.make(args.env)
    dqn = model.DQN((num_stacked_frames, 84, 84), env.action_space.n).to(device)
    target_dqn = model.DQN((num_stacked_frames, 84, 84), env.action_space.n).to(device)
    dqn.load_state_dict(torch.load('checkpoint.pth', map_location='cpu'))
    target_dqn.load_state_dict(torch.load('checkpoint.pth', map_location='cpu'))

    optimizer = torch.optim.Adam(dqn.parameters(), lr=learning_rate)
    buffer = utils.Buffer()
    frame = env.reset()
    next_frame = np.zeros(frame.shape)
    state = buffer.stack_frames(frame, start_frame=True)
    print(type(frame), frame.shape)

    done = True
    step_idx = 0
    episode_rewards = []
    losses = []
    episode_reward = 0.0
    for t in range(NUMBER_OF_TRAINING_STEPS):
        if done:  # if the last trajectory ends, start a new one
            frame = env.reset()
            state = buffer.stack_frames(frame, start_frame=True)
            print('Trajectory length: ', step_idx)
            step_idx = 0
        step_idx += 1
        episode_reward += reward

        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        reward_sum = 0
        for j in range(k):
            next_frame, reward, done, _ = env.step(action)
            reward_sum += reward
            if done:
                break
        next_state = buffer.stack_frames(next_frame)
        buffer.push_transition(state, action, next_state, reward_sum, done)
        state = next_state

        if buffer.len() > batch_size:
            states, actions, next_states, rewards, dones = buffer.sample_transition(batch_size)

            states = torch.tensor(np.float32(states)).to(device)
            next_states = torch.tensor(np.float32(next_states)).to(device)
            actions = torch.tensor(actions).to(device)
            rewards = torch.tensor(rewards).to(device)
            dones = torch.FloatTensor(dones).to(device)

            current_q = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q = target_dqn(next_states).max(1)[0]
            expected_current_q = rewards + gamma * next_q * (1 - dones)
            loss = (current_q - expected_current_q.data).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if step_idx % 10000 == 0:
                plot_stats(step_idx, episode_rewards, losses)

            if t % Q_NETWORK_RESET_INTERVAL == 0:
                torch.save(dqn.state_dict(), 'checkpoint.pth')
                target_dqn.load_state_dict(torch.load('checkpoint.pth', map_location='cpu'))
                print('t = ', t, '  loss = ', loss, '  action = ', action)

            # assert False,'I stop here'

    env.close()


if __name__ == '__main__':
    main()
