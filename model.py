import torch
import torch.nn as nn
import numpy as np

from parameters import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.mlp(x)
        return x

    def act(self, state, device=device):
        state = torch.tensor(np.float32(state)).unsqueeze(0).to(device)
        q_value = self.forward(state)
        action = q_value.max(1)[1].data[0]
        return action


def main():
    shape = num_stacked_frames, 84, 84
    m = DQN(shape, 5)
    x = torch.randn(3, *shape)
    print(m(x).shape)
    print(m(x))


if __name__ == '__main__':
    main()
