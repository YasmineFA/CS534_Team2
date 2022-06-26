
import torch
from torch import nn
from ut3.UT3Logic import BOARD_SIZE  # board size constant


class UT3NNet(nn.Module):
    def __init__(self, game, args):
        self.size = game.getBoardSize()
        self.channels = game.getBoardChannels()
        self.actions = game.getActionSize()
        self.args = args

        super(UT3NNet, self).__init__()
        self.drop = nn.Dropout(self.args.dropout)
        self.norm = nn.BatchNorm1d(4*(BOARD_SIZE**2)*args.width)
        self.relu, self.tanh = nn.ReLU(), nn.Tanh()
        self.soft = nn.LogSoftmax(dim=1)

        self.conv0 = nn.Conv2d(self.channels, args.width, BOARD_SIZE, 1, 1)

        self.conv1 = nn.Conv2d(self.channels + args.width, args.width,
                               kernel_size=(BOARD_SIZE, BOARD_SIZE), stride=3)
        self.conv2 = nn.Conv2d(self.channels + args.width, BOARD_SIZE*args.width,
                               kernel_size=(BOARD_SIZE, (BOARD_SIZE**2)), stride=3)
        self.conv3 = nn.Conv2d(self.channels + args.width, BOARD_SIZE*args.width,
                               kernel_size=((BOARD_SIZE**2), BOARD_SIZE), stride=3)
        self.conv4 = nn.Conv2d(self.channels + args.width, (BOARD_SIZE**2)*args.width,
                               kernel_size=((BOARD_SIZE**2), (BOARD_SIZE**2)))

        self.out_pi = nn.Linear(4*(BOARD_SIZE**2)*args.width, self.actions)
        self.out_v = nn.Linear(4*(BOARD_SIZE**2)*args.width, 1)

    def forward(self, x):
        y = self.conv0(x)
        print(y[0][0][0])
        x = torch.cat((x, y), dim=1)
        x1 = self.conv1(x).view(-1, (BOARD_SIZE**2)*self.args.width)
        x2 = self.conv2(x).view(-1, (BOARD_SIZE**2)*self.args.width)
        x3 = self.conv3(x).view(-1, (BOARD_SIZE**2)*self.args.width)
        x4 = self.conv4(x).view(-1, (BOARD_SIZE**2)*self.args.width)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.relu(self.norm(x))

        pi = self.out_pi(x)
        v = self.out_v(x)

        return self.soft(pi), self.tanh(v)
