
import torch
from torch import nn
from ut3.UT3Logic import b_sz

class UT3NNet(nn.Module):
    def __init__(self, game, args):
        self.size = game.getBoardSize()
        self.channels = game.getBoardChannels()
        self.actions = game.getActionSize()
        self.args = args

        super(UT3NNet, self).__init__()
        self.drop = nn.Dropout(self.args.dropout)
        self.norm = nn.BatchNorm1d(4*(b_sz**2)*args.width)
        self.relu, self.tanh = nn.ReLU(), nn.Tanh()
        self.soft = nn.LogSoftmax(dim=1)

        self.conv0 = nn.Conv2d(self.channels, args.width, 3, 1, 1)

        self.conv1 = nn.Conv2d(self.channels + args.width, args.width,
            kernel_size=(b_sz,b_sz), stride=b_sz)
        self.conv2 = nn.Conv2d(self.channels + args.width, (b_sz)*args.width,
            kernel_size=(b_sz,(b_sz**2)), stride=b_sz)
        self.conv3 = nn.Conv2d(self.channels + args.width, (b_sz)*args.width,
            kernel_size=((b_sz**2),b_sz), stride=b_sz)
        self.conv4 = nn.Conv2d(self.channels + args.width, (b_sz**2)*args.width,
            kernel_size=((b_sz**2),(b_sz**2)))

        self.out_pi = nn.Linear(4*(b_sz**2)*args.width, self.actions)
        self.out_v = nn.Linear(4*(b_sz**2)*args.width, 1)

    def forward(self, x):
        x = torch.cat((x, self.conv0(x)), dim=1)
        x1 = self.conv1(x).view(-1, (b_sz**2)*self.args.width)
        x2 = self.conv2(x).view(-1, (b_sz**2)*self.args.width)
        x3 = self.conv3(x).view(-1, (b_sz**2)*self.args.width)
        x4 = self.conv4(x).view(-1, (b_sz**2)*self.args.width)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.relu(self.norm(x))

        pi = self.out_pi(x)
        v = self.out_v(x)

        return self.soft(pi), self.tanh(v)
