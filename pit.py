import Arena
from MCTS import MCTS
from ut3.UT3Game import UT3Game, display
from ut3.UT3Players import *
from ut3.pytorch.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = UT3Game()

# all players
rp1 = RandomPlayer(g).play
rp2 = RandomPlayer(g).play
hp = HumanUT3Player(g).play
mp1 = MinMaxUT3Player(g, 0).play
mp2 = MinMaxUT3Player(g, 3).play
mcp = MCTSUT3Player(g, args=dotdict({'numPlayoutsPerMove': 20})).play

# nnet players
n1 = NNet(g)
# n1.load_checkpoint('./temp/', 'best.pth.tar')
args1 = dotdict({'numMCTSSims': 10, 'cpuct': 2})
mcts1 = MCTS(g, n1, args1)
def np1(x): return np.argmax(mcts1.getActionProb(x, temp=0))


n2 = NNet(g)
# n2.load_checkpoint('/dev/8x50x25/','best.pth.tar')
args2 = dotdict({'numMCTSSims': 10, 'cpuct': 2})
mcts2 = MCTS(g, n2, args2)
def np2(x): return np.argmax(mcts2.getActionProb(x, temp=0))


arena = Arena.Arena([rp1, mcp], g, display=display)

print(arena.playGames(20, verbose=False))
