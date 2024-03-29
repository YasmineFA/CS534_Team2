from Coach import Coach
from Arena import numPlayers
from ut3.UT3Game import UT3Game as Game
from ut3.pytorch.NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 100,
    'numEps': 20,
    'tempThreshold': 15,
    'updateThreshold': 0.52 if numPlayers == 2 else 0.34,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 10,
    'arenaCompare': 30,
    'cpuct': 2,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp/', 'temp.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

if __name__ == "__main__":
    g = Game()
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(*args.load_folder_file)

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
