from Coach import Coach
from ut3.UT3Game import UT3Game as Game
from ut3.pytorch.NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 1000,           # number of total iterations
    'numEps': 50,              # number of self-play games per iteration
    'tempThreshold': 15,
    'updateThreshold': 0.52,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 10,
    'arenaCompare': 40,         # number of games played to determine if model has improved
    'cpuct': 2,

    'checkpoint': './temp/',
    'load_model': True,
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
