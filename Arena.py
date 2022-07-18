import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time

numPlayers = 2

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, players, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.players = players
        self.game = game
        self.display = display
        #self.assignPositions(self.players)
        
    def assignPositions(self, players):
    	for x in players:
    		from ut3.UT3Players import MinMaxUT3Player
    		
    		if isinstance(x, type(MinMaxUT3Player(self.game).play)):
    			print('Found MinMaxPlayer')
    			#x.__self__.pos = (players.index(x)+numPlayers)%numPlayers
    			#x.__self__.active = (players.index(x)+numPlayers)%numPlayers

    def getNextPlayer(self, players, player):
        nextPlayer = players[0]

        if player >= numPlayers:
            return nextPlayer
        else:
            nextPlayer = players[player]
            return nextPlayer

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = self.players

        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert(self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)

            action = self.getNextPlayer(players, curPlayer)(
                self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(
                self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                print(action)
                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            assert(self.display)
            print("Game over: Turn ", str(it), "Result ",
                  str(self.game.getGameEnded(board, 1)))
            self.display(board)
        # print(board)
        return self.game.getGameEnded(board, 1)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        num = int(num/2)
        playerWins = [0]*numPlayers
        draws = 0
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                playerWins[0] += 1
            elif gameResult == 2:
                playerWins[1] += 1
            elif gameResult == 3:
                playerWins[2] += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps, maxeps=maxeps, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()

        self.players[0], self.players[1] = self.players[1], self.players[0]
        #self.assignPositions(self.players)

        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 2:
                playerWins[0] += 1
            elif gameResult == 1:
                playerWins[1] += 1
            elif gameResult == 3:
                playerWins[2] += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps, maxeps=maxeps, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()

        bar.finish()

        return playerWins, draws
