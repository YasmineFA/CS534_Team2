from Arena import numPlayers
from ut3.UT3Game import display
import random
import numpy as np
import math

EPS = 1e-8


class Player():
    def __init__(self, game):
        self.game = game
        self.playerNumber = None


class RandomPlayer(Player):
    def __init__(self, game):
        Player.__init__(self, game)

    def play(self, board):
        valid = self.game.getValidMoves(board, 1)
        while True:
            a = random.randrange(self.game.getActionSize())
            if valid[a]:
                return a


class HumanUT3Player(Player):
    def __init__(self, game):
        Player.__init__(self, game)

    def play(self, board):
        valid = self.game.getValidMoves(board, 1)
        print('Valid moves:')
        print(', '.join(str(int(i/self.game.n**2))+' '+str(int(i % self.game.n**2))
                        for i, v in enumerate(valid) if v))
        while True:
            a = input()
            x, y = [int(x) for x in a.split(' ')]
            a = self.game.n**2 * x + y
            if valid[a]:
                break
            else:
                print('Invalid')
        return a


class MinMaxUT3Player(Player):
    def __init__(self, game, depth=2):
        Player.__init__(self, game)
        self.depth = depth
        self.end = {}
        self.valid = {}
        self.draw = np.finfo(float).eps

    def search(self, board, depth):
        key = self.game.stringRepresentation(board)

        if key not in self.end:
            self.end[key] = np.empty(numPlayers)

            for i in range(numPlayers):
                j = i+1
                winner = self.game.getGameEnded(board, j)

                if winner == j:
                    self.end[key][i] = winner
                elif winner == self.draw:
                    self.end[key][i] = self.draw
                else:
                    self.end[key][i] = 0

        if key not in self.valid:
            self.valid[key] = [a for a, val in enumerate(
                self.game.getValidMoves(board, 1)) if val]

        if any(self.end[key]):

            return self.end[key], None

        if depth == 0:
            return self.end[key], random.choice(self.valid[key])

        value_action = []

        active_player = self.playerNumber

        for a in self.valid[key]:
            next_board, next_player = self.game.getNextState(
                board, active_player, a)
            next_board = self.game.getCanonicalForm(next_board, next_player)
            active_player = (active_player % numPlayers)+1
            value_action.append((self.search(next_board, depth-1)[0], a))

        wins = [(v, a) for v, a in value_action if any(v == active_player)]
        if len(wins):
            value, action = random.choice(wins)
            return value, action

        unknowns = [(v, a) for v, a in value_action if all(v)]
        if len(unknowns):
            value, action = random.choice(unknowns)
            return value, action

        draws = [(v, a) for v, a in value_action if any(v < 1)]
        if len(draws):
            value, action = random.choice(draws)
            return value, action

        value, action = random.choice(value_action)
        return value, action

    def play(self, board):
        temp = self.search(board, self.depth)
        # print(temp)
        return temp[1]


class MCTSUT3Player(Player):
    def __init__(self, game, args):
        Player.__init__(self, game)
        self.args = args

    def play(self, board):
        valid = [a for a, val in enumerate(
            self.game.getValidMoves(board, 1)) if val]
        move_evaluations = []  # evaluations of each move, parallel to valid
        for i in range(len(valid)):
            move = valid[i]
            # print("TESTING MOVE:", move)
            move_evaluations.append(self.evaluateAction(board, move))
            # print("MOVE OUTCOME:", move_evaluations[-1])

        best_move_index = 0
        best_eval = move_evaluations[0]
        for i in range(len(valid)):
            if move_evaluations[i] > best_eval:
                best_move_index = i
                best_eval = move_evaluations[i]

        return valid[best_move_index]

    def evaluateAction(self, board, move):
        # runs the given number of playouts, returns the average value of the playouts
        resulting_board, next_player = self.game.getNextState(
            board, self.playerNumber, move)
        total = 0
        for _ in range(self.args.numPlayoutsPerMove):
            total += self.runPlayout(resulting_board, next_player)

        return total/self.args.numPlayoutsPerMove

    def runPlayout(self, board, current_player):
        # returns 1 for a win, -1 for a loss, and 0 for a draw

        while self.game.getGameEnded(board, 1) == 0:
            # display(board)
            move = self.getRandomMove(board)
            # print("Move made", move)
            board, current_player = self.game.getNextState(
                board, current_player, move)

        result = self.game.getGameEnded(board, 1)

        if result > 0 and result < 1:   # draw
            return 0
        if result == self.playerNumber:  # win
            return 1
        else:                           # loss
            return -1

    def getRandomMove(self, board):
        valid = [a for a, val in enumerate(
            self.game.getValidMoves(board, 1)) if val]
        return random.choice(valid)
