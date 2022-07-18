'''
Author: James Keal
Date: May 5, 2019
Board class.
Macro board data:
  1.=X, -1.=O, 0.=empty, -0.=blocked
Pieces board data:
  1.=X, -1.=O, 0.=empty
'''

from itertools import product
from Arena import numPlayers
import numpy as np

b_sz = 3


class Board:
    def __init__(self, n=b_sz):
        # Create empty macro and board arrays
        self.n = n
        self.macro = np.zeros((self.n, self.n))
        self.pieces = np.zeros((self.n**2, self.n**2))
        self.draw = np.finfo(self.pieces.dtype).eps

    def __getitem__(self, index):
        return self.pieces[index]

    def get_microboard(self, index):
        return self[tuple(slice(self.n*i, self.n*(i+1)) for i in index)]

    def getInnerBoards(self, bboard):
        boards = []

        for i in range(b_sz-2):
            for j in range(b_sz-2):
                miniboard = np.array(bboard[i:i+3, j:j+3])

                boards.append(miniboard)
        return boards

    def get_legal_moves(self, player):
        """Returns all legal moves for a given player."""
        moves = []
        for u in product(range(self.n), range(self.n)):
            if not self.macro[u] and 0 < np.copysign(1, self.macro[u]):
                for move in product(*(range(self.n*i, self.n*(i+1)) for i in u)):
                    if self.pieces[move] == 0:
                        moves.append(move)
        return moves

    def is_full(self, board=None):
        """Check whether a board is full."""
        if board is None:
            board = self.macro
        return board.all()

    def is_win(self, player, board=None):
        """Check whether a given player has a line in any direction."""
        if board is None:
            board = self.macro
        for i in range(len(board)):
            if (board[i, :] == player).all():
                return True
            if (board[:, i] == player).all():
                return True
        if (board.diagonal() == player).all():
            return True
        if (board[::-1].diagonal() == player).all():
            return True
        return False

    def execute_move(self, move, player):
        """Place a piece on the board and update the macro."""
        _u = tuple(int(i/self.n) for i in move)
        _v = tuple(int(i % self.n) for i in move)
        assert self.pieces[move] == 0
        self.pieces[move] = player
        uboard = self.get_microboard(_u)
        insides = self.getInnerBoards(uboard)

        if self.is_full(uboard):
            self.macro[_u] = self.draw

        for player in range(1, numPlayers+1):
            for b in insides:
                if self.is_win(player, b):
                    self.macro[_u] = player

        for u in product(range(self.n), range(self.n)):
            if not self.macro[u]:
                self.macro[u] = 0. if self.macro[_v] or u == _v else -0.
