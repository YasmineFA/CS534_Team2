
import numpy as np
from ut3.UT3Logic import Board, BOARD_SIZE
from Game import Game
import sys
sys.path.append('..')


class UT3Game(Game):
    def __init__(self, n=BOARD_SIZE):
        self.n = n

    def getArray(self, b):
        macro = np.tile(b.macro, (self.n, self.n))
        return np.stack((b.pieces, macro))

    def getBoardChannels(self):
        return 2

    def getBoardSize(self):
        return self.n**2, self.n**2

    def getActionSize(self):
        return self.n**4

    def getInitBoard(self):
        b = Board(self.n)
        return self.getArray(b)

    def getNextState(self, board, player, action):
        b = Board(self.n)
        b.pieces = np.copy(board[0])
        b.macro = np.copy(board[1, :BOARD_SIZE, :BOARD_SIZE])
        move = int(action/self.n**2), action % self.n**2
        b.execute_move(move, player)
        return self.getArray(b), -player

    def getValidMoves(self, board, player):
        valid = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board[0])
        b.macro = np.copy(board[1, :BOARD_SIZE, :BOARD_SIZE])
        for x, y in b.get_legal_moves(player):
            valid[x*self.n**2 + y] = 1
        return np.array(valid)

    def getGameEnded(self, board, player):
        # Return 0 if not ended, 1 if player 1 won, -1 if player 1 lost.
        # Return small non-zero value for a draw.
        b = Board(self.n)
        b.pieces = np.copy(board[0])
        b.macro = np.copy(board[1, :BOARD_SIZE, :BOARD_SIZE])
        for player in -1, 1:
            if b.is_win(player):
                return player
            if b.is_full():
                return b.draw
        return 0

    def getCanonicalForm(self, board, player):
        return np.where(board, player*board, board)

    def getSymmetries(self, board, pi):
        # rotate, mirror
        assert(len(pi) == self.getActionSize())  # 1 for pass
        pi_board = np.reshape(pi, self.getBoardSize())
        sym, x, y = [], -2, -1

        for rot in range(4):
            for flip in True, False:
                newB = np.rot90(board, rot, (x, y))
                newPi = np.rot90(pi_board, rot, (x, y))
                if flip:
                    newB = np.flip(newB, y)
                    newPi = np.flip(newPi, y)
                sym.append((newB, list(newPi.ravel())))
        return sym

    def stringRepresentation(self, board):
        return board.tostring()


def display(board, indent='  '):
    print('')

    # print(indent + '   0 | 1 | 2 ‖ 3 | 4 | 5 ‖ 6 | 7 | 8')
    # Creates row of numbers at top of the board
    topRow = '    '
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            topRow += str(i*BOARD_SIZE + j)
            if i*BOARD_SIZE + j < 10:
                topRow += ' '
            if j != BOARD_SIZE - 1:
                topRow += '| '
            elif i != BOARD_SIZE - 1:
                topRow += '‖ '
    print(indent + topRow)

    print('')
    for n, row in enumerate(board[0]):
        if n:
            if n % BOARD_SIZE:
                sep = '---' + ('+---' * (BOARD_SIZE - 1))
                # print(indent + '- ' + sep + '‖' + sep + '‖' + sep)
                fullPrint = ' - '
                for i in range(BOARD_SIZE):
                    fullPrint += sep
                    if i != BOARD_SIZE - 1:
                        fullPrint += '‖'
                print(indent + fullPrint)

            else:
                sep = '='*(4*BOARD_SIZE - 1)
                # print(indent + '= ' + sep + '#' + sep + '#' + sep)
                fullPrint = ' = '
                for i in range(BOARD_SIZE):
                    fullPrint += sep
                    if i != BOARD_SIZE - 1:
                        fullPrint += '#'
                print(indent + fullPrint)

        row = ' ‖ '.join(' | '.join(
            map(str, map(int, row[i:i+BOARD_SIZE]))) for i in range(0, len(row), BOARD_SIZE))
        adjustedIndent = indent
        if n < 10:
            adjustedIndent += ' '
        print(adjustedIndent + str(n) + '  ' + row.replace('-1',
              'O').replace('1', 'X').replace('0', '.'))
    print('')
