
import numpy as np
from colorama import Back
from .UT3Logic import Board, b_sz
from Arena import numPlayers
from Game import Game
import sys
sys.path.append('..')


class UT3Game(Game):
    def __init__(self, n=b_sz):
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
        b.macro = np.copy(board[1, :b_sz, :b_sz])
        move = int(action/self.n**2), action % self.n**2
        b.execute_move(move, player)
        if player+1 <= numPlayers:
            player = player+1
        else:
            player = 1
        return self.getArray(b), player

    def getValidMoves(self, board, player):
        valid = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board[0])
        b.macro = np.copy(board[1, :b_sz, :b_sz])
        for x, y in b.get_legal_moves(player):
            valid[x*self.n**2 + y] = 1
        return np.array(valid)

    def getGameEnded(self, board, player):
        # Return 0 if not ended, 1 if player 1 won, -1 if player 1 lost.
        # Return small non-zero value for a draw.
        b = Board(self.n)
        b.pieces = np.copy(board[0])
        b.macro = np.copy(board[1, :b_sz, :b_sz])

        insides = b.getInnerBoards(b.macro)

        for i in range(len(insides)):
            for player in range(1, numPlayers+1):
                if b.is_win(player, insides[i]):
                    return player
                if b.is_full():
                    return b.draw
        return 0

    def getCanonicalForm(self, board, player):
        arr = []
        arr.append(np.where(board[0] > 0, (board[0] % numPlayers)+1, board[0]))
        arr.append(board[1])
        # np.where((board >= 1.0) & (board < numPlayers+1), (board%numPlayers)+1 , board)
        return np.array(arr)

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
    b = Board(b_sz)
    b.pieces = np.copy(board[0])
    b.macro = np.copy(board[1, :b_sz, :b_sz])

    top = '     '
    sep = '|'
    msep = '‖ '
    rowSep = '    ' + ('---' + '+---'*(b_sz-1) + '‖') * \
        (b_sz-1) + '---' + '+---'*(b_sz-1)
    rows = []
    idex = []
    jdex = []
    windex = []
    minirows = []
    mw = []

    for i in range(b_sz):
        for j in range(b_sz):
            bp = b.macro[i][j]

            if bp in range(1, 4):
                idex.append(i)
                jdex.append(j)
                mw.append(bp)

                windex.append(tuple((i, j)))

    print('')

    for x in range(b_sz):
        for y in range(b_sz):
            adder = ''

            for xX in range(b_sz):

                minirow = ''

                for yY in range(b_sz):

                    if ((y*b_sz+yY) % (b_sz) == b_sz-1):
                        if (y == b_sz-1) & (yY == b_sz-1):
                            sep = ' '
                        else:
                            sep = '‖'
                    else:
                        sep = '|'

                    if (x, y) in windex:
                        dex = windex.index((x, y))

                        mask = mw[dex]

                        if mask == 1:
                            adder = Back.RED + str(b.pieces[x*b_sz+xX][y*b_sz+yY]).replace(
                                '1.0', ' X ').replace('2.0', ' O ').replace('3.0', ' ^ ') + Back.RESET
                        elif mask == 2:
                            adder = Back.BLUE + str(b.pieces[x*b_sz+xX][y*b_sz+yY]).replace(
                                '1.0', ' X ').replace('2.0', ' O ').replace('3.0', ' ^ ') + Back.RESET
                        elif mask == 3:
                            adder = Back.GREEN + str(b.pieces[x*b_sz+xX][y*b_sz+yY]).replace(
                                '1.0', ' X ').replace('2.0', ' O ').replace('3.0', ' ^ ') + Back.RESET

                    else:
                        temp = b.pieces[x*b_sz+xX][y*b_sz+yY]

                        if temp == 1.0:
                            adder = Back.RED + \
                                str(b.pieces[x*b_sz+xX][y*b_sz+yY]
                                    ).replace('1.0', ' X ') + Back.RESET
                        elif temp == 2.0:
                            adder = Back.BLUE + \
                                str(b.pieces[x*b_sz+xX][y*b_sz+yY]
                                    ).replace('2.0', ' O ') + Back.RESET
                        elif temp == 3.0:
                            adder = Back.GREEN + \
                                str(b.pieces[x*b_sz+xX][y*b_sz+yY]
                                    ).replace('3.0', ' ^ ') + Back.RESET
                        else:
                            adder = str(b.pieces[x*b_sz+xX][y*b_sz+yY])

                    minirow += adder + sep

                minirows.append(minirow)

    t = 0

    for r in range(b_sz**2):
        if r < 10:
            row = ' ' + str(r) + '  '

            if r % b_sz == b_sz-1:
                top += str(r) + ' ' + msep
            else:
                top += str(r) + ' | '
        else:
            row = str(r) + '  '

            if r % b_sz == b_sz-1:
                if r == (b_sz**2)-1:
                    top += str(r)
                else:
                    top += str(r) + msep
            else:
                top += str(r) + '| '

        for c in range(b_sz):

            t = (r*b_sz)+(c*b_sz)-(r % b_sz)*(b_sz-1)

            row += minirows[t]
            """if r in range(b_sz):
    			t = r+c*b_sz
    			row += minirows[t]
    		elif r in range(b_sz, b_sz*(b_sz-1)):
    			t = r+(b_sz**2)+c*b_sz-3
    			row += minirows[t]
    		else:
    			t = r+(b_sz**2)+(c*b_sz)+b_sz
    			row += minirows[t]"""

        rows.append(row)

        if r % b_sz == b_sz-1:
            if (r != 0) & (r != (b_sz**2)-1):
                rows.append('    ' + ('='*((4*b_sz)-1) + '#')
                            * (b_sz-1) + '='*((4*b_sz)-1))
        else:
            rows.append(rowSep)

    print(top)

    print('')

    for b in range(len(rows)):
        print(rows[b].replace('0.0', ' . '))

    """for rn in range(b_sz**2):
    	 row = ''
    	 sep = '|'
    	 mSep = '‖ '
    	 
    	 if rn < 10:
    	 	row += ' ' + str(rn) + '  '
    	 	
    	 	if rn%b_sz == b_sz-1:
    	 		top += str(rn) + ' ' + mSep
    	 	else:
    	 		top += str(rn) + ' | '
    	 else:
    	 	row += str(rn) + '  '
    	 	
    	 	if rn%b_sz == b_sz-1:
    	 		if rn == (b_sz**2)-1:
    	 			top += str(rn)
    	 		else:
    	 			top += str(rn) + mSep
    	 	else:
    	 		top += str(rn) + '| '
    	 
    	 for cn in range(b_sz**2):
    	 	if cn%b_sz == b_sz-1:
    	 		if cn==(b_sz**2)-1:
    	 			sep = ''
    	 		else:
    	 			sep = '‖'
    	 	else:
    	 		sep = '|'
    	
    	 	if (idex > 0) & (jdex > 0):
    	 		if (rn in range(b_sz*idex, b_sz*idex+b_sz)) & (cn in range(b_sz*jdex, b_sz*jdex+b_sz)):
    	 			# print("({}, {}) {}".format(idex, jdex, mw))
    	 			if mw == 1:
    	 				row += Back.RED + str(b.pieces[rn, cn]).replace('1.0', ' X ').replace('2.0', ' O ').replace('3.0', ' ^ ') + Back.RESET + sep
    	 			elif mw == 2:
    	 				row += Back.BLUE + str(b.pieces[rn, cn]).replace('2.0', ' O ').replace('3.0', ' ^ ').replace('1.0', ' X ') + Back.RESET + sep
    	 			elif mw == 3:
    	 				row += Back.GREEN + str(b.pieces[rn, cn]).replace('3.0', ' ^ ').replace('2.0', ' O ').replace('1.0', ' X ') + Back.RESET + sep
    	 		else:
    	 			temp = b.pieces[rn, cn]
    	 			
    	 			if temp == 1.0:
    	 				row += str(b.pieces[rn, cn]).replace('1.0', Back.RED + ' X ' + Back.RESET) + sep
    	 			elif temp == 2.0:
    	 				row += str(b.pieces[rn, cn]).replace('2.0', Back.BLUE + ' O ' + Back.RESET) + sep
    	 			elif temp == 3.0:
    	 				row += str(b.pieces[rn, cn]).replace('3.0', Back.GREEN + ' ^ ' + Back.RESET) + sep
    	 			else:
    	 				row += str(b.pieces[rn, cn]) + sep
    	 	else:
    	 		temp = b.pieces[rn, cn]
    	 			
    	 		if temp == 1.0:
    	 			row += str(b.pieces[rn, cn]).replace('1.0', Back.RED + ' X ' + Back.RESET) + sep
    	 		elif temp == 2.0:
    	 			row += str(b.pieces[rn, cn]).replace('2.0', Back.BLUE + ' O ' + Back.RESET) + sep
    	 		elif temp == 3.0:
    	 			row += str(b.pieces[rn, cn]).replace('3.0', Back.GREEN + ' ^ ' + Back.RESET) + sep
    	 		else:
    	 			row += str(b.pieces[rn, cn]) + sep
         
    	 if rn%b_sz == 0:
    		 if rn != 0:
    		 	rows.append('    ' + ('='*((b_sz**2)-1) + '#')*(b_sz-1) + '='*((4*b_sz)-1))
    	 else:
    	 	 rows.append(rowSep)
         	
    	 rows.append(row) #replace('1.0', Back.RED + ' X ' + Back.RESET).replace('2.0', Back.BLUE + ' O ' + Back.RESET).replace('3.0', Back.GREEN + ' ^ ' + Back.RESET).replace('0.0','   ')) #replace('1.0', ' X ').replace('2.0', ' O ').replace('3.0', ' ^ '))
    	 
    print(top)
    
    for i in range(len(rows)):
    	print(rows[i].replace('0.0', ' . '))"""

    """print('')
    # print(indent + '    0 | 1 | 2 ‖ 3 | 4 | 5 ‖ 6 | 7 | 8')
    topRow = '     '
    for i in range(b_sz):
    	for j in range(b_sz):
    		topRow += str(i*b_sz + j)
    		if i*b_sz + j < 10:
    			topRow += ' '
    		if j != b_sz - 1:
    			topRow += '| '
    		elif i != b_sz - 1:
    			topRow += '‖ '
    print(indent + topRow)
    print('')
    
    for n, row in enumerate(board[0]):
        if n:
            if n % b_sz:
                sep = '---' + '+---'*(b_sz-1)
                
                fullPrint = ' -  '
                for i in range(b_sz):
                	fullPrint += sep
                	if i != b_sz - 1:
                		fullPrint += '‖'
                print(indent + fullPrint)
                # print(indent + '-  ' + sep + '‖' + sep + '‖' + sep)
            else:
                sep = '='*(4*b_sz - 1)
                
                fullPrint = ' =  '
                for i in range(b_sz):
                	fullPrint += sep
                	if i != b_sz - 1:
                		fullPrint += '#'
                print(indent + fullPrint)
                # print(indent + '= ' + sep + '#' + sep + '#' + sep)
        row = '‖'.join('|'.join(map(str, map(int, row[i:i+b_sz]))) for i in range(0, len(row), b_sz))
        adjustedIndent = indent
        if n < 10:
        	adjustedIndent += ' '
        print(adjustedIndent + str(n) + '  ' + row.replace('1', Back.BLUE + ' O ' + Back.RESET).replace('2', Back.RED + ' X ' + Back.RESET).replace('3', Back.GREEN + ' ^ ' + Back.RESET).replace('0',' . '))"""
    print('')
