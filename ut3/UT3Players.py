import random
import numpy as np
import math

EPS = 1e-8

from Arena import numPlayers


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid = self.game.getValidMoves(board, 1)
        while True:
            a = random.randrange(self.game.getActionSize())
            if valid[a]:
                return a


class HumanUT3Player():
    def __init__(self, game):
        self.game = game

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


class MinMaxUT3Player():
    def __init__(self, game, depth=2):
        self.game = game
        self.depth = depth
        self.end = {}
        self.valid = {}
        self.draw = np.finfo(float).eps

    def search(self, board, depth, curPlayer):
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

        active_player = 1
	
        for a in self.valid[key]:
            next_board, next_player = self.game.getNextState(board, active_player, a)
            next_board = self.game.getCanonicalForm(next_board, next_player)
            active_player = (active_player%numPlayers)+1
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
        #print(temp)
        return temp[1]
        
class MCTSUT3Player():
    """This class handles the MCTS tree."""

    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)
        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        valids = self.game.getValidMoves(canonicalBoard, 1)
        counts = [self.Nsa[(s, a)] if (
            s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA] = 1
            return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        
        probMat = valids*counts*probs
        
        #print(probMat)
        
        best_move = max(probMat)

        possible = [a for a, val in enumerate(probMat) if val == best_move]

        return random.choice(possible)

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        #canonicalBoard %= (numPlayers)
        #canonicalBoard %= numPlayers+1
        s = self.game.stringRepresentation(canonicalBoard)
        # print(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return self.Es[s]

        if s not in self.Ps:
            # leaf node

            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = valids/len(valids)
            v = self.game.getGameEnded(canonicalBoard, 1)
            # print(valids)
            # print(canonicalBoard)
            self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
                # print(canonicalBoard)
            else:
                # if all valid moves were masked make all valid moves equally probable
                print(canonicalBoard)
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])
                # print(v)

            self.Vs[s] = valids
            self.Ns[s] = 0

            return v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct*self.Ps[s][a] * \
                        math.sqrt(self.Ns[s])/(1+self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * \
                        self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] *
                                self.Qsa[(s, a)] + v)/(self.Nsa[(s, a)]+1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return self.Qsa[(s, a)]

    def play(self, board):
    
        return self.getActionProb(board)

