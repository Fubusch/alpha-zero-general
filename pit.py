import Arena
from AlphaBeta import AlphaBeta
from AlphaBetaMCTS import AlphaBetaMCTS
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

mini_othello = False  # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = True

if mini_othello:
    g = OthelloGame(6)
else:
    g = OthelloGame(8)

# all players
rp = RandomPlayer(g).play
gp = GreedyOthelloPlayer(g).play
hp = HumanOthelloPlayer(g).play



# nnet players
n1 = NNet(g)
if mini_othello:
    n1.load_checkpoint('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
else:
    n1.load_checkpoint('./pretrained_models/othello/pytorch/','8x8_100checkpoints_best.pth.tar')
args1 = dotdict({'numMCTSSims': 15, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

ap = AlphaBeta(g, n1, dotdict({'ab_depth': 9, 'kbest': 5, 'move_ordering': True})).play

ab_mcts = AlphaBetaMCTS(g, n1, dotdict({'numMCTSSims': 100, 'cpuct':1.0, 'prior_weight': 10, 'ab_params' : dotdict({'ab_depth': 3, 'move_ordering': True})}))
n2p = lambda x: np.argmax(ab_mcts.getActionProb(x, temp=0))

arena = Arena.Arena(ap, n1p, g, display=OthelloGame.display)

print(arena.playGames(2, verbose=True))
