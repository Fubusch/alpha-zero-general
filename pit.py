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

def get_mcts_player(num_mcts_sims=100):
    args1 = dotdict({'numMCTSSims': num_mcts_sims, 'cpuct': 1.0})
    mcts1 = MCTS(g, n1, args1)
    return lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

def get_alpha_beta_mcts_player(num_mcts_sims=100, kbest=3, depth=3, prior_weight=10, move_ordering=False):
    ab_mcts = AlphaBetaMCTS(g, n1, dotdict({'numMCTSSims': num_mcts_sims, 'cpuct':1.0, 'prior_weight': prior_weight, 'ab_params' : dotdict({'ab_depth': depth,'kbest': kbest, 'move_ordering': move_ordering})}))
    return lambda x: np.argmax(ab_mcts.getActionProb(x, temp=0))

if __name__ == '__main__':
    mini_othello = True  # Play in 6x6 instead of the normal 8x8.
    human_vs_cpu = True
    variance_net = True
    ab_mcts = False
    if mini_othello:
        g = OthelloGame(6)
        filepath = '6x100x25_best.pth.tar'
    else:
        g = OthelloGame(8)
        filepath = '8x8_100checkpoints_best.pth.tar'

    # all players
    rp = RandomPlayer(g).play
    gp = GreedyOthelloPlayer(g).play
    hp = HumanOthelloPlayer(g).play

    # nnet players
    n1 = NNet(g)
    n1.load_checkpoint('./pretrained_models/othello/pytorch/', filepath)

    if ab_mcts:
        for num_mcts_sims in [10, 15, 25, 50, 100]:
            for prior_weight in [5, 10, 15, 25, 50]:
                if prior_weight > num_mcts_sims:
                    continue
                ab_mcts = get_alpha_beta_mcts_player(num_mcts_sims=num_mcts_sims, kbest=None, depth=3, prior_weight=prior_weight,
                                                     move_ordering=False)
                n1p = get_mcts_player(num_mcts_sims=num_mcts_sims)
                arena = Arena.Arena(ab_mcts, n1p, g, display=OthelloGame.display)
                print(num_mcts_sims, prior_weight)
                print(arena.playGames(50, verbose=False))
    else:
        for num_mcts_sims in [10, 15, 25, 50, 100]:
            for kbest in [None]:
                for ab_depth in range(2,7):
                    n1p = get_mcts_player(num_mcts_sims=num_mcts_sims)
                    ap = AlphaBeta(g, n1, dotdict({'ab_depth': ab_depth, 'kbest': kbest, 'move_ordering': False})).play
                    arena = Arena.Arena(ap, rp, g, display=OthelloGame.display)
                    print(num_mcts_sims, kbest, ab_depth)
                    print(arena.playGames(20, verbose=False))


