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

def get_alpha_beta_mcts_player(num_mcts_sims=100, kbest=3, depth=3, prior_weight=10, move_ordering=False, num_visits_before_ab=0, second_half=False, variance_net=False, variance_threshold=0):
    if variance_net:
        net = abz
    else:
        net = n1
    ab_mcts = AlphaBetaMCTS(g, net, dotdict({'numMCTSSims': num_mcts_sims, 'cpuct':1.0, 'prior_weight': prior_weight, 'second_half': second_half, 'num_visits_before_ab': num_visits_before_ab, 'variance_threshold': variance_threshold, 'ab_params' : dotdict({'ab_depth': depth,'kbest': kbest, 'move_ordering': move_ordering})}), variance_net=variance_net)
    return lambda x: np.argmax(ab_mcts.getActionProb(x, temp=0))

if __name__ == '__main__':
    mini_othello = True  # Play in 6x6 instead of the normal 8x8.
    human_vs_cpu = True
    variance_net = True
    ab_mcts = True
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

    abz = NNet(g, variance_net=variance_net)
    abz.load_checkpoint('./pretrained_models/othello/pytorch/', 'variance_net_' + filepath)

    if ab_mcts:
        for second_half in [False, True]:
                for variance_threshold in [0, 0.25, 0.5, 0.75, 0.9, 1.5]:
                    #for num_mcts_sims in [10, 15, 25, 50]:
                    for num_mcts_sims in [25]:
                        for prior_weight in [1, 5, 10, 15, 25]:#, 50]:
                            if prior_weight > num_mcts_sims:
                                continue
                            ab_mcts = get_alpha_beta_mcts_player(num_mcts_sims=num_mcts_sims, kbest=None, depth=2, prior_weight=prior_weight,
                                                                 move_ordering=False, second_half=second_half, variance_net=variance_net, variance_threshold=variance_threshold)
                            n1p = get_mcts_player(num_mcts_sims=num_mcts_sims)
                            arena = Arena.Arena(ab_mcts, n1p, g, display=OthelloGame.display)
                            print(f"Experiment: sh:{second_half}, sims:{num_mcts_sims}, pw:{prior_weight}, variance_threshold:{variance_threshold}")
                            print(arena.playGames(100, verbose=False))
    else:
        for num_mcts_sims in [10, 15, 25, 50, 100]:
            for kbest in [None]:
                for ab_depth in range(2,7):
                    n1p = get_mcts_player(num_mcts_sims=num_mcts_sims)
                    ap = AlphaBeta(g, n1, dotdict({'ab_depth': ab_depth, 'kbest': kbest, 'move_ordering': False})).play
                    arena = Arena.Arena(ap, rp, g, display=OthelloGame.display)
                    print(num_mcts_sims, kbest, ab_depth)
                    print(arena.playGames(20, verbose=False))


