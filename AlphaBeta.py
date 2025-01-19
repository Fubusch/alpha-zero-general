import logging

import numpy
import numpy as np
from collections import defaultdict
from numba import jit, njit

EPS = 1e-8

log = logging.getLogger(__name__)

def get_stable_discs(stable_discs, canonicalBoard, board_dims):
    for corner in [(0,0), (board_dims[0] -1, 0), (0, board_dims[1]-1), tuple(dim - 1 for dim in board_dims)]:
        corner_value = canonicalBoard[corner]
        if corner_value != 0:
            stable_discs[corner_value][corner] = 1
        else:
            continue
        stable_discs[corner_value] = get_stable_discs_for_corner(canonicalBoard, corner, corner_value, stable_discs[corner_value])
    return stable_discs

def get_stable_discs_for_corner(canonicalBoard, corner, corner_value, stable_discs):
    max_i = [0, 0]
    for dim in range(2):
        if corner[dim] > 0:
            step = -1
        else:
            step = 1
        for i in range(1, canonicalBoard.shape[dim]):
            field_index = [min(c, s - 1) for c, s in zip(corner, canonicalBoard.shape)]
            field_index[dim] += (i * step)
            if stable_discs[field_index[0], field_index[1]] == 1:
                break
            if canonicalBoard[field_index[0], field_index[1]] == corner_value:
                stable_discs[field_index[0], field_index[1]] = 1
            else:
                max_i[dim] = step*i
                break
    if abs(max_i[0]) > 1 and abs(max_i[1]) > 1:
        canonicalSlice = np.ones_like(canonicalBoard)
        if (max_i[0] < 0):
            canonicalSlice[:max_i[0]] = 0
        else:
            canonicalSlice[max_i[0]:] = 0
        if(max_i[1] < 0):
            canonicalSlice[:, :(max_i[1])] = 0
        else:
            canonicalSlice[:, max_i[1]:] = 0
        stable_discs[canonicalSlice.astype(np.bool_)] = get_stable_discs_for_corner(canonicalBoard[canonicalSlice.astype(np.bool_)].reshape(abs(max_i[0]), abs(max_i[1])), corner, corner_value, stable_discs[canonicalSlice.astype(numpy.bool_)].reshape(abs(max_i[0]), abs(max_i[1]))).flatten()
    return stable_discs


class AlphaBeta():
    def __init__(self, game, nnet, args, variance_net=False):
        self.game = game
        self.evaluation_fuction = nnet.predict# self.evaluation_function
        self.policy_head = True
        self.args = args
        self.evals = defaultdict(dict)
        self.game_ended = {}
        self.variance_net = variance_net

    def evaluation_function(self, canonicalBoard: np.array):
        board_dims = canonicalBoard.shape
        stable_discs = {-1: np.zeros(board_dims), 1: np.zeros(board_dims)}
        stable_discs = get_stable_discs(stable_discs, canonicalBoard, board_dims)
        stable_disc_values = {key: 10 * val.sum() for key, val in stable_discs.items()}
        num_moves_player = sum(self.game.getValidMoves(canonicalBoard, 1))
        num_moves_opponent = sum(self.game.getValidMoves(canonicalBoard, -1))
        return np.clip((stable_disc_values[1] - stable_disc_values[-1] + num_moves_player - num_moves_opponent) / np.multiply(*canonicalBoard.shape), -1, 1)

    def play(self, canonicalBoard):
        """
        This function performs alpha beta search with ab_depth starting from
        canonicalBoard.

        Returns:
            move
        """
        # NOTE: IN CANONICAL BOARD, WE ARE ALWAYS PLAYER 1
        current_player = 1
        valid_moves = self.get_valid_moves(canonicalBoard, current_player)
        if len(valid_moves) == 1:
            return valid_moves[0]
        if self.args.move_ordering:
            pi, v = self.evaluation_fuction(canonicalBoard)
            valid_moves = valid_moves[pi[valid_moves].argsort()][:self.args.kbest]
        _, current_best_move = self.get_best_eval(-1, 1, canonicalBoard, self.args.ab_depth, valid_moves, current_player)
        return current_best_move

    def get_valid_moves(self, canonicalBoard, player):
        return np.argwhere(self.game.getValidMoves(canonicalBoard, player) == 1).flatten()

    def search(self, ccurrent_board, currentPlayer, depth=0, alpha=-1, beta=1, pi=None):
        s = self.game.stringRepresentation(ccurrent_board)
        if s not in self.game_ended:
            self.game_ended[s] = self.game.getGameEnded(ccurrent_board, 1)
        if self.game_ended[s] != 0:
            return self.game_ended[s]
        if s in self.evals[depth]:
            return self.evals[depth][s]
        if depth <= 0:
            if self.variance_net:
                _, self.evals[depth][s], var = self.evaluation_fuction(ccurrent_board)
            else:
                self.evals[depth][s] = self.evaluation_fuction(ccurrent_board)
            return self.evals[depth][s]
        moves = self.get_valid_moves(ccurrent_board, currentPlayer)
        if self.args.move_ordering:
            if pi is None:
                pi, _ = self.evaluation_fuction(ccurrent_board)
            moves = moves[(currentPlayer * pi[moves]).argsort()][:self.args.get('kbest')]
        self.evals[depth][s], _ = self.get_best_eval(alpha, beta, ccurrent_board, depth, moves, currentPlayer)
        return self.evals[depth][s]

    def get_best_eval(self, alpha, beta, ccurrent_board, depth, moves, currentPlayer):
        current_best_eval = -currentPlayer
        current_prefered_move = -1
        for move in moves:
            state_after_move, next_player = self.game.getNextState(ccurrent_board, currentPlayer, move)
            eval = self.search(state_after_move, next_player, depth - 1, alpha, beta)
            if currentPlayer == 1:
                current_best_eval = max(current_best_eval, eval)
                alpha = max(alpha, current_best_eval)
            else:
                current_best_eval = min(current_best_eval, eval)
                beta = min(beta, current_best_eval)
            if current_best_eval == eval:
                current_prefered_move = move
            if beta <= alpha:
                return current_best_eval, current_prefered_move
        return current_best_eval, current_prefered_move
