import logging

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class AlphaBeta():
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.evals = {}

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
        current_best_eval = -1
        current_best_move = -1
        if len(valid_moves) == 1:
            return valid_moves[0]
        for move in valid_moves:
            state_after_move, next_player = self.game.getNextState(canonicalBoard, current_player, move)
            eval = self.search(state_after_move, next_player, self.args.ab_depth - 1)
            if eval == 1:
                return move
            if eval >= current_best_eval:
                current_best_eval = eval
                current_best_move = move
        return current_best_move

    def get_valid_moves(self, canonicalBoard, player):
        return np.argwhere(self.game.getValidMoves(canonicalBoard, player) == 1).flatten()

    def search(self, ccurrent_board, currentPlayer, depth=0, alpha=-1, beta=1):
        game_ended = self.game.getGameEnded(ccurrent_board, 1)
        if game_ended != 0:
            return game_ended
        elif depth == 0:
            _, v = self.nnet.predict(ccurrent_board)
            return v
        moves = self.get_valid_moves(ccurrent_board, currentPlayer)
        if currentPlayer == 1:
            current_best_eval = self.get_best_eval(alpha, beta, ccurrent_board, depth - 1, moves, max)
        else:
            current_best_eval = self.get_best_eval(alpha, beta, ccurrent_board, depth - 1, moves, min)
        return current_best_eval

    def get_best_eval(self, alpha, beta, ccurrent_board, depth, moves, max_or_min):
        current_best_eval = -max_or_min(-1, 1)
        for move in moves:
            state_after_move, next_player = self.game.getNextState(ccurrent_board, 1, move)
            eval = self.search(state_after_move, next_player, depth - 1, alpha, beta)
            current_best_eval = max_or_min(current_best_eval, eval)
            beta = max_or_min(beta, current_best_eval)
            if beta <= alpha:
                return current_best_eval
        return current_best_eval
