import sys
from fileinput import filename

from othello.pytorch.OthelloNNet import OthelloNNet

sys.path.append('..')
from utils import *
from othello.OthelloGame import OthelloGame

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class OthelloVarianceNNet(nn.Module):
    def __init__(self, game, args):
        super(OthelloVarianceNNet, self).__init__()
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        self.onnet = OthelloNNet(game, args)

        self.fc5 = nn.Linear(512, 1)

    def forward(self, s):
        s = self.onnet.get_intermediate_representation(s)

        pi = self.onnet.fc3(s)                                                                         # batch_size x action_size
        v = self.onnet.fc4(s)                                                                          # batch_size x 1
        var = self.fc5(s)  # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v), torch.sigmoid(var)


if __name__ == '__main__':
    args = dotdict({
        'lr': 0.001,
        'dropout': 0.3,
        'epochs': 10,
        'batch_size': 64,
        'cuda': torch.cuda.is_available(),
        'num_channels': 512,
    })
    alpha_zero_general_checkpoints = {
        6: ('../../pretrained_models/othello/pytorch/', '6x100x25_best.pth.tar'),
        8: ('../../pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')}
    for othello_size, (folder, filename) in alpha_zero_general_checkpoints.items():
        g = OthelloGame(othello_size)
        ovn = OthelloVarianceNNet(g, args)
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception("No model in path {}".format(filepath))
        map_location = None if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        ovn.onnet.load_state_dict(state_dict=checkpoint['state_dict'])
        filepath = os.path.join(folder, 'variance_net_' + filename)
        torch.save({
            'state_dict': ovn.state_dict(),
        }, filepath)
