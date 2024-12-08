import logging

import coloredlogs
from torch.nn.functional import mse_loss

from Coach import Coach
from othello.OthelloGame import OthelloGame
from othello.pytorch.NNet import NNetWrapper as NNet
from utils import *
from torch.optim import AdamW
from torch.utils.data import DataLoader

args = dotdict({
    'numIters': 1,
    'numEps': 5,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

def main():
    g = OthelloGame(6)
    filepath = '6x100x25_best.pth.tar'

    # nnet players
    n1 = NNet(g)
    n1.load_checkpoint('./pretrained_models/othello/pytorch/', filepath)


    c = Coach(g, n1, args)

    train_examples = c.get_train_examples(2)
    data_loader = DataLoader(train_examples, batch_size=1, shuffle=True)
    optimizer = AdamW(n1.nnet.parameters())
    epochs = 100
    for i in range(epochs):
        for batch in data_loader:
            board, _, _, variance = batch
            _, _, var = n1.nnet(board)
            var_loss = mse_loss(var, variance)
            var_loss.backward()
            optimizer.step()
            optimizer.zero_grad()


if __name__ == "__main__":
    main()
