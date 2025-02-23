import logging

import coloredlogs
from torch.nn.functional import mse_loss

from Coach import Coach
from othello.OthelloGame import OthelloGame
from othello.pytorch.NNet import NNetWrapper as NNet
from utils import *
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch
import numpy as np
import os

from tqdm import tqdm

args = dotdict({
    'numIters': 1,
    'numEps': 500,              # Number of complete self-play games to simulate during a new iteration.
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
    train_full_net = False
    g = OthelloGame(6)

    filepath = '6x100x25_best.pth.tar'
    folder = './pretrained_models/othello/pytorch/'
    variance_net_filepath = 'variance_net_' + filepath

    az_general = NNet(g)
    az_general.load_checkpoint(folder, filepath)
    abz = NNet(g, True)
    abz.load_checkpoint(folder, variance_net_filepath)

    c = Coach(g, az_general, args)
    example_file = './temp/prepared_examples.pt'
    if os.path.isfile(example_file):
        prepared_examples = torch.load(example_file)
    else:
        train_examples = c.get_train_examples(0)
        prepared_examples = [(torch.tensor(x0.copy(), device='cuda', dtype=torch.float), torch.tensor(x1, device='cuda', dtype=torch.float), torch.tensor(x2, device='cuda', dtype=torch.float), torch.tensor(x3, device='cuda', dtype=torch.float)) for x0,x1,x2,x3 in train_examples]
        torch.save(prepared_examples, example_file)
    only_variances = torch.tensor([p[3] for p in prepared_examples])
    slice_size = 500
    bins = 30
    histogram = torch.histogram(only_variances, bins).hist.to(int)
    count_sum = 0
    balanced_examples = []
    for i in range(bins - 1):
        count_sum += histogram[i].item()
        prepared_slice = prepared_examples[count_sum:count_sum+histogram[i+1].item()]
        random_indices = torch.multinomial(torch.ones(histogram[i+1].item()), slice_size, False)
        balanced_examples.extend([example for i, example in enumerate(prepared_slice) if i in random_indices])
    train_size = int(len(balanced_examples) * 0.8)
    data_loader_train = DataLoader(balanced_examples[:train_size], batch_size=512, shuffle=True, drop_last=True)
    data_loader_test = DataLoader(balanced_examples[train_size:], batch_size=512, shuffle=True, drop_last=True)
    if train_full_net:
        optimizer = AdamW(abz.nnet.parameters(), lr=1e-2)
    else:
        optimizer = AdamW([abz.nnet.fc5.weight], lr=1e-2)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
    epochs = 25
    with torch.no_grad():
        print_test_loss(abz, data_loader_test)
    for i in tqdm(range(epochs)):
        for board, policy, value, variance in data_loader_train:
            pi, v, var = abz.nnet(board)
            var_loss = mse_loss(var, variance)
            total_loss = var_loss
            if train_full_net:
                pi_loss = abz.loss_pi(policy, pi)
                value_loss = abz.loss_v(value, v)
                total_loss += pi_loss + value_loss
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        with torch.no_grad():
            losses = print_test_loss(abz, data_loader_test)
            scheduler.step(np.mean(losses))
    abz.save_checkpoint(folder, variance_net_filepath)


def print_test_loss(abz, data_loader_test):
    var_losses = []
    pi_losses = []
    value_losses = []
    for board, policy, value, variance in data_loader_test:
        pi, v, var = abz.nnet(board)
        var_loss = mse_loss(var, variance)
        pi_loss = abz.loss_pi(policy, pi)
        value_loss = abz.loss_v(value, v)
        var_losses.append(var_loss.item())
        pi_losses.append(pi_loss.item())
        value_losses.append(value_loss.item())
    print("var:",np.mean(var_losses), "pi:",np.mean(pi_losses), "value:",np.mean(value_losses) )
    return var_losses


if __name__ == "__main__":
    main()
