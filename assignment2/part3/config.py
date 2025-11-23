import argparse

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['gcn', 'matrix-gcn', 'gat'])
    return parser.parse_args()

def get_config():
    config = parse_args()
    config.num_epochs = 50

    config.n_layers = 2
    config.dropout = 0.1
    config.hidden_dim = 124

    config.lr = 1e-3
    config.weight_decay = 1e-5

    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")

    else:
        config.device = torch.device("mps")

    config.progress_bar = True
    config.log_dir = './logs'

    config.seed = 42
    return config