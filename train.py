import argparse
import pandas as pd
import numpy as np

import torch
from  torchfm.model.afn import AdaptiveFactorizationNetwork
from torch.utils.data import DataLoader
from dataloader import TrainDataset


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_csv', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=20)

    parser.add_argument('--embed_dims', type=int, default=16)
    parser.add_argument('--LNN_dim', type=int, default=1500)
        
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    return args


def train_one_epoch(model, loader, criterion, device, optimizer):

    model.train()

    for i, (fields, targets) in enumerate(loader):
        
        fields, targets = fields.to(device), fields.to(device)
        y = model(fields)

        loss = criterion(targets.float(), y)

        model.zero_grad()
        loss.backward()
        optimizer.step()

def validate_model(model, loader, device):

    model.to(device)

    for i, (fields, targets) in enumerate(loader):

        y = model(fields)
        

if __name__ == '__main__':

    args = get_args()
    
    model = AdaptiveFactorizationNetwork(field_dims=22, embed_dims=args.embed_dims, LNN_dim=args.LNN_dim,
                                         mlp_dims=(400, 400, 400), dropouts=(0, 0, 0))

    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = args.lr,
                                 weight_decay=1e-6)

    
