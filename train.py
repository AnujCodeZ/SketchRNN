import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import Configs
from data import StrokesDataset
from test import Sampler


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=3e-3)
parser.add_argument('--epochs', type=int, default=100)
opt = parser.parse_args()
print(opt)

def train(opt):
    configs = Configs(opt.name, opt.batch_size, opt.lr, opt.epochs)

    data = configs.data

    train_set = StrokesDataset(data['train'], configs.max_seq_len)
    val_set = StrokesDataset(data['valid'], configs.max_seq_len, train_set.scale)

    trainloader = DataLoader(train_set, opt.batch_size)
    validloader = DataLoader(val_set, opt.batch_size)
    
    for epoch in range(opt.epochs):
        print(f"Epoch {epoch + 1}/{opt.epochs}")
        for data, mask in tqdm(trainloader):
            data = data.to(configs.device).transpose(0,1)
            mask = mask.to(configs.device).transpose(0,1)
            z, mu, sigma_hat = configs.encoder(data)
            
            z_stack = z.unsqueeze(0).expand(data.shape[0] - 1, -1, -1)
            inputs = torch.cat([data[:-1], z_stack], 2)
            
            dist, q_logits, _ = configs.decoder(inputs, z, None)
            
            kl_loss = configs.kldiv_loss(sigma_hat, mu)
            
            reconstruction_loss = configs.reconstruction_loss(mask, data[1:], dist, q_logits)
            
            loss = kl_loss + reconstruction_loss
            
            configs.optimizer.zero_grad()
            
            loss.backward()
            
            configs.optimizer.step()
        print(f"Loss: {loss}")
    
    with torch.no_grad():
        data, _ = val_set(np.random.choice(len(val_set)))
        data = data.unsqueeze(1).to(configs.device)
        Sampler.sample(data, configs.temperature)

if __name__=='__main__':
    train(opt)