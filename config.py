import torch

from utils import load_data, ReconstructionLoss, KLDivLoss
from model import Encoder, Decoder


class Configs:
    def __init__(self, name, batch_size=128, lr = 3e-3, epochs=100):
        self.data = load_data(name)
        
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')
        
        self.enc_hidden_size = 256
        self.dec_hidden_size = 512
        
        self.batch_size = batch_size
        
        self.d_z = 128
        self.n_distributions = 20
        self.kldiv_loss_weight = 0.5
        
        self.grad_clip = 1.
        self.temperature = 0.4
        self.max_seq_len = 200
        
        self.lr = lr
        self.epochs = epochs
        
        self.encoder = Encoder(self.d_z, self.enc_hidden_size).to(self.device)
        self.decoder = Decoder(self.d_z, self.dec_hidden_size, self.n_distributions).to(self.device)
        
        self.reconstruction_loss = ReconstructionLoss()
        self.kldiv_loss = KLDivLoss()
        
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), 
                                          lr=self.lr)
        
        
        