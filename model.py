import torch
import einops
import torch.nn as nn

from utils import BivariateGuassianMixture


class Encoder(nn.Module):
    def __init__(self, d_z: int, enc_hidden_size: int):
        super().__init__()
        
        self.lstm = nn.LSTM(5, enc_hidden_size, bidirectional=True)
        
        self.mu_head = nn.Linear(2 * enc_hidden_size, d_z)
        self.sigma_head = nn.Linear(2 * enc_hidden_size, d_z)
        
    def forward(self, inputs, state=None):
        
        _, (hidden, cell) = self.lstm(inputs.float(), state)
        
        hidden = einops.rearrange(hidden, 'fb b h -> b (fb h)')
        
        mu = self.mu_head(hidden)
        
        sigma_hat = self.sigma_head(hidden)
        
        sigma = torch.exp(sigma_hat / 2.)
        
        z = mu + sigma * torch.normal(mu.new_zeros(mu.shape), mu.new_ones(mu.shape))
        
        return z, mu, sigma

class Decoder(nn.Module):
    def __init__(self, d_z: int, dec_hidden_size: int, n_distributions: int):
        super().__init__()
        
        self.lstm = nn.LSTM(d_z + 5, dec_hidden_size)
        
        self.init_state = nn.Linear(d_z, dec_hidden_size * 2)
        
        self.mixture = nn.Linear(dec_hidden_size, 6 * n_distributions)
        
        self.q_head = nn.Linear(dec_hidden_size, 3)
        
        self.q_log_softmax = nn.LogSoftmax(-1)
        
        self.dec_hidden_size = dec_hidden_size
        self.n_distributions = n_distributions
    
    def forward(self, x, z, state):
        
        if state is None:
        
            h, c = torch.split(torch.tanh(self.init_state(z)), self.dec_hidden_size, 1)
            state = (h.unsqueeze(0).contiguous(), c.unsqueeze(0).contiguous())
        
        outputs, state = self.lstm(x, state) 
        
        q_logits = self.q_log_softmax(self.q_head(outputs))
        
        pi_logits, mu_x, mu_y, sigma_x, sigma_y, rho_xy = \
            torch.split(self.mixture(outputs), self.n_distributions, 2)
        
        dist = BivariateGuassianMixture(pi_logits, mu_x, mu_y, 
                                        torch.exp(sigma_x), torch.exp(sigma_y),torch.tanh(rho_xy))
        
        return dist, q_logits, state