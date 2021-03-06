import os
import math
import torch
import torch.nn as nn
import numpy as np


def load_data(name: str):
    data = np.load(os.path.join('data', name + '.npz'), 
                                encoding='latin1', 
                                allow_pickle=True)
    return data
    
class BivariateGuassianMixture:
    def __init__(self, pi_logits, mu_x, mu_y, sigma_x, sigma_y, rho_xy):
        self.pi_logits = pi_logits
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.rho_xy = rho_xy
        
    @property
    def n_distributions(self):
        return self.pi_logits.shape[-1]
    
    def set_temperature(self, temperature):
        
        self.pi_logits /= temperature
        
        self.sigma_x *= math.sqrt(temperature)
        self.sigma_y *= math.sqrt(temperature)
    
    def get_distribution(self):
        
        sigma_x = torch.clamp_min(self.sigma_x, 1e-5)
        sigma_y = torch.clamp_min(self.sigma_y, 1e-5)
        rho_xy = torch.clamp(self.rho_xy, -1 + 1e-5, 1 - 1e-5)
        
        mean = torch.stack([self.mu_x, self.mu_y], -1)
        
        cov = torch.stack([
            sigma_x * sigma_x, rho_xy * sigma_x * sigma_y,
            rho_xy * sigma_x * sigma_y, sigma_y * sigma_y
        ], -1)
        cov = cov.view(*sigma_y.shape, 2, 2)
        
        multi_dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        
        cat_dist = torch.distributions.Categorical(logits=self.pi_logits)
        
        return cat_dist, multi_dist

class ReconstructionLoss():
    
    def __call__(self, mask, target, dist, q_logits):
        
        pi, mix = dist.get_distribution()
        xy = target[:, :, 0:2].unsqueeze(-2).expand(-1, -1, dist.n_distributions, -1)
        probs = torch.sum(pi.probs * torch.exp(mix.log_prob(xy)), 2)
        
        loss_stroke = -torch.mean(mask * torch.log(1e-5 + probs))
        loss_pen = -torch.mean(target[:, :, 2:] * q_logits)
        
        return loss_stroke + loss_pen

class KLDivLoss():
    
    def __call__(self, sigma_hat, mu):
        
        return -0.5 * torch.mean(1 + sigma_hat - mu ** 2 - torch.exp(sigma_hat))
    