import torch
import matplotlib.pyplot as plt


class Sampler:
    
    def __init__(self, encoder, decoder):
        
        self.encoder = encoder
        self.decoder = decoder
        
    def sample(self, data, temperature):
        
        longest_seq_len = len(data)
        z, _, _ = self.encoder(data)
        s = data.new_tensor([0, 0, 1, 0, 0])
        seq = [s]
        
        state = None
        with torch.no_grad():
            for i in range(longest_seq_len):
                
                data = torch.cat([s.view(1, 1, -1), z.unsqueeze(0)], 2)
                dist, q_logits, state = self.decoder(data, z, state)
                s = self._sample_step(dist, q_logits, temperature)
                seq.append(s)
                
                if s[4] == 1:
                    break
        
        seq = torch.stack(seq)
        self.plot(seq)
    
    @staticmethod
    def _sample_step(dist, q_logits, temperature):
        
        dist.set_temperature(temperature)
        pi,mix = dist.get_distribution()
        idx = pi.sample()[0, 0]
        
        q = torch.distributions.Categorical(logits=q_logits / temperature)
        q_idx = q.sample()[0, 0]
        
        xy = mix.sample()[0, 0, idx]
        
        stroke = q_logits.new_zeros(5)
        stroke[:2] = xy
        stroke[q_idx + 2] = 1
        
        return stroke
    
    @staticmethod
    def plot(seq):
        
        seq[:, 0:2] = torch.cumsum(seq[:, 0:2], dim=0)
        seq[:, 2] = seq[:, 3]
        seq = seq[:, 0:3].detach().cpu().numpy()
        
        strokes = np.split(seq, np.where(seq[:, 2] > 0)[0] + 1)
        for s in strokes:
            plt.plot(s[:, 0], -s[:, 1])
        
        plt.axis('off')
        plt.show()
        
        
        