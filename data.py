from typing import Tuple, Optional, Any

import torch
import numpy as np
from torch.utils.data import Dataset


class StrokesDataset(Dataset):
    def __init__(self, dataset: np.array, max_seq_len: int, scale: Optional[float] = None):
        
        data = []
        for seq in dataset:
            if 10 < len(seq) <= max_seq_len:
                # clamp seq between [-1000, 1000]
                seq = np.maximum(seq, -1000)
                seq = np.minimum(seq, 1000)
                
                seq = np.array(seq, dtype=np.float32)
                data.append(seq)
        # calculating standard scale for x and y
        if scale is None:
            scale = np.std(np.concatenate([np.ravel(s[:, 0:2]) for s in data]))
        self.scale = scale
        
        longest_seq_len = max([len(seq) for seq in data])
        
        # data of size (batch_size, longest_seq_len + start and end of seq, 
        # (del_x, del_y, p1, p2, p3))
        self.data = torch.zeros(len(data), longest_seq_len + 2, 5, dtype=torch.float)
        
        # mask of size (batch_size, longest_seq_len + one step prediction)
        self.mask = torch.zeros(len(data), longest_seq_len + 1)
        
        for i, seq in enumerate(data):
            
            seq = torch.from_numpy(seq)
            len_seq = len(seq)
            
            self.data[i, 1:len_seq+1, :2] = seq[:, :2] / scale # del_x, del_y
            self.data[i, 1:len_seq+1, 2] = 1 - seq[:, 2] # p1
            self.data[i, 1:len_seq+1, 3] = seq[:, 2] # p2
            self.data[i, 1:len_seq+1, 4] = 1 # p3
            
            self.mask[i, :len_seq + 1] = 1 # end of seq
            
        self.data[:, 0, 2] = 1 # start of seq
    
    def __len__(self) -> int:
        
        return len(self.data)
    
    def __getitem__(self, index: int):
        
        return self.data[index], self.mask[index]