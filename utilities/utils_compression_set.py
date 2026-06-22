import torch

class CompressionSetIndexes(torch.Tensor):
    def __init__(self, n : int ):
        super().__init__()
        self.complement_set = torch.ones(n, dtype=torch.bool) # True if the data is in the validation set

    def get_complement_size(self):
        return int(self.complement_set.sum())
    
    def get_compression_size(self):
        return int(self.complement_set.shape[0] - self.get_complement_size())
    
    def get_complement_data(self):
        return self.complement_set
    
    def get_compression_data(self):
        return ~self.complement_set
    
    def update_compression_set(self, indices) -> None:
        self.complement_set[indices] = False

    def correct_idx(self,indices):
        return self.complement_set.nonzero()[indices]