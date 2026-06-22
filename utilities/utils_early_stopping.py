import torch

class StoppingCriterion:

    def __init__(self, max_compression_set_size, stop_criterion=torch.log(torch.tensor(2)),  patience=3,
                  use_early_stopping=True, use_p2l_stopping=True):
        
        self.max_compression_set_size = max_compression_set_size
        try:
            self.stop_criterion = stop_criterion.item()
        except AttributeError:
            self.stop_criterion = stop_criterion
        self.patience = patience
        self.use_early_stopping = use_early_stopping
        self.use_p2l_stopping= use_p2l_stopping
        self.iterations = 0
        self.min_loss = torch.inf
        self.stop = False

    def check_early_stop(self, loss):
        if not self.use_early_stopping:
            return True
        
        if loss < self.min_loss:
            self.min_loss = loss
            self.iterations = 0
            return True
        
        self.iterations += 1
        return not (self.iterations >= self.patience)
    
    def check_p2l_stop(self, max_error):
        return self.stop_criterion <= max_error
    
    def check_compression_set_stop(self, compression_set_size):
        return compression_set_size < self.max_compression_set_size
    
    def check_stop(self, loss, max_error, compression_set_size):
        self.stop = not (self.check_early_stop(loss)
                    and self.check_p2l_stop(max_error)
                    and self.check_compression_set_stop(compression_set_size)
                    )