from torch.optim import Optimizer

# adapted from pytorch official implementation. 
class SGD(Optimizer):
    r"""Implements stochastic gradient descent.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """
    def __init__(self, params, lr, weight_decay=0):
        print("Using optimizer: SGD")
        defaults = dict(lr=lr, weight_decay=weight_decay)
        
        super(SGD, self).__init__(params, defaults)

    def step(self):
        """Performs a single optimization step.
        """
        #print([i for i in self.param_groups])
        for group in self.param_groups:
            
            
            weight_decay = group['weight_decay']
        
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                    
                p.data.add_(-group['lr'], d_p)
