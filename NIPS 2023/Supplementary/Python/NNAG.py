from torch.optim import Optimizer
import torch
# adapted from pytorch official implementation for SGD. 
class NNAG(Optimizer):
    r"""Implements Noisy Nesterov Accelerated Gradient method.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        t_k (float): accumlating step-size
        s_k (float): decreasing step-size
        a (float): tuning parameter usually equal to Lipschitz constant L
        
    """
    def __init__(self, params, v ,lr ,t_k,s_k , a,weight_decay=0,k=0,alpha=3/4):
        print("Using optimizer: NNAG")

        defaults = dict(v=v,lr=lr,t_k=lr,s_k=lr,a=a,weight_decay=weight_decay , k=k,alpha=alpha)
        
        super(NNAG , self).__init__(params,  defaults)
    
    
    def opt__init(self):
        """ Can be used for testing, otherwise not used in the main stream
        """
        self.param_groups[0]['k'] =0
        self.param_groups[0]['t_k']=0
      

    def step(self):
        """Performs a single optimization step.
        """
  
        a = self.param_groups[0]['a']
        t_k = self.param_groups[0]['t_k']
        s_k = self.param_groups[0]['s_k']
        k = self.param_groups[0]['k']
        c =self.param_groups[0]['lr']
        alpha = self.param_groups[0]['alpha']
        s_k1= c/((k+1)**alpha)    
        t_k1=t_k +s_k
        

        for group in self.param_groups:

            weight_decay = group['weight_decay']
            
            for p, v in zip(group['params'],group['v']):
                
                if p.grad is None:
                    continue
                d_p = p.grad.data
               
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                
                
                v.data = v.data + torch.mul(d_p,-1/2*(t_k*s_k+2*s_k*a))

                p.data = p.data + torch.mul(v.data, (2*s_k1/t_k1))
               
                p.data = p.data + torch.mul(d_p,(-s_k1*a))
                
                p.data = torch.mul(p.data,1/(1+2*s_k1/t_k1))

        
        self.param_groups[0]['t_k'] = t_k1
        self.param_groups[0]['s_k'] = s_k1
        self.param_groups[0]['k'] += 1
