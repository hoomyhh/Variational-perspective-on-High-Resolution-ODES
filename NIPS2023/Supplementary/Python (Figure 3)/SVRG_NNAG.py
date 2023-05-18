from torch.optim import Optimizer
import torch
import copy


class SVRG_NNAG(Optimizer):
    r"""Optimization class for calculating the gradient of one iteration.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        v (iterable): deep copy of the model parameters (auxilary variable for optimization)
        lr (float): learning rate
        t_k (float): accumulatin step size
        s_k (float): decreasing step size
        a (float): tuning parameter usually set to \sqrt{L/10}
        k (integer): iteration counter for calculating step sizes
        alpha (float): tuning parameter set to 3/4
        
        
    """
    def __init__(self, params, v , lr, t_k,s_k ,a,weight_decay=0,k=0,alpha=3/4):
        print("Using optimizer: NNAG+SVRG")
        self.u = None

        defaults = dict(v=v,lr=lr,t_k=lr,s_k=lr,a=a,weight_decay=weight_decay, k=k,alpha=alpha)
        super(SVRG_NNAG, self).__init__(params, defaults)
    
    def get_param_groups(self):
            return self.param_groups

        
    def opt__init(self):
        """ Used for resetting the t_k and k at the beginning of each epoch
        """
        self.param_groups[0]['k'] =0
        self.param_groups[0]['t_k']=0

    def set_u(self, new_u):
        """Set the mean gradient for the current epoch. 
        """
        if self.u is None:
            self.u = copy.deepcopy(new_u)
        for u_group, new_group in zip(self.u, new_u):  
            for u, new_u in zip(u_group['params'], new_group['params']):
                u.grad = new_u.grad.clone()

    def step(self, params):
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
        
        for group, new_group, u_group  in zip(self.param_groups, params, self.u):
            weight_decay = group['weight_decay']

            for p, q, u , v in zip(group['params'], new_group['params'], u_group['params'], group['v']):
                if p.grad is None:
                    continue
                if q.grad is None:
                    continue
                # core SVRG_NNAG gradient update 
                new_d = p.grad.data - q.grad.data + u.grad.data
                if weight_decay != 0:
                    new_d.add_(weight_decay, p.data)

                v.data = v.data + torch.mul(new_d,-1/2*(t_k*s_k+2*s_k*a))

                p.data = p.data + torch.mul(v.data, (2*s_k1/t_k1))
               
                p.data = p.data + torch.mul(new_d,(-s_k1*a))
                
                p.data = torch.mul(p.data,1/(1+2*s_k1/t_k1))
                
        self.param_groups[0]['t_k'] = t_k1
        self.param_groups[0]['s_k'] = s_k1
        self.param_groups[0]['k'] += 1

class SVRG_Snapshot(Optimizer):
    r"""Optimization class for calculating the mean gradient (snapshot) of all samples.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """
    def __init__(self, params):
        defaults = dict()
        super(SVRG_Snapshot, self).__init__(params, defaults)
      
    def get_param_groups(self):
            return self.param_groups
    
    def set_param_groups(self, new_params):
        """Copies the parameters from another optimizer. 
        """
        for group, new_group in zip(self.param_groups, new_params): 
            for p, q in zip(group['params'], new_group['params']):
                  p.data[:] = q.data[:]
