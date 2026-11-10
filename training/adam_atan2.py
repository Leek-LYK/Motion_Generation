"""Minimal Adam-atan2 with explicit bias correction and serializable FP32 moments.

Update follows Everett et al., arXiv:2407.05872v2, Appendix C.5:
  p <- p - lr * a * atan2(m_hat, b * sqrt(v_hat)), a=b=1 by default.
Bias correction is INSIDE atan2. No epsilon, parameter scaling, cautious update,
or other third-party optimizer extensions are added. M5 uses weight_decay=0.
"""
import math
import torch
from torch.optim import Optimizer

VERSION='m5-adam-atan2-bias-corrected-v1'


class AdamAtan2(Optimizer):
    def __init__(self,params,lr=2e-5,betas=(.9,.999),weight_decay=0.,a=1.,b=1.):
        if len(betas)!=2 or not all(math.isfinite(v) for v in [lr,weight_decay,a,b,*betas]) or lr<0 or weight_decay<0 or a<=0 or b<=0 or not all(0<=v<1 for v in betas):raise ValueError('Invalid Adam-atan2 configuration')
        super().__init__(params,dict(lr=lr,betas=tuple(betas),weight_decay=weight_decay,a=a,b=b))

    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        for group in self.param_groups:
            beta1,beta2=group['betas']
            for p in group['params']:
                if p.grad is None:continue
                if p.dtype not in (torch.float32,torch.float64):raise TypeError('Adam-atan2 parameters/moments must be FP32 or FP64; BF16 forward autocast is supported')
                g=p.grad
                if g.is_sparse:raise RuntimeError('Sparse Adam-atan2 gradients are unsupported')
                state=self.state[p]
                if not state:state.update(step=0,exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
                state['step']+=1;m=state['exp_avg'];v=state['exp_avg_sq'];t=state['step']
                m.lerp_(g,1-beta1);v.mul_(beta2).addcmul_(g,g,value=1-beta2)
                numerator=m/(1-beta1**t);denominator=(v/(1-beta2**t)).sqrt().mul_(group['b'])
                update=torch.atan2(numerator,denominator).mul_(group['a'])
                if group['weight_decay']:p.mul_(1-group['lr']*group['weight_decay'])
                p.add_(update,alpha=-group['lr'])
        return loss
