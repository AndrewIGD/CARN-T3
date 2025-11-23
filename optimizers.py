import torch
import torch.optim as optim

def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, lr=1e-3, momentum=0.9, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)

        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                g = p.grad
                state = self.state[p]
                
                # Initialize momentum buffer
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                
                p.data.mul_(len(p.data)**0.5 / p.data.norm())
                
                flat_g = g.reshape(len(g), -1)
                update = zeropower_via_newtonschulz5(flat_g).view(g.shape)
                
                p.data.add_(update, alpha=-lr)

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

def get_optimizer(optimizer_name: str, model_parameters, lr: float = 0.001, **kwargs):
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        weight_decay = kwargs.get('weight_decay', 0.0)
        nesterov = kwargs.get('nesterov', False)
        return optim.SGD(
            model_parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
    
    elif optimizer_name == 'adam':
        betas = kwargs.get('betas', (0.9, 0.999))
        weight_decay = kwargs.get('weight_decay', 0.0)
        return optim.Adam(
            model_parameters,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )
    
    elif optimizer_name == 'adamw':
        betas = kwargs.get('betas', (0.9, 0.999))
        weight_decay = kwargs.get('weight_decay', 0.01)
        return optim.AdamW(
            model_parameters,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )
    
    elif optimizer_name == 'muon':

        model_parameters_list = list(model_parameters)

        muon_params = [p for p in model_parameters_list if p.ndim == 4]
        sgd_params = [p for p in model_parameters_list if p.ndim < 4]

        base_optimizer_str = kwargs.get('base_optimizer', 'sgd')
        base_optimizer = get_optimizer(base_optimizer_str, sgd_params, lr=lr, **kwargs)
        momentum = kwargs.get('momentum', 0.9)
        nesterov = kwargs.get('nesterov', False)
        return Muon(muon_params, base_optimizer, lr=lr, momentum=momentum, nesterov=nesterov)
    
    elif optimizer_name == 'sam':
        base_optimizer_str = kwargs.get('base_optimizer', 'sgd')
        base_optimizer = None
        if base_optimizer_str == 'sgd':
            base_optimizer = torch.optim.SGD
        elif base_optimizer_str == 'adam':
            base_optimizer = torch.optim.Adam
        elif base_optimizer_str == 'adamw':
            base_optimizer = torch.optim.AdamW
        elif base_optimizer_str == 'muon':
            base_optimizer = Muon

        rho = kwargs.get('rho', 0.05)
        adaptive = kwargs.get('adaptive', False)
        return SAM(model_parameters, base_optimizer, rho=rho, adaptive=adaptive)

    return None
