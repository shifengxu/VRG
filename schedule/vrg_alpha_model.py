import torch
from torch import nn
from utils import log_info

class VrgAlphaModel(nn.Module):
    def __init__(self, alpha_list=None, alpha_bar_list=None, learning_portion=0.01, log_fn=log_info):
        super().__init__()
        if alpha_list is not None:
            a_base = alpha_bar_list
            a_base = torch.tensor(a_base)
        elif alpha_bar_list is not None:
            a_bar = torch.tensor(alpha_bar_list)
            a_tmp = a_bar[1:] / a_bar[:-1]
            a_base = torch.cat([a_bar[0:1], a_tmp], dim=0)
        else:
            raise ValueError(f"Both alpha_list and alpha_bar_list are None")
        a_min = torch.min(a_base)
        a_max = torch.max(a_base)
        assert a_min > 0., f"all alpha_list must be > 0.: a_min: {a_min}"
        assert a_max < 1., f"all alpha_list must be < 1.: a_max: {a_max}"
        self.out_channels = len(a_base)
        self.learning_portion = learning_portion
        # make sure learning-portion is small enough. Then new alpha_list won't exceed range of [0, 1]
        _lp = torch.mul(torch.ones_like(a_base, dtype=torch.float64), learning_portion)
        _lp = torch.minimum(1-a_base, _lp)
        _lp = torch.minimum(a_base, _lp)
        _lp = torch.nn.Parameter(_lp, requires_grad=False)
        self._lp = _lp
        self.log_fn = log_fn
        # hard code the alpha_list base, which is from DPM-Solver
        # a_base = [0.370370, 0.392727, 0.414157, 0.434840, 0.457460,   # by original TS: 49, 99, 149,,,
        #           0.481188, 0.506092, 0.532228, 0.559663, 0.588520,
        #           0.618815, 0.650649, 0.684075, 0.719189, 0.756066,
        #           0.794792, 0.835464, 0.878171, 0.923015, 0.970102, ]
        # ab.reverse()
        #
        # by geometric with ratio 1.07
        # a_base = [0.991657, 0.978209, 0.961940, 0.942770, 0.920657,
        #           0.895610, 0.867686, 0.828529, 0.797675, 0.750600,
        #           0.704142, 0.654832, 0.597398, 0.537781, 0.477242,
        #           0.417018, 0.353107, 0.292615, 0.236593, 0.177778, ]
        self.alpha_base = torch.nn.Parameter(a_base, requires_grad=False)
        self.linear1 = torch.nn.Linear(1000,  2000, dtype=torch.float64)
        self.linear2 = torch.nn.Linear(2000,  2000, dtype=torch.float64)
        self.linear3 = torch.nn.Linear(2000,  self.out_channels, dtype=torch.float64)

        # the seed. we choose value 0.5. And it is better than value 1.0
        ones_k = torch.mul(torch.ones((1000,), dtype=torch.float64), 0.5)
        self.seed_k = torch.nn.Parameter(ones_k, requires_grad=False)
        f2s = lambda arr: ' '.join([f"{f:.6f}" for f in arr])
        log_fn(f"VrgAlphaModel::__init__()...")
        log_fn(f"  out_channels     : {self.out_channels}")
        log_fn(f"  learning_portion : {self.learning_portion}")
        log_fn(f"  _lp length       : {len(self._lp)}")
        log_fn(f"  _lp[:5]          : [{f2s(self._lp[:5])}]")
        log_fn(f"  _lp[-5:]         : [{f2s(self._lp[-5:])}]")
        log_fn(f"  alpha_base       : {len(self.alpha_base)}")
        log_fn(f"  alpha_base[:5]   : [{f2s(self.alpha_base[:5])}]")
        log_fn(f"  alpha_base[-5:]  : [{f2s(self.alpha_base[-5:])}]")
        log_fn(f"VrgAlphaModel::__init__()...Done")

    def gradient_clip(self):
        if self.linear1.weight.grad is not None:
            self.linear1.weight.grad = torch.tanh(self.linear1.weight.grad)
            self.linear1.weight.grad = torch.tanh(self.linear1.weight.grad)
        if self.linear2.weight.grad is not None:
            self.linear2.weight.grad = torch.tanh(self.linear2.weight.grad)
            self.linear2.weight.grad = torch.tanh(self.linear2.weight.grad)
        if self.linear3.weight.grad is not None:
            self.linear3.weight.grad = torch.tanh(self.linear3.weight.grad)
            self.linear3.weight.grad = torch.tanh(self.linear3.weight.grad)

    def forward(self):
        output = self.linear1(self.seed_k)
        output = self.linear2(output)
        output = self.linear3(output)
        output = torch.tanh(output)
        alpha_list = torch.add(self.alpha_base, output * self._lp)
        alphabar_list = torch.cumprod(alpha_list, dim=0)

        return alpha_list, alphabar_list
