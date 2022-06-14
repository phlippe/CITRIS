"""
Summarizing all normalizing flow layers
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import scipy.linalg

import sys
sys.path.append('../../')
from models.shared import AutoregLinear


class AutoregNormalizingFlow(nn.Module):
    """ 
    Base class for the autoregressive normalizing flow 
    We use a combination of affine autoregressive coupling layers,
    activation normalization, and invertible 1x1 convolutions / 
    orthogonal invertible transformations.
    """

    def __init__(self, num_vars, num_flows, act_fn, hidden_per_var=16, zero_init=False, use_scaling=True, use_1x1_convs=True, init_std_factor=0.2):
        super().__init__()
        self.flows = nn.ModuleList([])
        transform_layer = lambda num_vars: AffineFlow(num_vars, use_scaling=use_scaling)
        for i in range(num_flows):
            self.flows.append(ActNormFlow(num_vars))
            if i > 0:
                if use_1x1_convs:
                    self.flows.append(OrthogonalFlow(num_vars))
                else:
                    self.flows.append(ReverseSeqFlow())
            self.flows.append(AutoregressiveFlow(num_vars, 
                                                 hidden_per_var=hidden_per_var, 
                                                 act_fn=act_fn, 
                                                 init_std_factor=(0 if zero_init else init_std_factor),
                                                 transform_layer=transform_layer))

    def forward(self, x):
        ldj = x.new_zeros(x.shape[0],)
        for flow in self.flows:
            x, ldj = flow(x, ldj)
        return x, ldj

    def reverse(self, x):
        for flow in reversed(self.flows):
            x = flow.reverse(x)
        return x


class AffineFlow(nn.Module):
    """ Affine transformation """

    def __init__(self, num_vars, use_scaling=True, hard_limit=-1):
        super().__init__()
        self.num_vars = num_vars
        self.hard_limit = hard_limit
        self.use_scaling = use_scaling
        if self.use_scaling and self.hard_limit <= 0:
            self.scaling = nn.Parameter(torch.zeros(num_vars,))

    def get_num_outputs(self):
        return 2

    def _get_affine_params(self, out):
        if isinstance(out, (list, tuple)):
            t, s = out
        else:
            t, s = out.unflatten(-1, (-1, 2)).unbind(dim=-1)
        if self.use_scaling:
            if self.hard_limit > 0:
                s = s - torch.max(s - self.hard_limit, torch.zeros_like(s)).detach()
                s = s + torch.max(-self.hard_limit - s, torch.zeros_like(s)).detach()
            else:
                sc = torch.tanh(self.scaling.exp()[None] / 3.0) * 3.0
                s = torch.tanh(s / sc.clamp(min=1.0)) * sc
        else:
            s = s * 0.0
        return t, s

    def forward(self, x, out, ldj):
        t, s = self._get_affine_params(out)
        x = (x + t) * s.exp()
        ldj = ldj - s.sum(dim=1)
        return x, ldj

    def reverse(self, x, out):
        t, s = self._get_affine_params(out)
        x = x * (-s).exp() - t
        return x


class AutoregressiveFlow(nn.Module):
    """ Autoregressive flow with arbitrary invertible transformation """

    def __init__(self, num_vars, hidden_per_var=16, 
                       act_fn=lambda: nn.SiLU(),
                       init_std_factor=0.2,
                       transform_layer=AffineFlow):
        super().__init__()
        self.transformation = transform_layer(num_vars)
        self.net = nn.Sequential(
                AutoregLinear(num_vars, 1, hidden_per_var, diagonal=False),
                act_fn(),
                AutoregLinear(num_vars, hidden_per_var, hidden_per_var, diagonal=True),
                act_fn(),
                AutoregLinear(num_vars, hidden_per_var, self.transformation.get_num_outputs(), diagonal=True,
                              no_act_fn_init=True, 
                              init_std_factor=init_std_factor, 
                              init_bias_factor=0.0,
                              init_first_block_zeros=True)
            )

    def forward(self, x, ldj):
        out = self.net(x)
        x, ldj = self.transformation(x, out, ldj)
        return x, ldj

    def reverse(self, x):
        inp = x * 0.0
        for i in range(x.shape[1]):
            out = self.net(inp)
            x_new = self.transformation.reverse(x, out)
            inp[:,i] = x_new[:,i]
        return x_new


class ActNormFlow(nn.Module):
    """ Activation normalization """

    def __init__(self, num_vars):
        super().__init__()
        self.num_vars = num_vars 
        self.data_init = False

        self.bias = nn.Parameter(torch.zeros(self.num_vars,))
        self.scales = nn.Parameter(torch.zeros(self.num_vars,))
        self.affine_flow = AffineFlow(self.num_vars, hard_limit=3.0)

    def get_num_outputs(self):
        return 2

    def forward(self, x, ldj):
        if self.training and not self.data_init:
            self.data_init_forward(x)
        x, ldj = self.affine_flow(x, [self.bias[None], self.scales[None]], ldj)
        return x, ldj

    def reverse(self, x):
        x = self.affine_flow.reverse(x, [self.bias[None], self.scales[None]])
        return x

    @torch.no_grad()
    def data_init_forward(self, input_data):
        if (self.bias != 0.0).any():
            self.data_init = True
            return 

        batch_size = input_data.shape[0]

        self.bias.data = -input_data.mean(dim=0)
        self.scales.data = -input_data.std(dim=0).log()
        self.data_init = True

        out, _ = self.forward(input_data, input_data.new_zeros(batch_size,))
        print(f"[INFO - ActNorm] New mean: {out.mean().item():4.2f}")
        print(f"[INFO - ActNorm] New variance {out.std(dim=0).mean().item():4.2f}")


class ReverseSeqFlow(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, ldj):
        return torch.flip(x, dims=(-1,)), ldj

    def reverse(self, x):
        return self.forward(x, None)[0]


class OrthogonalFlow(nn.Module):
    """ Invertible 1x1 convolution / orthogonal flow """

    def __init__(self, num_vars, LU_decomposed=True):
        super().__init__()
        self.num_vars = num_vars
        self.LU_decomposed = LU_decomposed

        # Initialize with a random orthogonal matrix
        w_init = np.random.randn(self.num_vars, self.num_vars)
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)

        if not self.LU_decomposed:
            self.weight = nn.Parameter(torch.from_numpy(w_init), requires_grad=True)
        else: 
            # LU decomposition can slightly speed up the inverse
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_init.shape, dtype=np.float32), -1)
            eye = np.eye(*w_init.shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)), requires_grad=True)
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)), requires_grad=True)
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)), requires_grad=True)
            self.register_buffer('l_mask', torch.Tensor(l_mask))
            self.register_buffer('eye', torch.Tensor(eye))

        self.eval_dict = defaultdict(lambda : self._get_default_inner_dict())

    def _get_default_inner_dict(self):
        return {"weight": None, "inv_weight": None, "sldj": None}

    def _get_weight(self, device_name, inverse=False):
        if self.training or self._is_eval_dict_empty(device_name):
            if not self.LU_decomposed:
                weight = self.weight
                sldj = torch.slogdet(weight)[1]
            else:
                l, log_s, u = self.l, self.log_s, self.u
                l = l * self.l_mask + self.eye
                u = u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(log_s))
                weight = torch.matmul(self.p, torch.matmul(l, u))
                sldj = log_s.sum()
        
        if not self.training:
            if self._is_eval_dict_empty(device_name):
                self.eval_dict[device_name]["weight"] = weight.detach()
                self.eval_dict[device_name]["sldj"] = sldj.detach()
                self.eval_dict[device_name]["inv_weight"] = torch.inverse(weight.double()).float().detach()
            else:
                weight, sldj = self.eval_dict[device_name]["weight"], self.eval_dict[device_name]["sldj"]
        elif not self._is_eval_dict_empty(device_name):
            self._empty_eval_dict(device_name)
        
        if inverse:
            if self.training:
                weight = torch.inverse(weight.double()).float()
            else:
                weight = self.eval_dict[device_name]["inv_weight"]
        
        return weight, sldj

    def _is_eval_dict_empty(self, device_name=None):
        if device_name is not None:
            return not (device_name in self.eval_dict)
        else:
            return len(self.eval_dict) == 0

    def _empty_eval_dict(self, device_name=None):
        if device_name is not None:
            self.eval_dict.pop(device_name)
        else:
            self.eval_dict = defaultdict(lambda : self._get_default_inner_dict())
        
    def forward(self, x, ldj):
        weight, sldj = self._get_weight(device_name=str(x.device), inverse=False)
        ldj = ldj - sldj
        z = torch.matmul(x, weight)
        return z, ldj

    def reverse(self, x):
        weight, sldj = self._get_weight(device_name=str(x.device), inverse=True)
        z = torch.matmul(x, weight)
        return z