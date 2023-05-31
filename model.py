from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Independent

from utils import float2bit, bit2float


class DBM(nn.Module):
    def __init__(self, size, nc, nh=None, bits=8, L=2):
        super().__init__()
        total_bit = nc * bits

        nv = nc * bits * size**2
        if nh is None:
            nh = nv
        self.weight = nn.ParameterList([nn.Parameter(torch.Tensor(nh, nv))])
        self.weight.extend([nn.Parameter(torch.Tensor(nh, nh)) for _ in range(L-1)])
        self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(nv))])
        self.bias.extend([nn.Parameter(torch.Tensor(nh)) for _ in range(L)])

        self.nv = nv
        self.nh = nh
        self.size = size
        self.nc = nc
        self.bits = bits
        self.L = L

        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weight:
            nn.init.orthogonal_(w)

        for b in self.bias:
            nn.init.zeros_(b)

    def forward(self, v):
        N = v.size(0)
        device = v.device

        # Positive phase
        v = float2bit(v, self.bits).flatten(1).float()

        if self.L == 1:
            if self.marginal:
                energy_pos = self.marginal_energy(v, None, True)
            else:
                v, h = self.gibbs_step(v, None, True,
                                       torch.ones(N, device=device))
                energy_pos = self.energy(v, h)
        else:
            h = []
            for _ in range(self.L):
                h_i = torch.empty(N, self.nh, device=device).bernoulli_()
                h.append(h_i)

            v, h = self.local_search(v, h, True)
            v, h = self.gibbs_step(v, h, True)

            energy_pos, v, h = self.coupling(v, h, True)

        # Negative phase
        v = torch.empty_like(v).bernoulli_()

        h = []
        for _ in range(self.L):
            h_i = torch.empty(N, self.nh, device=device).bernoulli_()
            h.append(h_i)

        v, h = self.local_search(v, h)
        v, h = self.gibbs_step(v, h)

        energy_neg, v, h = self.coupling(v, h)

        loss = energy_pos - energy_neg

        return loss

    @torch.no_grad()
    def local_search(self, v, h, fix_v=False):
        N = v.size(0)
        device= v.device

        rand_u = torch.rand(N, device=device)
        _v, _h = deepcopy((v, h))
        v, h = self.gibbs_step(v, h, fix_v, rand_u=rand_u, T=0)

        converged = torch.ones(N, dtype=torch.bool, device=device) if fix_v \
                    else torch.all(v == _v, 1)
        for i in range(self.L):
            converged = converged.logical_and(torch.all(h[i] == _h[i], 1))

        while not converged.all():
            not_converged = converged.logical_not()
            _v = v[not_converged]
            _h = [h[i][not_converged] for i in range(self.L)]
            M = _v.size(0)

            v_, h_ = self.gibbs_step(_v, _h, fix_v,
                                     rand_u=rand_u[not_converged], T=0)

            if fix_v:
                converged_ = torch.ones(M, dtype=torch.bool, device=device)
            else:
                converged_ = torch.all(v_ == _v, 1)
                v[not_converged] = v_

            for i in range(self.L):
                converged_ = converged_.logical_and(torch.all(h_[i] == _h[i], 1))
                h[i][not_converged] = h_[i]

            converged[not_converged] = converged_

        return v, h

    def coupling(self, v, h, fix_v=False):
        N = v.size(0)
        device = v.device
        _v, _h = deepcopy((v, h))

        v, h = self.mh_step(v, h, fix_v)
        energy = self.energy(v, h)

        converged = torch.ones(N, dtype=torch.bool, device=device) if fix_v \
                    else torch.all(v == _v, 1)
        for i in range(self.L):
            converged = converged.logical_and(torch.all(h[i] == _h[i], 1))

        while not converged.all():
            not_converged = converged.logical_not()
            _v = v[not_converged]
            _h = [h[i][not_converged] for i in range(self.L)]
            M = _v.size(0)

            rand_v = None if fix_v else torch.rand_like(_v)
            rand_h = [torch.rand_like(_h[i]) for i in range(self.L)]
            rand_u = torch.rand(M, device=device)

            v_, h_ = self.mh_step(_v, _h, fix_v, rand_v, rand_h, rand_u)
            energy[not_converged] += self.energy(v_, h_) - self.energy(_v, _h)

            if fix_v:
                converged_ = torch.ones(M, dtype=torch.bool, device=device)
            else:
                converged_ = torch.all(v_ == _v, 1)
                v[not_converged] = v_

            for i in range(self.L):
                converged_ = converged_.logical_and(torch.all(h_[i] == _h[i], 1))
                h[i][not_converged] = h_[i]

            converged[not_converged] = converged_

        return energy, v, h

    def energy(self, v, h):
        energy = - torch.sum(v * self.bias[0].unsqueeze(0), 1)

        for i in range(self.L):
            logits = F.linear(v if i==0 else h[i-1],
                              self.weight[i], self.bias[i+1])

            energy -= torch.sum(h[i] * logits, 1)

        return energy

    @torch.no_grad()
    def gibbs_step(self, v, h, fix_v=False,
                   rand_v=None, rand_h=None, rand_u=None, rand_z=None, T=1):
        N = v.size(0)
        device = v.device

        v_, h_ = deepcopy((v, h))

        if rand_u is None:
            rand_u = torch.rand(N, device=device)

        even = rand_u < 0.5
        odd = even.logical_not()

        if even.sum() > 0:
            if not fix_v:
                logits = F.linear(h_[0][even],
                                  self.weight[0].t(), self.bias[0])

                if T == 0:
                    v_[even] = (logits >= 0).float()
                else:
                    logits /= T

                    if rand_v is None:
                        v_[even] = Independent(Bernoulli(logits=logits), 1).sample()
                    else:
                        v_[even] = (rand_v[even] < logits.sigmoid()).float()

            for i in range(1, len(h), 2):
                logits = F.linear(h_[i-1][even],
                                  self.weight[i], self.bias[i+1])
                if i+1 < len(h):
                    logits += F.linear(h_[i+1][even],
                                       self.weight[i+1].t(), None)

                if T == 0:
                    h_[i][even] = (logits >= 0).float()
                else:
                    logits /= T

                    if rand_h is None:
                        h_[i][even] = Independent(Bernoulli(logits=logits), 1).sample()
                    else:
                        h_[i][even] = (rand_h[i][even] < logits.sigmoid()).float()

            for i in range(0, len(h), 2):
                logits = F.linear(v_[even] if i==0 else h_[i-1][even],
                                  self.weight[i], self.bias[i+1])
                if i+1 < len(h):
                    logits += F.linear(h_[i+1][even],
                                       self.weight[i+1].t(), None)

                if T == 0:
                    h_[i][even] = (logits >= 0).float()
                else:
                    logits /= T

                    if rand_h is None:
                        h_[i][even] = Independent(Bernoulli(logits=logits), 1).sample()
                    else:
                        h_[i][even] = (rand_h[i][even] < logits.sigmoid()).float()

        if odd.sum() > 0:
            for i in range(0, len(h), 2):
                logits = F.linear(v_[odd] if i==0 else h_[i-1][odd],
                                  self.weight[i], self.bias[i+1])
                if i+1 < len(h):
                    logits += F.linear(h_[i+1][odd],
                                       self.weight[i+1].t(), None)

                if T == 0:
                    h_[i][odd] = (logits >= 0).float()
                else:
                    logits /= T

                    if rand_h is None:
                        h_[i][odd] = Independent(Bernoulli(logits=logits), 1).sample()
                    else:
                        h_[i][odd] = (rand_h[i][odd] < logits.sigmoid()).float()

            if not fix_v:
                logits = F.linear(h_[0][odd],
                                  self.weight[0].t(), self.bias[0])

                if T == 0:
                    v_[odd] = (logits >= 0).float()
                else:
                    logits /= T

                    if rand_v is None:
                        v_[odd] = Independent(Bernoulli(logits=logits), 1).sample()
                    else:
                        v_[odd] = (rand_v[odd] < logits.sigmoid()).float()

            for i in range(1, len(h), 2):
                logits = F.linear(h_[i-1][odd],
                                  self.weight[i], self.bias[i+1])
                if i+1 < len(h):
                    logits += F.linear(h_[i+1][odd],
                                       self.weight[i+1].t(), None)

                if T == 0:
                    h_[i][odd] = (logits >= 0).float()
                else:
                    logits /= T

                    if rand_h is None:
                        h_[i][odd] = Independent(Bernoulli(logits=logits), 1).sample()
                    else:
                        h_[i][odd] = (rand_h[i][odd] < logits.sigmoid()).float()

        return v_, h_

    @torch.no_grad()
    def mh_step(self, v, h, fix_v=False,
                rand_v=None, rand_h=None, rand_u=None):
        N = v.size(0)
        device = v.device

        if fix_v:
            v_ = v
        else:
            if rand_v is None:
                v_ = torch.empty_like(v).bernoulli_()
            else:
                v_ = (rand_v < 0.5).float()

        if rand_h is None:
            h_ = [torch.empty_like(h[i]).bernoulli_() for i in range(self.L)]
        else:
            h_ = [(rand_h[i] < 0.5).float() for i in range(self.L)]

        log_ratio = self.energy(v, h) - self.energy(v_, h_)

        if rand_u is None:
            accepted = log_ratio.exp().clamp(0, 1).bernoulli().bool()
        else:
            accepted = rand_u < log_ratio.exp()

        if not fix_v:
            v = torch.where(accepted.unsqueeze(1), v_, v)
        h = [torch.where(accepted.unsqueeze(1), h_[i], h[i]) for i in range(self.L)]

        return v, h

    @torch.no_grad()
    def sample(self, N):
        device = next(self.parameters()).device

        v = torch.empty(N, self.nv, device=device).bernoulli_()
        h = [torch.empty(N, self.nh,
                         device=device).bernoulli_() for _ in range(self.L)]

        v_mode, h_mode = self.local_search(v, h)
        v_rand, h_rand = self.gibbs_step(v_mode, h_mode)

        v_mode = v_mode.unflatten(1, (self.nc, self.size, self.size, self.bits))
        v_mode = bit2float(v_mode.bool(), self.bits)

        v_rand = v_rand.unflatten(1, (self.nc, self.size, self.size, self.bits))
        v_rand = bit2float(v_rand.bool(), self.bits)

        return v_mode, v_rand

    @torch.no_grad()
    def reconstruct(self, v):
        N = v.size(0)
        device = v.device

        v = float2bit(v, self.bits).flatten(1).float()
        h = [torch.empty(N, self.nh,
                         device=device).bernoulli_() for _ in range(self.L)]

        v, h = self.local_search(v, h, True)
        v_mode, h_mode = self.gibbs_step(v, h, T=0)
        v_rand, h_rand = self.gibbs_step(v, h)

        v_mode = v_mode.unflatten(1, (self.nc, self.size, self.size, self.bits))
        v_mode = bit2float(v_mode.bool(), self.bits)

        v_rand = v_rand.unflatten(1, (self.nc, self.size, self.size, self.bits))
        v_rand = bit2float(v_rand.bool(), self.bits)

        return v_mode, v_rand
