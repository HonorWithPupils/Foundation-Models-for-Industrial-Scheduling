import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from typing import Tuple, Union, Dict, NoReturn

from src.models.attention import SelfAttentions


class Actor(nn.Module):
    def __init__(
        self,
        m: int,
        h: int,
        n_max: int = 100,
        deep: int = 6,
        dim: int = 1024,
        n_head: int = 16,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.m = m
        self.h = h
        self.n_max = n_max

        self.bone = SelfAttentions(deep, dim, n_head, mlp_ratio * dim, dropout)

        self.embTM = nn.Embedding(m, dim)
        self.embTN = nn.Embedding(n_max, dim)

        self.linearTM = nn.Linear(1, dim)
        self.linearTN = nn.Linear(1, dim)

        self.linearOT = nn.Linear(m, dim)
        self.linearOM = nn.Linear(m, dim)
        self.embOorder = nn.Embedding(self.h, dim)
        self.embOjob = nn.Embedding(n_max, dim)
        self.embOfinished = nn.Embedding(2, dim)

        self.output = nn.Linear(dim, m)

        nn.init.constant_(self.output.weight, 0)
        nn.init.constant_(self.output.bias, 0)

    def forward(
        self,
        Tm: Tensor,
        Tn: Tensor,
        om: Tensor,
        ot: Tensor,
        ooder: Tensor,
        ojob: Tensor,
        ofinished: Tensor,
        valid: Tensor,
    ) -> Tensor:

        b, n = Tn.shape
        b, l = ooder.shape

        reindx = torch.stack([torch.randperm(self.n_max)[:n] for _ in range(b)])
        reindx = reindx.to(Tm.device)
        # (b, n)

        ojob = (F.one_hot(ojob, n) * reindx.reshape(b, 1, -1)).sum(-1)

        Tm = self.linearTM(Tm.reshape(b, self.m, 1)) + self.embTM(
            torch.arange(self.m, device=Tm.device).reshape(1, self.m)
        )

        Tn = self.linearTN(Tn.reshape(b, n, 1)) + self.embTN(reindx)

        oT = self.linearOT(ot)
        oM = self.linearOM(om.to(oT.dtype))
        oOrder = self.embOorder(ooder)
        oJob = self.embOjob(ojob)
        oF = self.embOfinished(ofinished.to(ojob.dtype))
        O = oT + oM + oOrder + oJob + oF

        mask = (
            torch.cat([torch.zeros(b, self.m + n, device=Tm.device), ofinished], dim=-1)
            == 1
        )

        x = torch.cat([Tm, Tn, O], dim=1)
        x = self.bone(x, mask)

        x = self.output(x[:, self.m + n :])

        # valid mask
        negInf = torch.finfo(x.dtype).min
        valid = valid.to(torch.bool)
        x = x * valid + negInf * ~valid

        x = F.softmax(x.reshape(b, l * self.m), dim=-1)

        return x


class Critic(nn.Module):
    def __init__(
        self,
        m: int,
        h: int,
        n_max: int = 100,
        deep: int = 6,
        dim: int = 1024,
        n_head: int = 16,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.m = m
        self.h = h
        self.n_max = n_max

        self.bone = SelfAttentions(deep, dim, n_head, mlp_ratio * dim, dropout)

        self.embTM = nn.Embedding(m, dim)
        self.embTN = nn.Embedding(n_max, dim)

        self.linearTM = nn.Linear(1, dim)
        self.linearTN = nn.Linear(1, dim)

        self.linearOT = nn.Linear(m, dim)
        self.linearOM = nn.Linear(m, dim)
        self.embOorder = nn.Embedding(self.h, dim)
        self.embOjob = nn.Embedding(n_max, dim)
        self.embOfinished = nn.Embedding(2, dim)

        self.output = nn.Linear(dim, 1)

    def forward(
        self,
        Tm: Tensor,
        Tn: Tensor,
        om: Tensor,
        ot: Tensor,
        ooder: Tensor,
        ojob: Tensor,
        ofinished: Tensor,
        valid: Tensor,
    ) -> Tensor:

        b, n = Tn.shape
        b, l = ooder.shape

        reindx = torch.stack([torch.randperm(self.n_max)[:n] for _ in range(b)])
        reindx = reindx.to(Tm.device)
        # (b, n)

        ojob = (F.one_hot(ojob, n) * reindx.reshape(b, 1, -1)).sum(-1)

        Tm = self.linearTM(Tm.reshape(b, self.m, 1)) + self.embTM(
            torch.arange(self.m, device=Tm.device).reshape(1, self.m)
        )

        Tn = self.linearTN(Tn.reshape(b, n, 1)) + self.embTN(reindx)

        oT = self.linearOT(ot)
        oM = self.linearOM(om.to(oT.dtype))
        oOrder = self.embOorder(ooder)
        oJob = self.embOjob(ojob)
        oF = self.embOfinished(ofinished.to(ojob.dtype))
        O = oT + oM + oOrder + oJob + oF

        mask = (
            torch.cat([torch.zeros(b, self.m + n, device=Tm.device), ofinished], dim=-1)
            == 1
        )

        x = torch.cat([Tm, Tn, O], dim=1)
        x = self.bone(x, mask)

        x = self.output(
            (x * (mask == 0).reshape(b, l + self.m + n, 1)).sum(-2)
        ).reshape(b)

        # a small trick: if all operation are finished, V(st) = 0
        x = ~ofinished.all(-1) * x
        return x
