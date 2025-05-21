import torch
from torch.nn import functional as F

from src.FJSP import Envs

from typing import Union


def _reapeat_o(o, repeat):

    return {k: o[k].repeat_interleave(repeat, dim=0) for k in o.keys()}


def _unique(state):
    C = torch.cat((state[0], state[1], state[-2]), dim=1).to("cpu")
    uniques = []
    if_unique = []
    for c in C:
        flag = True
        for u in uniques:
            if (c == u).all():
                flag = False
                break
        if_unique.append(flag == False)
        if flag:
            uniques.append(c)
    if_unique = torch.tensor(if_unique)

    return if_unique


@torch.no_grad()
def BeamSearch(
    FJSPinfo, actor, critic, num_beams=32, device: Union[str, torch.device] = "cpu"
):
    """Beam Search"""

    FJSPinfo = FJSPinfo.copy()

    n = FJSPinfo["n"]
    m = FJSPinfo["m"]

    batch = 2 * num_beams
    FJSPinfo["o"] = _reapeat_o(FJSPinfo["o"], batch)

    envs = Envs(FJSPinfo, device=device)
    state = envs.reset()

    r = []
    actions = []

    for i in range(envs.l):

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            P = actor(*state)

        if i == 0:

            valid = state[-1][0].reshape(-1)
            if valid.sum() > n:
                _, indx = torch.topk(P[0], n, dim=-1)
                indx = indx.repeat(batch // n + 1)[:batch]
            else:
                _, indx = torch.topk(P[0], valid.sum(), dim=-1)
                indx = indx.repeat(batch // valid.sum() + 1)[:batch]

            action = indx

        else:

            _, indx = torch.topk(P, 2, dim=-1)

            unique = _unique(state).to(device).long()

            action = indx[torch.arange(indx.size(0)), unique]

        actions.append(action)

        action = F.one_hot(action, num_classes=envs.l * m)
        state, reward, done = envs.step(action)
        r.append(reward)

    r = torch.stack(r, dim=-1)
    actions = torch.stack(actions, dim=-1)

    makespan = -r.sum(-1)
    indx = makespan.argmin()

    return makespan[indx].item(), actions[indx]
