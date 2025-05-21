import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from typing import Tuple, Union, Dict, NoReturn


class Envs:
    def __init__(
        self,
        FJSPinfo: Dict,
        device: Union[str, torch.device] = "cpu",
    ) -> NoReturn:
        """Create FJSP envs

        Args:
            FJSPinfo (Dict): the information of FJSP. More than one FJSP can be included. The format is as follows:
                {
                    "n": number of jobs, int;
                    "m": number of machines, int;
                    "o": {
                        "job": job index of each job on each machine, int, Tensor(B, number of operations);
                        "order": order of each job on each machine, int, Tensor(B, number of operations);
                        "machine": feasibility of each job on each machine, bool, Tensor(B, number of operations, number of machines);
                        "time": time of each job on each machine, float[0,1], Tensor(B, number of operations, number of machines);
                        "finished": whether each job has been finished, bool, Tensor(B, number of operations), default to zeros;
                    }
                }
            device (Union[str, torch.device], optional): device for the deployment environment. Defaults to "cpu".
        """
        super().__init__()

        # load FJSPinfo
        FJSPinfo = FJSPinfo.copy()

        self.n = FJSPinfo["n"]  # number of jobs
        self.m = FJSPinfo["m"]  # number of machines

        o = FJSPinfo["o"]
        self.o = {k: v.to(device) for k, v in o.items()}
        self.l = len(o["time"][0])  # number of operations

        self.batch = len(self.o["time"])

        self.device = device

        # initialize Tn, Tm, start
        self.Tn = torch.zeros((self.batch, self.n), dtype=torch.float32, device=device)
        self.Tm = torch.zeros((self.batch, self.m), dtype=torch.float32, device=device)
        self.start = torch.zeros((self.batch), dtype=torch.float32, device=device)

        # If no finished in FJSPinfo["o"], initialize it with zeros
        if "finished" not in self.o.keys():
            self.o["finished"] = torch.zeros(
                (self.batch, self.l), dtype=torch.bool, device=device
            )

        self.initState = self.o["finished"].clone()

    @torch.no_grad()
    def validActions(self) -> Tensor:
        """Get the valid actions for each job on each machine

        Returns:
            Tensor: valid actions for each job on each machine, bool, (B, number of operations, number of machines)
                    false for invalid actions, true for valid actions
        """

        o = self.o

        # Initialize valid to all zeros
        valid = torch.zeros((self.batch, self.l), dtype=torch.bool, device=self.device)

        # For each job, find the next operation that can be executed
        for i in range(self.n):

            tmp = (o["job"] != i) * 1e8 + o["finished"] * 1e8 + o["order"]

            next_order = torch.min(tmp, dim=1, keepdim=True)[0]  # batch, l

            valid += (tmp == next_order) & (next_order < 1e8)

        valid = valid.unsqueeze(-1) * o["machine"]

        return valid

    @torch.no_grad()
    def randomActions(self) -> Tensor:
        """Get random actions for each job on each machine

        Returns:
            Tensor: random actions for each job on each machine, bool, (B, number of operations, number of machines)
        """

        valid = self.validActions().flatten(1, 2)

        action = torch.multinomial(valid.float() + 1e-8, 1)
        action = F.one_hot(action, num_classes=self.l * self.m)

        return action.reshape(self.batch, self.l * self.m)  # batch, l, m

    @torch.no_grad()
    def reset(self) -> Tuple:
        """Reset the environment

        Returns:
            State: the state of the environment
                {
                    "Tm": the time of each machine, float, Tensor(B, number of machines);
                    "Tn": the time of each job, float, Tensor(B, number of jobs);
                    "machine": feasibility of each job on each machine, bool, Tensor(B, number of operations, number of machines);
                    "time": time of each job on each machine, float[0,1], Tensor(B, number of operations, number of machines);
                    "order": order of each job on each machine, int, Tensor(B, number of operations);
                    "job": job index of each job on each machine, int, Tensor(B, number of operations);
                    "finished": whether each job has been finished, bool, Tensor(B, number of operations);
                    "valid": valid actions for each job on each machine, bool, Tensor(B, number of operations, number of machines);
                }
        """

        self.o["finished"] = self.initState.clone()

        b, n, m, device = self.batch, self.n, self.m, self.device
        self.Tn = torch.zeros((b, n), dtype=torch.float32, device=device)
        self.Tm = torch.zeros((b, m), dtype=torch.float32, device=device)
        self.start = torch.zeros((b,), dtype=torch.float32, device=device)

        return (
            self.Tm.clone().detach(),
            self.Tn.clone().detach(),
            self.o["machine"].detach(),
            self.o["time"].detach(),
            self.o["order"].detach(),
            self.o["job"].detach(),
            self.o["finished"].clone().detach(),
            self.validActions().detach(),
        )

    @torch.no_grad()
    def step(self, action: Tensor) -> Tuple[Tuple, Tensor, Tensor]:
        """Take a step in the environment

        Args:
            action (Tensor): the action to take, bool, Tensor(B, number of operations, number of machines).

        Returns:
            Tuple[Tuple, Tensor, Tensor]: the state of the environment, the reward of the action, whether the environment is done
        """

        makespan = torch.max(self.Tm, dim=-1)[0]

        action = action.reshape(self.batch, self.l, self.m)
        operation = action.sum(-1) == 1  # batch, l
        machine = action.sum(-2) == 1  # batch, m

        job = F.one_hot(self.o["job"][operation], num_classes=self.n) == 1  # batch, n

        time = self.o["time"][operation][machine]  # batch,

        done = torch.all(self.o["finished"], dim=-1)
        self.o["finished"][operation] = True

        endTime = torch.max(self.Tm[machine], self.Tn[job]) + time  # batch,

        self.Tm[machine] = endTime * (~done) + self.Tm[machine] * done
        self.Tn[job] = endTime * (~done) + self.Tn[job] * done

        dmakespan = torch.max(self.Tm, dim=-1)[0] - makespan

        s = torch.min(
            torch.min(self.Tm, dim=-1)[0], torch.min(self.Tm, dim=-1)[0]
        )  # batch,
        self.Tm -= s.reshape(-1, 1)
        self.Tn -= s.reshape(-1, 1)

        self.start += s

        state = (
            self.Tm.clone().detach(),
            self.Tn.clone().detach(),
            self.o["machine"].detach(),
            self.o["time"].detach(),
            self.o["order"].detach(),
            self.o["job"].detach(),
            self.o["finished"].clone().detach(),
            self.validActions().detach(),
        )
        reward = -dmakespan

        return state, reward, done


def RandomFJSSP(
    batch_size: int,
    n: int,
    m: int,
    h: int = None,
    different_h: bool = False,
    device: Union[str, torch.device] = "cpu",
):
    """generate random FJSSP problem"""

    h = m if h is None else h

    b = batch_size
    l = n * h

    machine = torch.randint(0, m, (b, l, int(0.8 * m)), device=device)
    machine = F.one_hot(machine, num_classes=m).sum(dim=-2) > 0

    time = torch.randint(1, 91, (b, l), device=device) / 100.0
    time = machine * time.reshape(b, l, 1)

    difftime = torch.randint(0, 10, (b, l, m), device=device) / 100.0
    time += machine * difftime

    order = (
        torch.arange(0, h, device=device)
        .reshape(1, 1, h)
        .repeat(batch_size, n, 1)
        .reshape(b, l)
    )

    job = (
        torch.arange(0, n, device=device)
        .reshape(1, n, 1)
        .repeat(batch_size, 1, h)
        .reshape(b, l)
    )

    if different_h:
        finished = torch.rand(b, l, device=device) > 0.8
    else:
        finished = torch.zeros(b, l, dtype=torch.bool, device=device)

    operations = {
        "machine": machine,
        "time": time,
        "order": order,
        "job": job,
        "finished": finished,
    }

    return {"n": n, "m": m, "o": operations}
