import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset


from src.FJSP import Envs


class Buffer:
    def __init__(self, envs, agent):

        self.envs = envs
        self.agent = agent

        self.n = envs.n
        self.m = envs.m
        self.l = envs.l
        self.device = envs.device

        self.update()

    @torch.no_grad()
    def update(self):

        states = []
        actions = []
        rewards = []
        probs = []
        states_next = []

        state = self.envs.reset()

        for _ in range(self.l):

            states.append(state)

            P = self.agent(*state)

            action = torch.multinomial(P, num_samples=1).view(-1)
            action = F.one_hot(action, num_classes=self.l * self.m)
            actions.append(action)

            prob = (action * P).sum(-1)
            probs.append(prob)

            state, reward, done = self.envs.step(action)

            rewards.append(reward)
            states_next.append(state)

        self.Makespan = -torch.stack(rewards, dim=-1).sum(-1).mean().item()

        R = [torch.zeros_like(reward) for _ in range(self.l)]
        tmp = torch.zeros_like(reward)
        for i in range(self.l - 1, -1, -1):
            R[i] = rewards[i] + tmp
            tmp = R[i]

        T = lambda x: list(zip(*x))
        states = T(states)
        states_next = T(states_next)

        self.data = [*states, actions, rewards, R, probs, *states_next]

        for i in range(len(self.data)):
            self.data[i] = torch.cat(self.data[i], dim=0).detach()

    def getDataset(self):

        return TensorDataset(*self.data)


@torch.no_grad()
def test(actor, FJSPinfo, accelerator, device="cpu", greedy=False):
    """Test the preformance on the FJSP problem"""

    n, m = FJSPinfo["n"], FJSPinfo["m"]

    envs = Envs(FJSPinfo, device=device)

    r = []

    state = envs.reset()

    for _ in range(envs.l):

        with accelerator.autocast():
            P = actor(*state)

        if greedy:
            action = P.max(-1)[1].view(-1)
        else:
            action = torch.multinomial(P, num_samples=1).view(-1)
        action = F.one_hot(action, num_classes=envs.l * m)

        state, reward, done = envs.step(action)
        r.append(reward)

    r = torch.stack(r, dim=0)
    makespan = -r.sum(0)

    return makespan.mean().item()


def vloss(critic, data, accelerator):
    """Critic loss function"""

    critic = critic.train()

    S, r, R, S_next = data[:8], data[9], data[10], data[-8:]

    with accelerator.autocast():
        V = critic(*S)

    return ((V - R.detach()) ** 2).mean()


def ppoloss(agent, critic, data, epsilon, beta, accelerator):
    """PPO loss function"""

    agent = agent.train()
    critic = critic.eval()

    S = data[:8]
    a, r, R, p_old = data[8:-8]
    S_next = data[-8:]

    with accelerator.autocast():
        probs = agent(*S)

    prob = (probs * a).sum(-1)

    with accelerator.autocast():
        with torch.no_grad():
            R = (R - critic(*S)).detach()

    ratio = prob / p_old

    L = torch.min(ratio * R, torch.clip(ratio, 1 - epsilon, 1 + epsilon) * R)

    # cross entropy
    valid = S[-1].reshape(probs.shape[0], -1)
    PriDis = ((1 - (valid == 0).sum(1) * 1e-10) / valid.sum(1))[
        :, None
    ] * valid + 1e-10 * (
        valid == 0
    )  # invalid move for 1e-10
    KLs = (probs * (torch.log(probs + 1e-10) - torch.log(PriDis))).sum(
        1
    )  # add 1e-10 to prevent -inf

    return -L.mean() + beta * KLs.mean()
