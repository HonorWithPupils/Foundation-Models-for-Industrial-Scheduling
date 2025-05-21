import os
import glob
import argparse

from src import FJSP
from src.models.actor_critic import Actor, Critic
from src.inference import BeamSearch
from src.utils import loadFJSP, load_from_multiGPU
from src.visualization import plotGANTT

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

import numpy as np

# Create the parser
parser = argparse.ArgumentParser(description="Set game parameters and model parameters")

# Add command line arguments
parser.add_argument("--m", type=int, default=10, help="Number of machines")
parser.add_argument("--h", type=int, default=10, help="Number of operations per job")
parser.add_argument("--actor_deep", type=int, default=4, help="Depth of actor network")
parser.add_argument(
    "--actor_dim", type=int, default=512, help="Dimension of actor network"
)
parser.add_argument(
    "--actor_n_head", type=int, default=16, help="Number of heads in actor network"
)
parser.add_argument(
    "--actor_mlp_ratio", type=int, default=2, help="MLP ratio in actor network"
)
parser.add_argument(
    "--critic_deep", type=int, default=4, help="Depth of critic network"
)
parser.add_argument(
    "--critic_dim", type=int, default=512, help="Dimension of critic network"
)
parser.add_argument(
    "--critic_n_head", type=int, default=16, help="Number of heads in critic network"
)
parser.add_argument(
    "--critic_mlp_ratio", type=int, default=2, help="MLP ratio in critic network"
)
parser.add_argument(
    "--actor_path",
    type=str,
    default=r"models\Distillate4Demo\actor.pth",
    help="Path to actor model",
)
parser.add_argument(
    "--critic_path",
    type=str,
    default=r"models\Distillate4Demo\critic.pth",
    help="Path to critic model",
)
parser.add_argument(
    "--FJSPpath",
    type=str,
    default=r"data\HurinkBenchmark\HurinkVdata25.fjs",
    help="Path to FJSP data",
)
parser.add_argument(
    "--num_beams", type=int, default=32, help="Number of beams for beam search"
)
parser.add_argument(
    "--GANTTsavepath",
    type=str,
    default=r"results\Gantt.png",
    help="Path to save Gantt chart",
)

args = parser.parse_args()

m = args.m
h = args.h
config_actor = {
    "deep": args.actor_deep,
    "dim": args.actor_dim,
    "n_head": args.actor_n_head,
    "mlp_ratio": args.actor_mlp_ratio,
}
config_critic = {
    "deep": args.critic_deep,
    "dim": args.critic_dim,
    "n_head": args.critic_n_head,
    "mlp_ratio": args.critic_mlp_ratio,
}
actor_path = args.actor_path
critic_path = args.critic_path
FJSPpath = args.FJSPpath
num_beams = args.num_beams
GANTTsavepath = args.GANTTsavepath

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if __name__ == "__main__":

    actor = Actor(m, h, **config_actor).to(device)
    critic = Critic(m, h, **config_critic).to(device)

    actor.load_state_dict(load_from_multiGPU(r"models\Distillate4Demo\actor.pth"))
    critic.load_state_dict(load_from_multiGPU(r"models\Distillate4Demo\critic.pth"))

    FJSPinfo = loadFJSP(FJSPpath)
    FJSPinfo["o"]["time"] = FJSPinfo["o"]["time"] / 100.0

    makespan, actions = BeamSearch(FJSPinfo, actor, critic, num_beams=32, device=device)

    print(f"Makespan: {round(makespan * 100)}")

    actions = actions.cpu().numpy()
    FJSPinfo["o"]["time"] = FJSPinfo["o"]["time"] * 100

    plotGANTT(actions=actions, FJSPinfo=FJSPinfo, save_path=GANTTsavepath)
