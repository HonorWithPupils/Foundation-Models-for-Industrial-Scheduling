import os
import glob
import argparse

import time
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from accelerate import Accelerator

from src import FJSP
from src.models.actor_critic import Actor, Critic
from src.rl_ppo import Buffer, ppoloss, vloss, test

from tqdm.auto import trange

from torch.utils.tensorboard import SummaryWriter


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Create the parser
parser = argparse.ArgumentParser(description="Set game parameters and model parameters")

# Add command line arguments
parser.add_argument("--n", type=int, default=10, help="FJSSP parameter n")
parser.add_argument("--m", type=int, default=5, help="FJSSP parameter m")
parser.add_argument("--h", type=int, default=5, help="FJSSP parameter h")

parser.add_argument(
    "--config_actor_deep", type=int, default=4, help="Model parameter config_actor_deep"
)
parser.add_argument(
    "--config_actor_dim", type=int, default=512, help="Model parameter config_actor_dim"
)
parser.add_argument(
    "--config_actor_n_head",
    type=int,
    default=16,
    help="Model parameter config_actor_n_head",
)
parser.add_argument(
    "--config_actor_mlp_ratio",
    type=int,
    default=4,
    help="Model parameter config_actor_mlp_ratio",
)

parser.add_argument(
    "--config_critic_deep",
    type=int,
    default=4,
    help="Model parameter config_critic_deep",
)
parser.add_argument(
    "--config_critic_dim",
    type=int,
    default=512,
    help="Model parameter config_critic_dim",
)
parser.add_argument(
    "--config_critic_n_head",
    type=int,
    default=16,
    help="Model parameter config_critic_n_head",
)
parser.add_argument(
    "--config_critic_mlp_ratio",
    type=int,
    default=4,
    help="Model parameter config_critic_mlp_ratio",
)

parser.add_argument(
    "--n_epoch", type=int, default=25000, help="Training parameter n_epoch"
)
parser.add_argument(
    "--batch_size_envs",
    type=int,
    default=32,
    help="Training parameter batch_size_envs",
)
parser.add_argument(
    "--batch_size_ppo", type=int, default=512, help="Training parameter batch_size_ppo"
)
parser.add_argument(
    "--lr_actor", type=float, default=1e-4, help="Training parameter lr_actor"
)
parser.add_argument(
    "--lr_critic", type=float, default=1e-4, help="Training parameter lr_critic"
)

parser.add_argument(
    "--reuse_time", type=int, default=1, help="PPO parameter reuse_time"
)
parser.add_argument("--epsilon", type=float, default=0.1, help="PPO parameter epsilon")
parser.add_argument("--beta", type=float, default=0.01, help="PPO parameter beta")

parser.add_argument("--TIMESTAMP", type=str, default="", help="As Running Stamp")

# Parse command line arguments
args = parser.parse_args()

n, m, h = args.n, args.m, args.h
config_actor = {
    "deep": args.config_actor_deep,
    "dim": args.config_actor_dim,
    "n_head": args.config_actor_n_head,
    "mlp_ratio": args.config_actor_mlp_ratio,
}
config_critic = {
    "deep": args.config_critic_deep,
    "dim": args.config_critic_dim,
    "n_head": args.config_critic_n_head,
    "mlp_ratio": args.config_critic_mlp_ratio,
}

n_epoch = args.n_epoch
batch_size_envs = args.batch_size_envs
batch_size_ppo = args.batch_size_ppo
lr_actor = args.lr_actor
lr_critic = args.lr_critic

reuse_time = args.reuse_time
epsilon = args.epsilon
beta = args.beta

TIMESTAMP = args.TIMESTAMP


if __name__ == "__main__":

    accelerator = Accelerator()

    path = f"n={n}_m={m}_h={h}_epsilon={epsilon}_beta={beta}_actor_deep={config_actor['deep']}_width={config_actor['dim']}_critic_deep={config_critic['deep']}_width={config_critic['dim']}"

    torch.manual_seed(1)
    testFJSPinfo = FJSP.RandomFJSSP(batch_size_envs, n, m, h, device=device)

    actor = Actor(m, h, **config_actor).to(device)
    critic = Critic(m, h, **config_critic).to(device)

    opt_actor = optim.AdamW(actor.parameters(), lr=lr_actor)
    opt_critic = optim.AdamW(critic.parameters(), lr=lr_critic)

    actor, critic, opt_actor, opt_critic = accelerator.prepare(
        actor, critic, opt_actor, opt_critic
    )

    if TIMESTAMP == "":
        TIMESTAMP = f"{datetime.now():%Y-%m-%dT%H-%M-%S}"

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=f"tf_dir/{path}/{TIMESTAMP}/")

    folder_path = f"checkpoint/{path}/{TIMESTAMP}"

    if not os.path.exists(folder_path):
        print(f"No checkpoint. Train from the ground up.")
        start = 0
    else:
        files = glob.glob(os.path.join(folder_path, "*"))
        if not files:
            print(f"No checkpoint. Train from the ground up.")
            start = 0
        else:
            subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
            max_folder = max(subfolders, key=lambda x: int(os.path.basename(x)))
            folder_number = int(os.path.basename(max_folder))
            print(f"Load Check Point: {max_folder}")
            accelerator.load_state(max_folder)
            start = folder_number + 1

    if (
        not os.path.exists(f"models/{path}/{TIMESTAMP}/actor")
        and accelerator.is_main_process
    ):
        os.makedirs(f"models/{path}/{TIMESTAMP}/actor")
    if (
        not os.path.exists(f"models/{path}/{TIMESTAMP}/critic")
        and accelerator.is_main_process
    ):
        os.makedirs(f"models/{path}/{TIMESTAMP}/critic")

    pbar = trange(start, n_epoch, disable=not accelerator.is_main_process)

    for i in pbar:

        seed = accelerator.process_index * time.time()
        # Generate different random seeds for each machine
        torch.manual_seed(seed)

        FJSPinfo = FJSP.RandomFJSSP(batch_size_envs, n, m, h, device=device)
        envs = FJSP.Envs(FJSPinfo, device=device)
        buffer = Buffer(envs, actor)

        dataset = buffer.getDataset()
        dataloader = DataLoader(dataset, batch_size=batch_size_ppo, shuffle=True)

        for j in range(reuse_time):

            for data in dataloader:

                loss = vloss(critic, data, accelerator)

                opt_critic.zero_grad()
                loss.backward()

                if accelerator.sync_gradients:
                    accelerator.clip_grad_value_(critic.parameters(), 0.03)

                opt_critic.step()

        for j in range(reuse_time):

            for data in dataloader:

                loss = ppoloss(
                    actor,
                    critic,
                    data,
                    epsilon=epsilon,
                    beta=beta,
                    accelerator=accelerator,
                )

                opt_actor.zero_grad()
                loss.backward()

                if accelerator.sync_gradients:
                    accelerator.clip_grad_value_(actor.parameters(), 0.03)

                opt_actor.step()

        # Test
        if (i % 100 == 1 or i == 0) and accelerator.is_main_process:

            actor = actor.eval()

            makespan_random = test(
                actor, testFJSPinfo, accelerator, device=device, greedy=False
            )
            writer.add_scalar("makespan_random", makespan_random, i)

            makespan_greedy = test(
                actor, testFJSPinfo, accelerator, device=device, greedy=True
            )
            writer.add_scalar("makespan_greedy", makespan_greedy, i)

        # Check Point
        if (i % (n_epoch // 40) == (n_epoch // 40) - 1) and accelerator.is_main_process:
            accelerator.save_state(f"checkpoint/{path}/{TIMESTAMP}/{i}")

    # Savce final checkpoint and model
    if accelerator.is_main_process:
        accelerator.save_state(f"checkpoint/{path}/{TIMESTAMP}/{n_epoch}")
    if accelerator.is_main_process:
        torch.save(actor.state_dict(), f"models/{path}/{TIMESTAMP}/actor/epoch{i}.pth")
        torch.save(
            critic.state_dict(), f"models/{path}/{TIMESTAMP}/critic/epoch{i}.pth"
        )
