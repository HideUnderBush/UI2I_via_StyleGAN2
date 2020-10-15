import argparse

import torch
from torchvision import utils
from model import Generator
from torchsummary import summary
from tqdm import tqdm
parser = argparse.ArgumentParser()

parser.add_argument('--size', type=int, default=1024)
parser.add_argument('--truncation', type=float, default=1)
parser.add_argument('--truncation_mean', type=int, default=4096)
parser.add_argument('--ckpt', type=str, default="stylegan2-ffhq-config-f.pt")
parser.add_argument('--channel_multiplier', type=int, default=2)

args = parser.parse_args()

args.latent = 512
args.n_mlp = 8
with torch.no_grad():
    g_ema = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to('cuda')

    # Summarize Model

    ms = summary(g_ema, input_size=(1, 512))

    print(ms)
