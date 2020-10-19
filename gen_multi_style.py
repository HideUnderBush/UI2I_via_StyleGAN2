import os
import argparse
import math
import pdb

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from utils import AddPepperNoise
import numpy as np
from scipy.linalg import solve

import lpips
from model import Generator
import random
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--fact", type=str, required=True)
parser.add_argument("--model1", type=str, required=True)
parser.add_argument("--model2", type=str, required=True)
parser.add_argument("--size1", type=int, default=1024)
parser.add_argument("--size2", type=int, default=1024)
parser.add_argument("-o", "--output", type=str, default="output")
parser.add_argument("--swap_layer", type=int, default=3)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument('--truncation_mean', type=int, default=4096)
parser.add_argument("-r", "--stylenum", type=int, default=20)
parser.add_argument("--fact_base", type=str)

args = parser.parse_args()
device = args.device

## load eigvec
eigvec = torch.load(args.fact_base)["eigvec"].to(args.device)
eigvec.requires_grad = False

factor_path = args.fact
item = torch.load(factor_path)
vec = next(iter(item.values()))['weight'].to(device)
input_latent = torch.mm(vec, eigvec) 

def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)

def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )

device = args.device

# generate images
## load model
g_ema1 = Generator(args.size1, 512, 8)
g_ema1.load_state_dict(torch.load(args.model1, map_location='cuda:0')["g_ema"], strict=False)
g_ema1.eval()
g_ema1 = g_ema1.to(device)

g_ema2 = Generator(args.size2, 512, 8)
g_ema2.load_state_dict(torch.load(args.model2, map_location='cuda:0')["g_ema"], strict=False)
g_ema2.eval()
g_ema2 = g_ema2.to(device)

## prepare input vector
sample_z = torch.randn(1, 512, device=args.device)
sample_z_style = torch.randn(1, 512, device=args.device)

## noise
noises_single = g_ema2.make_noise()
noises = []
for noise in noises_single:
    noises.append(noise.repeat(1, 1, 1, 1).normal_())
noise_normalize_(noises)

## gen images
with torch.no_grad():
    mean_latent = g_ema2.mean_latent(args.truncation_mean)

img1, swap_res = g_ema1([input_latent], input_is_latent=True, save_for_swap=True, swap_layer=args.swap_layer)

for i in range(args.stylenum):
    sample_z_style = torch.randn(1, 512, device=args.device)
    img_style, _ = g_ema2([input_latent], truncation=0.5, truncation_latent=mean_latent, swap=True, swap_layer=args.swap_layer,  swap_tensor=swap_res, multi_style=True, multi_style_latent=[sample_z_style])
    print(i)
    img_style_name = args.output + "_style_" + str(i) + ".png"
    img_style = make_image(img_style)
    out_style = Image.fromarray(img_style[0])
    out_style.save(img_style_name)
