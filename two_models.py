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

parser = argparse.ArgumentParser()
parser.add_argument("--model1", type=str, required=True)
parser.add_argument("--model2", type=str, required=True)
parser.add_argument("--size1", type=int, default=1024)
parser.add_argument("--size2", type=int, default=1024)
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument('--truncation_mean', type=int, default=4096)
#parser.add_argument("factor_face", type=str)
#parser.add_argument("factor_metface", type=str)

args = parser.parse_args()

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

## noise
noises_single = g_ema1.make_noise()
noises = []
for noise in noises_single:
    noises.append(noise.repeat(1, 1, 1, 1).normal_())
noise_normalize_(noises)

## gen images
with torch.no_grad():
    #mean_latent1 = g_ema1.mean_latent(args.truncation_mean)
    mean_latent1 = g_ema2.mean_latent(args.truncation_mean)
    mean_latent2 = g_ema2.mean_latent(args.truncation_mean)
    #mean_latent2 = g_ema1.mean_latent(args.truncation_mean)

img1, _ = g_ema1([sample_z], truncation=0.5, truncation_latent=mean_latent1) 
img1_name = args.output + "_a.png" 
img1 = make_image(img1)
out1 = Image.fromarray(img1[0])
out1.save(img1_name)

img2, _ = g_ema2([sample_z], truncation=0.5, truncation_latent=mean_latent2) 
img2_name = args.output + "_b.png" 
img2 = make_image(img2)
out2 = Image.fromarray(img2[0])
out2.save(img2_name)
