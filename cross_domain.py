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
parser.add_argument("--ckpt1", type=str, required=True)
#parser.add_argument("--ckpt2", type=str, required=True)
parser.add_argument("-m", "--model", type=str, required=True)
parser.add_argument("--size", type=int, default=1024)
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("factor_face", type=str)
parser.add_argument("factor_metface", type=str)

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

# vectors 
vec1_key = list(torch.load(args.ckpt1).keys())[0]
#vec2_key = list(torch.load(args.ckpt2).keys())[0]
#vec_in_key = list(torch.load(args.ckpt).keys())[0]
vec1 = torch.load(args.ckpt1)[vec1_key]["weight"].to(device)
#vec2 = torch.load(args.ckpt2)[vec2_key]["weight"].to(device)
#vec_in = torch.load(args.ckpt)[vec_in_key]["weight"].to(device)

# generate images
## load model
g_ema = Generator(args.size, 512, 8)
g_ema.load_state_dict(torch.load(args.model)["g_ema"], strict=False)
g_ema.eval()
g_ema = g_ema.to(device)

## load eigvec
eigvec1 = torch.load(args.factor_face)["eigvec"].to(args.device)
eigvec1_np = eigvec1.cpu().numpy()
eigvec1_norm = np.linalg.norm(eigvec1_np)
#eigvec1_np = eigvec1_np - np.min(eigvec1_np)/ (np.max(eigvec1_np) - np.min(eigvec1_np)) 
eigvec1.requires_grad = False

eigvec2 = torch.load(args.factor_metface)["eigvec"].to(args.device)
eigvec2_np = eigvec2.cpu().numpy()
eigvec2_norm = np.linalg.norm(eigvec2_np)
#eigvec2_np = eigvec2_np - np.min(eigvec2_np)/ (np.max(eigvec2_np) - np.min(eigvec2_np)) 
eigvec2.requires_grad = False

## solve transform calculator
V = solve(eigvec1_np, eigvec2_np)
V_t = torch.from_numpy(V).to(args.device)

## prepare input vector
vec_cross = torch.mm(vec1, eigvec1)
#vec_cross = torch.mm(vec_cross, eigvec2)

## noise
noises_single = g_ema.make_noise()
noises = []
for noise in noises_single:
    noises.append(noise.repeat(1, 1, 1, 1).normal_())
noise_normalize_(noises)

## gen images
img, _ = g_ema([vec_cross], input_is_latent=True, noise = noises) 

img_name = args.output + "_cross.png" 

img = make_image(img)

out = Image.fromarray(img[0])

out.save(img_name)
