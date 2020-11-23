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
parser.add_argument("--fact_ref", type=str, required=True)
parser.add_argument("--model1", type=str, required=True)
parser.add_argument("--model2", type=str, required=True)
parser.add_argument("--size1", type=int, default=1024)
parser.add_argument("--size2", type=int, default=1024)
parser.add_argument("-o", "--output", type=str, default="output")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument('--truncation_mean', type=int, default=4096)
parser.add_argument("-r", "--randnum", type=int, default=10)
parser.add_argument("--fact_base1", type=str, required=True)
parser.add_argument("--fact_base2", type=str, required=True)

args = parser.parse_args()
device = args.device

## load eigvec
eigvec1 = torch.load(args.fact_base1)["eigvec"].to(args.device)
eigvec1.requires_grad = False
eigvec2 = torch.load(args.fact_base2)["eigvec"].to(args.device)
eigvec2.requires_grad = False

fact_path = args.fact
item = torch.load(fact_path)
vec = next(iter(item.values()))['weight'].to(device)

fact_path_ref = args.fact_ref
item_ref = torch.load(fact_path_ref)
vec_ref = next(iter(item_ref.values()))['weight'].to(device)

input_latent = torch.mm(vec, eigvec1) 
style_latent = torch.mm(vec_ref, eigvec2) 

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

## noise
noises_single = g_ema2.make_noise()
noises = []
for noise in noises_single:
    noises.append(noise.repeat(1, 1, 1, 1).normal_())
noise_normalize_(noises)

## gen images
with torch.no_grad():
    mean_latent2 = g_ema2.mean_latent(args.truncation_mean)

# generate ref and identity
swap_res = []
swap_ref_res = []
for j in range(1, 6, 2):
    img1, swap_res_i = g_ema1([input_latent], truncation=0.5, truncation_latent=mean_latent2, save_for_swap=True, swap_layer=j)
    swap_res.append(swap_res_i)

    img2, swap_ref_res_i = g_ema2([style_latent], truncation=0.5, truncation_latent=mean_latent2, save_for_swap=True, swap_layer=j)
    swap_ref_res.append(swap_ref_res_i)

# swap=5
img3, _ = g_ema2([input_latent], truncation=0.5, truncation_latent=mean_latent2, swap=True, swap_layer=5, swap_tensor=swap_res[2],  multi_style=True, multi_style_layers=3,  multi_style_latent=[style_latent])
img3_name = args.output + "_ls5_" + ".png"
img3 = make_image(img3)
out3 = Image.fromarray(img3[0])
out3.save(img3_name)

# swap=3
img4, _ = g_ema2([input_latent], truncation=0.5, truncation_latent=mean_latent2, swap=True, swap_layer=3, swap_tensor=swap_res[1],  multi_style=True, multi_style_layers=3,  multi_style_latent=[style_latent])
img4_name = args.output + "_ls3_" + ".png"
img4 = make_image(img4)
out4 = Image.fromarray(img4[0])
out4.save(img4_name)

# swap=1
img5, _ = g_ema2([input_latent], truncation=0.5, truncation_latent=mean_latent2, swap=True, swap_layer=1, swap_tensor=swap_res[0],  multi_style=True, multi_style_layers=3,  multi_style_latent=[style_latent])
img5_name = args.output + "_ls1_" + ".png"
img5 = make_image(img5)
out5 = Image.fromarray(img5[0])
out5.save(img5_name)


