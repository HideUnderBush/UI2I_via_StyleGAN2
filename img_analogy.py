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
import numpy

import lpips
from model import Generator

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt1", type=str, required=True)
parser.add_argument("--ckpt2", type=str, required=True)
parser.add_argument("-i", "--ckpt", type=str, required=True)
parser.add_argument("-m", "--model", type=str, required=True)
parser.add_argument("--size", type=int, default=1024)
#parser.add_argument("-a", "--apply", type=str, required=True)
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("factor_face", type=str)

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
vec2_key = list(torch.load(args.ckpt2).keys())[0]
vec_in_key = list(torch.load(args.ckpt).keys())[0]
vec1 = torch.load(args.ckpt1)[vec1_key]["weight"].to(device)
vec2 = torch.load(args.ckpt2)[vec2_key]["weight"].to(device)
vec_in = torch.load(args.ckpt)[vec_in_key]["weight"].to(device)

# distance
dist = vec2 - vec1

# max direction
max_index = torch.argmax(abs(dist))
print(max_index)

# full apply
vec_full_res = vec_in + dist

# max apply
vec_max_res = vec_in
vec_max_res[:, max_index] = vec_in[:, max_index] + dist[:, max_index]

# generate images
## load model
g_ema = Generator(args.size, 512, 8)
g_ema.load_state_dict(torch.load(args.model)["g_ema"], strict=False)
g_ema.eval()
g_ema = g_ema.to(device)

## load eigvec
eigvec = torch.load(args.factor_face)["eigvec"].to(args.device)
eigvec.requires_grad = False

## prepare input vector
input_full = torch.mm(vec_full_res, eigvec)
input_max = torch.mm(vec_max_res, eigvec)

## noise
noises_single = g_ema.make_noise()
noises = []
for noise in noises_single:
    noises.append(noise.repeat(1, 1, 1, 1).normal_())
noise_normalize_(noises)

## gen images
img_full, _ = g_ema([input_full], input_is_latent=True, noise = noises) 
img_max, _ = g_ema([input_max], input_is_latent=True, noise = noises) 

img_full_name = args.output + "full.png" 
img_max_name = args.output + "max.png" 

img_full = make_image(img_full)
img_max = make_image(img_max)

full_out = Image.fromarray(img_full[0])
max_out = Image.fromarray(img_max[0])

full_out.save(img_full_name)
max_out.save(img_max_name)
