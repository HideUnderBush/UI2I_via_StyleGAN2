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
parser.add_argument("--model3", type=str, required=True)
parser.add_argument("--size1", type=int, default=1024)
parser.add_argument("--size2", type=int, default=1024)
parser.add_argument("--size3", type=int, default=1024)
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument('--truncation_mean', type=int, default=4096)

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

g_ema3 = Generator(args.size3, 512, 8)
g_ema3.load_state_dict(torch.load(args.model3, map_location='cuda:0')["g_ema"], strict=False)
#g_ema3.style = g_ema1.style
g_ema3.eval()
g_ema3 = g_ema3.to(device)

for i in range(1):
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
        mean_latent1 = g_ema2.mean_latent(args.truncation_mean)
        mean_latent2 = g_ema2.mean_latent(args.truncation_mean)
    swap_res = []
    for item in range(1,6,2):
        img1, swap_res_i = g_ema1([sample_z], truncation=1, truncation_latent=mean_latent1, save_for_swap=True, swap_layer=item)
        img1_name = args.output + str(i) + "_a.png"
        img1 = make_image(img1)
        out1 = Image.fromarray(img1[0])
        out1 = out1.resize((256,256))
        out1.save(img1_name)
        swap_res.append(swap_res_i)

    img2, _ = g_ema2([sample_z], truncation=0.5, truncation_latent=mean_latent2, swap=False)
    img2_name = args.output + str(i) + "_woft.png"
    img2 = make_image(img2)
    out2 = Image.fromarray(img2[0])
    out2 = out2.resize((256,256))
    out2.save(img2_name)

    img3, _ = g_ema3([sample_z], truncation=0.7, truncation_latent=mean_latent2, swap=False)
    img3_name = args.output + str(i) + "_ft.png"
    img3 = make_image(img3)
    out3 = Image.fromarray(img3[0])
    out3 = out3.resize((256,256))
    out3.save(img3_name)

    img3, _ = g_ema3([sample_z], truncation=0.7, truncation_latent=mean_latent2, swap=True, swap_tensor=swap_res[0], swap_layer=1)
    img3_name = args.output + str(i) + "_fcftl1.png"
    img3 = make_image(img3)
    out3 = Image.fromarray(img3[0])
    out3 = out3.resize((256,256))
    out3.save(img3_name)

    img4, _ = g_ema3([sample_z], truncation=0.7, truncation_latent=mean_latent2, swap=True, swap_tensor=swap_res[1], swap_layer=3)
    img4_name = args.output + str(i) + "_fcftl3.png"
    img4 = make_image(img4)
    out4 = Image.fromarray(img4[0])
    out4 = out4.resize((256,256))
    out4.save(img4_name)

    img5, _ = g_ema3([sample_z], truncation=0.7, truncation_latent=mean_latent2, swap=True, swap_tensor=swap_res[2], swap_layer=5)
    img5_name = args.output + str(i) + "_fcftl5.png"
    img5 = make_image(img5)
    out5 = Image.fromarray(img5[0])
    out5 = out5.resize((256,256))
    out5.save(img5_name)
    #clear swap_res
    swap_res = []
