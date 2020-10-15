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
parser.add_argument("--factor", type=str, required=True)
parser.add_argument("--factor_ref", type=str, required=True)
parser.add_argument("--model1", type=str, required=True)
parser.add_argument("--model2", type=str, required=True)
# parser.add_argument("--model3", type=str, required=True)
parser.add_argument("--size1", type=int, default=1024)
parser.add_argument("--size2", type=int, default=1024)
# parser.add_argument("--size3", type=int, default=1024)
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument('--truncation_mean', type=int, default=4096)
parser.add_argument("-r", "--randnum", type=int, default=10)
parser.add_argument("factor_face", type=str)
#parser.add_argument("factor_face", type=str)
#parser.add_argument("factor_metface", type=str)

args = parser.parse_args()
device = args.device

#torch.manual_seed(8009268845030607599)
#seed = random.randrange(sys.maxsize)
seed=8009268845030607599
###torch.manual_seed(seed)
###print("Seed was:", seed)

## load eigvec
eigvec = torch.load(args.factor_face)["eigvec"].to(args.device)
eigvec.requires_grad = False

factor_path = args.factor
item = torch.load(factor_path)
vec = next(iter(item.values()))['weight'].to(device)

factor_path_ref = args.factor_ref
item_ref = torch.load(factor_path_ref)
vec_ref = next(iter(item_ref.values()))['weight'].to(device)

input_latent = torch.mm(vec, eigvec) 
style_latent = torch.mm(vec_ref, eigvec) 

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

# g_ema3 = Generator(args.size3, 512, 8)
# g_ema3.load_state_dict(torch.load(args.model3, map_location='cuda:0')["g_ema"], strict=False)
# #g_ema3.style = g_ema1.style
# g_ema3.eval()
# g_ema3 = g_ema3.to(device)

## prepare input vector
sample_z_style = torch.randn(1, 512, device=args.device)
sample_z1 = torch.randn(1, 512, device=args.device)

num = 5
sample_z = []
sample_ref_z = []
for i in range(num):
    sample_zi = torch.randn(1, 512, device=args.device)
    sample_ref_zi = torch.randn(1, 512, device=args.device)

    sample_z.append(sample_zi)
    sample_ref_z.append(sample_ref_zi)

## noise
noises_single = g_ema2.make_noise()
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

# generate ref and identity
swap_res = []
swap_total = []

swap_ref_res = []
swap_ref_total = []
for i in range(num):
    print("processing base figure [{}/{}]".format(i, num))
    # identity 
    for j in range(1, 6, 2):
        img1, swap_res_i = g_ema1([sample_z[i]], truncation=0.5, truncation_latent=mean_latent2, save_for_swap=True, swap_layer=j)
        swap_res.append(swap_res_i)

        img2, swap_ref_res_i = g_ema2([sample_ref_z[i]], truncation=0.5, truncation_latent=mean_latent2, save_for_swap=True, swap_layer=j)
        swap_ref_res.append(swap_ref_res_i)

    img1_name = args.output + str(i) + "_identity.png"
    img1 = make_image(img1)
    out1 = Image.fromarray(img1[0])
    out1.save(img1_name)
    swap_total.append(swap_res)
    swap_res = []

    # ref
    img2_name = args.output + str(i) + "_ref.png"
    img2 = make_image(img2)
    out2 = Image.fromarray(img2[0])
    out2.save(img2_name)
    swap_ref_total.append(swap_ref_res)
    swap_ref_res = []

for i in range(num):
    print("processing I2I [{}/{}]".format(i, num))
    # swap=5, style=5
    img3, _ = g_ema2([sample_z[i]], truncation=0.5, truncation_latent=mean_latent2, swap=True, swap_layer=5, swap_tensor=swap_total[i][2],  multi_style=True, multi_style_layers=1,  multi_style_latent=[sample_ref_z[i]])
    img3_name = args.output + str(i) + "_w5s1_" + ".png"
    img3 = make_image(img3)
    out3 = Image.fromarray(img3[0])
    out3.save(img3_name)

    # swap=3, style=5
    img4, _ = g_ema2([sample_z[i]], truncation=0.5, truncation_latent=mean_latent2, swap=True, swap_layer=3, swap_tensor=swap_total[i][1],  multi_style=True, multi_style_layers=3,  multi_style_latent=[sample_ref_z[i]])
    img4_name = args.output + str(i) + "_w3s5_" + ".png"
    img4 = make_image(img4)
    out4 = Image.fromarray(img4[0])
    out4.save(img4_name)

    # swap=1, style=5
    img5, _ = g_ema2([sample_z[i]], truncation=0.5, truncation_latent=mean_latent2, swap=True, swap_layer=1, swap_tensor=swap_total[i][0],  multi_style=True, multi_style_layers=1,  multi_style_latent=[sample_ref_z[i]])
    img5_name = args.output + str(i) + "_w1s1_" + ".png"
    img5 = make_image(img5)
    out5 = Image.fromarray(img5[0])
    out5.save(img5_name)

    '''
    # swap=1, style=3
    img6, _ = g_ema2([sample_z[i]], truncation=0.5, truncation_latent=mean_latent2, swap=True, swap_layer=3, swap_tensor=swap_ref_total[i][1],  multi_style=True, multi_style_layers=3,  multi_style_latent=[sample_z[i]])
    #img6, _ = g_ema2([sample_z[i]], truncation=0.5, truncation_latent=mean_latent2, swap=True, swap_layer=3, swap_tensor=swap_ref_total[i][1],  multi_style=True, multi_style_layers=1,  multi_style_latent=[sample_ref_z[i]])
    img6_name = args.output + str(i) + "_wi3s1_" + ".png"
    img6 = make_image(img6)
    out6 = Image.fromarray(img6[0])
    out6.save(img6_name)

    # swap=1, style=1
    img7, _ = g_ema2([sample_z[i]], truncation=0.5, truncation_latent=mean_latent2, swap=True, swap_layer=1, swap_tensor=swap_ref_total[i][0],  multi_style=True, multi_style_layers=1,  multi_style_latent=[sample_ref_z[i]])
    img7_name = args.output + str(i) + "_wi1s1_" + ".png"
    img7 = make_image(img7)
    out7 = Image.fromarray(img7[0])
    out7.save(img7_name)
    '''






