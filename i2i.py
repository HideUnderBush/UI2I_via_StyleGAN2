import pdb
import argparse
import math
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from utils import AddPepperNoise

import lpips
from model import Generator

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


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr_rampdown", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--noise_ramp", type=float, default=0.75)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--noise_regularize", type=float, default=1e5)
    parser.add_argument("--mse", type=float, default=0)
    parser.add_argument("--w_plus", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fact", type=str,required=True )

    args = parser.parse_args()
    data = torch.load(args.fact)
    input_name = list(data.keys())[0] 
    latent = data[input_name]['latent'].to(device)
    latent.requires_grad = False

    g_ema2 = Generator(args.size, 512, 8)
    g_ema2.load_state_dict(torch.load(args.model)["g_ema"], strict=False)
    g_ema2.eval()
    g_ema2 = g_ema2.to(device)

    noises_single = g_ema2.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(1, 1, 1, 1).normal_())

    img_i2i, _ = g_ema2([latent], input_is_latent=True, noise=noises)
    img_i2i = make_image(img_i2i)
    #img_i2i_name = os.path.splitext(os.path.basename(input_name))[0] + "-i2i.png"
    img_i2i_name = "multi-domain-test.png"
    pil_img = Image.fromarray(img_i2i.squeeze())
    pil_img.save(img_i2i_name)



