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

#torch.manual_seed(111)

def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


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
    parser.add_argument("--ckpt", type=str, required=True)
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
    parser.add_argument("files", metavar="FILES", nargs="+")

    args = parser.parse_args()
    eigvec = torch.load(args.fact)["eigvec"].to(args.device)
    eigvec.requires_grad = False

    n_mean_degree = 10000
    n_mean_weight = 10000

    resize = min(args.size, 256)

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    imgs = []

    for imgfile in args.files:
        img = transform(Image.open(imgfile).convert("RGB"))
        imgs.append(img)

    imgs = torch.stack(imgs, 0).to(device)

    g_ema = Generator(args.size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)
    trunc = g_ema.mean_latent(4096)

    with torch.no_grad():
        noise_sample = torch.randn(1, 512, device=device)
        latent = g_ema.get_latent(noise_sample)

        degree_sample = torch.randn(n_mean_degree, 1, device=device) * 1000
        weight_sample = torch.randn(n_mean_weight, 512, device=device).unsqueeze(1) # row of factor

        degree_mean = degree_sample.mean(0)
        degree_std = ((degree_sample - degree_mean).pow(2).sum() / n_mean_degree) ** 0.5

        weight_mean = weight_sample.mean(0)
        weight_std = ((weight_sample - weight_mean).pow(2).sum() / n_mean_weight) ** 0.5 

        direction = torch.mm(eigvec, weight_mean.T) 

    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )

    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

    direction_in = direction.T
    weight_in = weight_mean 

    weight_mean.requires_grad = True

    for noise in noises:
        noise.requires_grad = True

    optimizer = optim.Adam([weight_mean] + noises, lr=args.lr)

    pbar = tqdm(range(args.step))
    latent_path = []
    weight_path = []

    imgs.detach()

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = weight_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        weight_n = latent_noise(weight_in, noise_strength.item())
        direction_n = torch.mm(weight_in, eigvec) 

        img_gen, _ = g_ema([direction_n], input_is_latent=True, noise=noises)

        batch, channel, height, width = img_gen.shape

        if height > 256:
            factor = height // 256

            img_gen = img_gen.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            img_gen = img_gen.mean([3, 5])


        p_loss = percept(img_gen, imgs).sum()
        n_loss = noise_regularize(noises)
        mse_loss = F.mse_loss(img_gen, imgs)

        loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        noise_normalize_(noises)

        if (i + 1) % 100 == 0:
            direction_in = torch.mm(weight_in, eigvec)
            latent_path.append(direction_in.detach().clone())
            weight_path.append(weight_in.detach().clone())

        pbar.set_description(
            (
                f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
            )
        )

    img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises)

    filename = os.path.splitext(os.path.basename(args.files[0]))[0] + ".pt"

    img_ar = make_image(img_gen)

    result_file = {}
    for i, input_name in enumerate(args.files):
        noise_single = []
        for noise in noises:
            noise_single.append(noise[i : i + 1])

        result_file[input_name] = {
            "img": img_gen[i],
            "latent": latent_path[i],
            "weight": weight_path[i],
            "noise": noise_single,
        }

        img_name = os.path.splitext(os.path.basename(input_name))[0] + "-project.png"
        pil_img = Image.fromarray(img_ar[i])
        pil_img.save(img_name)

    torch.save(result_file, filename)
