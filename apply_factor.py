import argparse
import pdb

import torch
from torchvision import utils

from model import Generator

# fix random seed for reproductivity
torch.manual_seed(0)

if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--index", type=int, default=0)
    parser.add_argument("-d", "--degree", type=float, default=5)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("-n", "--n_sample", type=int, default=7)
    parser.add_argument("--truncation", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_prefix", type=str, default="factor")
    parser.add_argument("factor_face", type=str)

    args = parser.parse_args()

    eigvec = torch.load(args.factor_face)["eigvec"].to(args.device)
    ckpt = torch.load(args.ckpt, map_location='cuda:0')
    g = Generator(args.size, 512, 8).to(args.device)
    g.load_state_dict(ckpt["g_ema"], strict=False)

    trunc = g.mean_latent(4096)

    latent = torch.randn(args.n_sample, 512, device=args.device)
    latent = g.get_latent(latent)

    direction = args.degree * eigvec[:, args.index].unsqueeze(0)

    img, _ = g(
        [latent],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    img1, _ = g(
        [latent + direction],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    img2, _ = g(
        [latent - direction],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )

    grid = utils.save_image(
        img,
        f"{args.out_prefix}_index-{args.index}_degree-{args.degree}_0.png",
        normalize=True,
        range=(-1, 1),
        nrow=args.n_sample,
    )
    grid = utils.save_image(
        img1,
        f"{args.out_prefix}_index-{args.index}_degree-{args.degree}_1.png",
        normalize=True,
        range=(-1, 1),
        nrow=args.n_sample,
    ) 
    grid = utils.save_image(
        img2,
        f"{args.out_prefix}_index-{args.index}_degree-{args.degree}_2.png",
        normalize=True,
        range=(-1, 1),
        nrow=args.n_sample,
    )
