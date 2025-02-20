#!/usr/bin/env python -W ignore::DeprecationWarning
import argparse
import torch
from data.dataset import Pataka_Dataset
from train.train_vae import train_vae, test_vae
import warnings
from models.vae import VAE
from utils.utils import test_vae

warnings.warn = lambda *args, **kwargs: None


def main():
    args = get_arguments()
    SEED = args.seed
    if args.device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)

    # Load test data
    test_dataset = Pataka_Dataset(
        DBs=["Gita", "Neurovoz"], train_size=0.91, mode="test", seed=SEED
    )
    test_dataset = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True
    )

    if args.pretrained_vae:
        vae = VAE(args.inChannels, z_dim=args.latent_dim).to(device)
        vae.load_state_dict(torch.load(args.vae_path, map_location=device)["model_state_dict"])
    else:
        vae_dataset = Pataka_Dataset(
            DBs=["Gita", "Neurovoz", "Saarbruecken"],
            train_size=0.91,
            mode="train",
            seed=SEED,
        )
        vae_dataset = torch.utils.data.DataLoader(
            vae_dataset, batch_size=args.batch_size, shuffle=True
        )

        vae = train_vae(
            vae_dataset,
            test_dataset,
            x_dim=args.inChannels,
            z_dim=args.latent_dim,
            epochs=args.nEpochs,
            lr=args.lr,
            device=device,
        )

    if args.test_vae:
        test_vae(
            vae,
            test_dataset,
            device,
            save_path="vae_samples/",
            n_samples=10,
            save_reconstructions=False,
            save_latent_space=False,
            save_samples=False,
        )
    return


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--new_training",
        action="store_true",
        default=False,
        help="load saved_model as initial model",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--nEpochs", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--inChannels", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument(
        "--lr", default=2e-5, type=float, help="learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--weight_decay", default=1e-7, type=float, help="weight decay (default: 1e-6)"
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--test_vae",
        action="store_true",
        default=False,
        help="test vae reconstruction",
    )
    parser.add_argument(
        "--pretrained_vae",
        action="store_false",
        default=True,
        help="Use pretrained VAE model",
    )
    parser.add_argument(
        "--vae_path",
        default="saved_models/vae_multilingual_Gita_Neurovoz.pth",
        type=str,
        metavar="PATH",
        help="path to pretrained vae model",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
