#!/usr/bin/env python -W ignore::DeprecationWarning
import torch
from data.dataset import Pataka_Dataset
from train.train_vae import train_vae
from train.train_diffusion import train_diffusion
import warnings
from models.vae import VAE
from utils.utils import test_vae, sample_plot_image, eval_class_pred_diff, read_config
from models.unet import UNet
from sklearn.metrics import classification_report

warnings.warn = lambda *args, **kwargs: None


def main():
    args = read_config("config/Configuration.json")
    SEED = args["optimization_parameters"]["seed"]
    if args["optimization_parameters"]["device"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)

    # Load test data
    test_dataset = Pataka_Dataset(
        DBs=["Gita", "Neurovoz"], train_size=0.91, mode="test", seed=SEED
    )
    test_dataset = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args["optimization_parameters"]["batch_size"],
        shuffle=True,
    )

    if not args["flags"]["train_vae"]:
        # load pretrained VAE model
        vae = VAE(
            args["model_parameters"]["in_channels"],
            z_dim=args["model_parameters"]["latent_dim"],
        ).to(device)
        vae.load_state_dict(
            torch.load(args["paths"]["vae_model_path"], map_location=device)[
                "model_state_dict"
            ]
        )
    else:
        # train VAE from scratch
        vae_dataset = Pataka_Dataset(
            DBs=["Gita", "Neurovoz", "Saarbruecken"],
            train_size=0.91,
            mode="train",
            seed=SEED,
        )
        vae_dataset = torch.utils.data.DataLoader(
            vae_dataset,
            batch_size=args["optimization_parameters"]["batch_size"],
            shuffle=True,
        )

        vae = train_vae(
            vae_dataset,
            test_dataset,
            x_dim=args["model_parameters"]["in_channels"],
            z_dim=args["model_parameters"]["latent_dim"],
            epochs=args["optimization_parameters"]["num_epochs_vae"],
            lr=args["optimization_parameters"]["learning_rate_vae"],
            device=device,
        )
    # test vae reconstructions quality
    if args["flags"]["test_vae"]:
        test_vae(
            vae,
            test_dataset,
            device,
            save_path="img_samples/",
            n_samples=10,
            save_reconstructions=False,
            save_latent_space=False,
            save_samples=False,
        )

    # train diffusion model
    if args["flags"]["train_diffusion"]:
        if args["flags"]["train_vae"]:
            diff_dataset = vae_dataset
        else:
            # train diffusion model
            diff_dataset = Pataka_Dataset(
                DBs=["Gita", "Neurovoz", "Saarbruecken"],
                train_size=0.91,
                mode="train",
                seed=SEED,
            )
            diff_dataset = torch.utils.data.DataLoader(
                diff_dataset,
                batch_size=args["optimization_parameters"]["batch_size"],
                shuffle=True,
            )

        diffusion_model = train_diffusion(
            vae,
            args["model_parameters"]["diffusion_steps"],
            diff_dataset,
            test_dataset,
            x_dim=args["model_parameters"]["in_channels"],
            z_dim=args["model_parameters"]["latent_dim"],
            epochs=args["optimization_parameters"]["num_epochs_diff"],
            lr=args["optimization_parameters"]["learning_rate_diff"],
            pred_diff_time=args["model_parameters"]["pred_diff_time"],
            device=device,
        )

        torch.save(
            {
                "model_state_dict": diffusion_model.state_dict(),
            },
            "saved_models/diffusion.pth",
        )

    if args["flags"]["sample_diffusion"]:
        # load diffusion model
        diffusion_model = UNet(
            in_channels=args["model_parameters"]["in_channels"],
            out_channels=1,
            num_classes=4,
            init_features=args["model_parameters"]["latent_dim"],
        ).to(device)
        diffusion_model.load_state_dict(
            torch.load("saved_models/diffusion.pth", map_location=device)[
                "model_state_dict"
            ]
        )

        # test diffusion model
        sample_plot_image(
            vae,
            diffusion_model,
            args["model_parameters"]["diffusion_steps"],
            args["model_parameters"]["latent_dim"],
            device,
            "img_samples/",
        )

    if args["flags"]["eval_classpred"]:
        # load diffusion model
        diffusion_model = UNet(
            in_channels=args["model_parameters"]["in_channels"],
            out_channels=1,
            num_classes=4,
            init_features=args["model_parameters"]["latent_dim"],
        ).to(device)
        diffusion_model.load_state_dict(
            torch.load("saved_models/diffusion.pth", map_location=device)[
                "model_state_dict"
            ]
        )
        true_labels, pred_labels, true_labels_speaker, pred_labels_speaker = (
            eval_class_pred_diff(
                test_dataset,
                vae,
                diffusion_model,
                args["model_parameters"]["diffusion_steps"],
                device,
                pred_T=args["model_parameters"]["pred_diff_time"],
            )
        )
        print("Frame-based Classification report:")
        print(
            classification_report(
                true_labels.detach().cpu(), pred_labels.detach().cpu()
            )
        )
        print("Patient-based Classification report:")
        print(
            classification_report(
                true_labels_speaker.detach().cpu(), pred_labels_speaker.detach().cpu()
            )
        )

    return


if __name__ == "__main__":
    main()
