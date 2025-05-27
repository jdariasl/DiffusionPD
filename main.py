#!/usr/bin/env python -W ignore::DeprecationWarning
import torch
from data.dataset import Pataka_Dataset
from train.train_vae import train_vae
from train.train_diffusion import train_diffusion
import warnings
from models.vae import VAE
from utils.utils import (
    test_vae,
    sample_plot_image,
    eval_class_pred_diff,
    eval_class_pred_diff_scheduler,
    read_config,
    sample_plot_image_scheduler,
    plot_kde_and_roc,
    pred_T_effect,
)
from models.unet import UNet, BatchNormlizer
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
        if not args["flags"]["resume_training_vae"]:
            # train VAE from scratch
            vae = train_vae(
                vae_dataset,
                test_dataset,
                x_dim=args["model_parameters"]["in_channels"],
                z_dim=args["model_parameters"]["latent_dim"],
                epochs=args["optimization_parameters"]["num_epochs_vae"],
                lr=args["optimization_parameters"]["learning_rate_vae"],
                device=device,
            )
        elif args["flags"]["resume_training_vae"]:
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
            vae.train()

            # train VAE from scratch
            vae = train_vae(
                vae_dataset,
                test_dataset,
                x_dim=args["model_parameters"]["in_channels"],
                z_dim=args["model_parameters"]["latent_dim"],
                epochs=args["optimization_parameters"]["num_epochs_vae"],
                lr=args["optimization_parameters"]["learning_rate_vae"],
                device=device,
                resume_training=True,
                vae=vae,
            )
        # save vae
        torch.save(
            {
                "model_state_dict": vae.state_dict(),
            },
            "saved_models/vae.pth",
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
            if args["flags"]["resume_training_diff"]:
                # load pretrained diffusion model
                diffusion_model = UNet(
                    in_channels=args["model_parameters"]["in_channels"],
                    out_channels=1,
                    num_classes=4,
                    init_features=args["model_parameters"]["latent_dim"],
                ).to(device)
                diffusion_model.load_state_dict(
                    torch.load(
                        args["paths"]["diffusion_model_path"], map_location=device
                    )["model_state_dict"]
                )
                diffusion_model, Norm = train_diffusion(
                    vae,
                    args["model_parameters"]["diffusion_steps"],
                    diff_dataset,
                    test_dataset,
                    x_dim=args["model_parameters"]["in_channels"],
                    z_dim=args["model_parameters"]["latent_dim"],
                    epochs=args["optimization_parameters"]["num_epochs_diff"],
                    lr=args["optimization_parameters"]["learning_rate_diff"],
                    lr_warmup_steps=args["optimization_parameters"]["lr_warmup_steps"],
                    pred_diff_time=args["model_parameters"]["pred_diff_time"],
                    device=device,
                    resume_training=True,
                    model=diffusion_model,
                )
            else:
                diffusion_model, Norm = train_diffusion(
                    vae,
                    args["model_parameters"]["diffusion_steps"],
                    diff_dataset,
                    test_dataset,
                    x_dim=args["model_parameters"]["in_channels"],
                    z_dim=args["model_parameters"]["latent_dim"],
                    epochs=args["optimization_parameters"]["num_epochs_diff"],
                    lr=args["optimization_parameters"]["learning_rate_diff"],
                    lr_warmup_steps=args["optimization_parameters"]["lr_warmup_steps"],
                    pred_diff_time=args["model_parameters"]["pred_diff_time"],
                    device=device,
                )

            torch.save(
                {
                    "model_state_dict": diffusion_model.state_dict(),
                },
                "saved_models/diffusion.pth",
            )
            torch.save(
                {
                    "model_state_dict": Norm.state_dict(),
                },
                "saved_models/normalizer.pth",
            )

    if args["flags"]["sample_diffusion"]:

        # load normalizer
        Norm = BatchNormlizer(num_features=args["model_parameters"]["latent_dim"]).to(
            device
        )
        Norm.load_state_dict(
            torch.load("saved_models/normalizer.pth", map_location=device)[
                "model_state_dict"
            ]
        )
        Norm.eval()

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
        sample_plot_image_scheduler(
            vae,
            Norm,
            diffusion_model,
            args["model_parameters"]["diffusion_steps"],
            args["model_parameters"]["latent_dim"],
            device,
            "img_samples/",
        )
        # sample_plot_image(
        #    vae,
        #    diffusion_model,
        #    args["model_parameters"]["diffusion_steps"],
        #    args["model_parameters"]["latent_dim"],
        #    device,
        #    "img_samples/",
        # )

    if args["flags"]["eval_classpred"]:
        # load normalizer
        # Norm = BatchNormlizer(num_features=args["model_parameters"]["latent_dim"]).to(
        #    device
        # )
        # Norm.load_state_dict(
        #    torch.load("saved_models/normalizer.pth", map_location=device)[
        #        "model_state_dict"
        #    ]
        # )
        # Norm.eval()
        # load diffusion model
        diffusion_model = UNet(
            in_channels=args["model_parameters"]["in_channels"],
            out_channels=1,
            num_classes=4,
            init_features=args["model_parameters"]["latent_dim"],
        ).to(device)
        diffusion_model.load_state_dict(
            torch.load(args["paths"]["diffusion_model_path"], map_location=device)[
                "model_state_dict"
            ]
        )
        (
            true_labels,
            pred_labels,
            scores,
            true_labels_speaker,
            pred_labels_speaker,
            scores_speaker,
        ) = eval_class_pred_diff_scheduler(
            test_dataset,
            vae,
            diffusion_model,
            args["model_parameters"]["diffusion_steps"],
            device,
            pred_T=args["model_parameters"]["pred_diff_time"],
        )
        plot_kde_and_roc(
            true_labels.detach().numpy(),
            scores.detach().numpy(),
            true_labels_speaker.detach().numpy(),
            scores_speaker.detach().numpy(),
            filename="performance_plot.png",
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

    if args["flags"]["pred_T_effect"]:

        # load diffusion model
        diffusion_model = UNet(
            in_channels=args["model_parameters"]["in_channels"],
            out_channels=1,
            num_classes=4,
            init_features=args["model_parameters"]["latent_dim"],
        ).to(device)
        diffusion_model.load_state_dict(
            torch.load(args["paths"]["diffusion_model_path"], map_location=device)[
                "model_state_dict"
            ]
        )
        AUC, AUC_speaker, Accuracy, Accuracy_speaker = pred_T_effect(
            test_dataset,
            vae,
            diffusion_model,
            args["model_parameters"]["diffusion_steps"],
            device,
        )
        print("AUC: ", AUC)
    return


if __name__ == "__main__":
    main()
