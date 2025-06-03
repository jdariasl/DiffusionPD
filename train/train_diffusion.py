import torch
import torch.nn.functional as F
from models.unet import UNet
from utils.utils import set_diffusion_params, sample_plot_image_scheduler
import torch.optim as optim
from losses.losses import diff_loss
from utils.utils import forward_diffusion_sample, reverse_diff
from torch.optim.lr_scheduler import ExponentialLR
from torch.cuda.amp import GradScaler
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from .sampler_diffusion import Class_DDPMPipeline
import wandb


def train_diffusion(
    vae,
    time_steps,
    train_loader,
    test_loader,
    x_dim,
    z_dim,
    epochs,
    lr,
    lr_warmup_steps,
    pred_diff_time,
    device,
    resume_training=False,
    model=None,
):
    # Start a new wandb run to track this script.
    wandb.login(key="d3ab2dfb9825e30136d030033cfd019c6dc63407")
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="jdariasl",
        # Set the wandb project where this run will be logged.
        project="Diffusion_PD",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": lr,
            "lr_warmup_steps": lr_warmup_steps,
            "architecture": "1DUNet",
            "dataset": "Saarbruecken_Gita_Neurovoz",
            "epochs": epochs,
            "batch_size": train_loader.batch_size,
            "latent_dim": z_dim,
            "time_steps": time_steps,
            "pred_diff_time": pred_diff_time,
        },
    )
    # best_loss_val = 1e10
    scaler_diffusion = GradScaler()
    if resume_training:
        model = model
    else:
        # backbone model
        model = UNet(
            in_channels=x_dim, out_channels=1, num_classes=4, init_features=z_dim
        ).to(device)

    # set constant diffusion parameters
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=time_steps,
        beta_schedule="linear",
        beta_start=0.0015,
        beta_end=0.0195,
    )

    # diff_params = set_diffusion_params(T=time_steps)
    # params = list(model.parameters()) + list(Norm.parameters())
    # optimizer = optim.Adam(params, lr=lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=(len(train_loader) * epochs),
    )

    # scheduler = ExponentialLR(optimizer, gamma=0.9)

    vae.eval()
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, _, label, _, _) in enumerate(train_loader):
            data = data.to(device)
            label = label.long().to(device)
            with torch.no_grad():
                mu, _ = vae.encode(data)
                mu = mu.to(device)
            
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda"):
                t = torch.randint(0, time_steps, (data.shape[0],)).to(device)
                noise = torch.randn_like(mu).to(device)
                noisy_data = noise_scheduler.add_noise(mu, noise, t)
                noise_pred = model(noisy_data, label, t)
                loss = F.mse_loss(noise_pred, noise)
                # loss = diff_loss(
                #    model,
                #    mu,
                #    label,
                #    t,
                #    diff_params["sqrt_alphas_cumprod"],
                #    diff_params["sqrt_one_minus_alphas_cumprod"],
                #    device,
                # )
            scaler_diffusion.scale(loss).backward()
            scaler_diffusion.step(optimizer)
            scaler_diffusion.update()
            lr_scheduler.step()

            # loss.backward()
            train_loss += loss.item()
            # optimizer.step()
        print(
            "====> Epoch: {} Average diff loss: {:.4f}".format(
                epoch, train_loss / len(train_loader.dataset)
            )
        )

        test_loss = test_diffusion_scheduler(
            vae, test_loader, model, noise_scheduler, device, pred_diff_time
        )
        # scheduler.step()
        # Log metrics to wandb.
        image = sample_plot_image_scheduler(
            vae,
            model.eval(),
            time_steps,
            z_dim,
            device,
            "img_samples/",
            n=1,
            return_image=True,
        )
        images = wandb.Image(image, caption="Top: Output, Bottom: Input")

        run.log(
            {
                "train_loss_mse": train_loss / len(train_loader.dataset),
                "test_loss_class_dist": test_loss,
                "gen_img": images,
            }
        )
    run.finish()
    return model


def test_diffusion_scheduler(
    vae, test_loader, model, noise_scheduler, device, T_pred=50
):
    model.eval()
    pipeline = Class_DDPMPipeline(unet=model, scheduler=noise_scheduler)
    test_loss = 0
    with torch.no_grad():
        for i, (data, _, label, _, _) in enumerate(test_loader):
            data = data.to(device)
            mu, _ = vae.encode(data)
            mu = mu.to(device)

            noise = torch.randn_like(mu).to(device)
            mu = noise_scheduler.add_noise(
                mu,
                noise,
                T_pred * torch.ones(mu.shape[0], dtype=torch.int64).to(device),
            )

            # mu = Norm(mu)
            label = label.long().to(device)
            recover_spec = pipeline(
                init_samples=mu,
                class_labels=label,
                num_inference_steps=T_pred,
                generator=torch.Generator(device="cpu").manual_seed(1234),
                device=device,
            )

            label_not = (~label.bool()).long().to(device)
            recover_spec_not = pipeline(
                init_samples=mu,
                class_labels=label_not,
                num_inference_steps=T_pred,
                generator=torch.Generator(device="cpu").manual_seed(1234),
                device=device,
            )

            loss = torch.mean(
                torch.mean(torch.abs(recover_spec - mu), axis=1)
                - torch.mean(torch.abs(recover_spec_not - mu), axis=1)
            )
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.8f}".format(test_loss))
    return test_loss


def test_diffusion(vae, time_steps, test_loader, model, diff_params, device, T_pred=50):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _, label, _, _) in enumerate(test_loader):
            data = data.to(device)
            mu, _ = vae.encode(data)
            label = label.long().to(device)
            t = T_pred * torch.ones(mu.shape[0], dtype=torch.int64).to(
                device
            )  # torch.randint(0, 300, (1,)).to(device)#the whole batch uses the same diffusion step
            x_noisy, _ = forward_diffusion_sample(
                mu,
                t,
                diff_params["sqrt_alphas_cumprod"],
                diff_params["sqrt_one_minus_alphas_cumprod"],
                device,
            )
            recover_spec = reverse_diff(model, x_noisy, label, time_steps, t[0], device)
            label_not = (~label.bool()).long()
            recover_spec_not = reverse_diff(
                model, x_noisy, label_not, time_steps, t[0], device
            )

            loss = torch.mean(
                torch.mean(torch.abs(recover_spec - mu), axis=1)
                - torch.mean(torch.abs(recover_spec_not - mu), axis=1)
            )
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))
    return test_loss
