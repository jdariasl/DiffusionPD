import torch
from models.unet import UNet
from utils.utils import set_diffusion_params
import torch.optim as optim
from losses.losses import diff_loss
from utils.utils import forward_diffusion_sample, reverse_diff
from torch.optim.lr_scheduler import ExponentialLR
from torch.cuda.amp import GradScaler


def train_diffusion(
    vae,
    time_steps,
    train_loader,
    test_loader,
    x_dim,
    z_dim,
    epochs,
    lr,
    pred_diff_time,
    device,
):
    # best_loss_val = 1e10
    scaler_diffusion = GradScaler()

    # backbone model
    model = UNet(
        in_channels=x_dim, out_channels=1, num_classes=4, init_features=z_dim
    ).to(device)

    # set constant diffusion parameters
    diff_params = set_diffusion_params(T=time_steps)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = ExponentialLR(optimizer, gamma=0.9)

    vae.eval()
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, label, _, _) in enumerate(train_loader):
            data = data.to(device)
            label = label.long().to(device)
            with torch.no_grad():
                mu, _ = vae.encode(data)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda"):
                t = torch.randint(0, time_steps, (data.shape[0],)).to(device)
                loss = diff_loss(
                    model,
                    mu,
                    label,
                    t,
                    diff_params["sqrt_alphas_cumprod"],
                    diff_params["sqrt_one_minus_alphas_cumprod"],
                    device,
                )
            scaler_diffusion.scale(loss).backward()
            scaler_diffusion.step(optimizer)
            scaler_diffusion.update()

            # loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(
            "====> Epoch: {} Average diff loss: {:.4f}".format(
                epoch, train_loss / len(train_loader.dataset)
            )
        )

        test_diffusion(
            vae, time_steps, test_loader, model, diff_params, device, pred_diff_time
        )
        scheduler.step()
    return model


def test_diffusion(vae, time_steps, test_loader, model, diff_params, device, T_pred=50):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, label, _, _) in enumerate(test_loader):
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
