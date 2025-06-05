import torch
import torch.nn.functional as F
from utils.utils import forward_diffusion_sample


def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (Binary Cross Entropy)
    x = torch.clip(x, min=0, max=1)
    recon_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction="sum")
    # recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    return 4 * recon_loss + kl_loss


def vae_loss_da(recon_x, x, mu, logvar, db_logit, db):
    # Reconstruction loss (Binary Cross Entropy)
    x = torch.clip(x, min=0, max=1)
    recon_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction="sum")
    # recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    # Domain Adaptation loss (Cross Entropy)
    db_loss = F.cross_entropy(db_logit, db)
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    return 4 * recon_loss + kl_loss + db_loss


def diff_loss(
    model,
    x_0,
    class_label,
    t,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
    device,
):
    x_noisy, noise = forward_diffusion_sample(
        x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device
    )
    noise_pred = model(x_noisy, class_label, t)
    # return F.l1_loss(noise, noise_pred)
    return F.mse_loss(noise, noise_pred)
