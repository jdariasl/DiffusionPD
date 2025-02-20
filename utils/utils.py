import torch
from torchvision.utils import save_image
import torch.nn.functional as F
def test_vae(
    vae,
    test_loader,
    device,
    save_path,
    n_samples=10,
    save_reconstructions=False,
    save_latent_space=False,
    save_samples = False
):
    print(save_path)
    vae.eval()
    with torch.no_grad():
        for i, (data, _, _, _) in enumerate(test_loader):
            data = data.to(device)
            mu, _ = vae.encode(data)
            recon_batch = vae.decode(mu)
            if i == 0:
                n = min(data.size(0), n_samples)
                comparison = torch.cat([data[:n], recon_batch.view(data.size(0), 1, 65, 41)[:n]])
                save_image(comparison.cpu(), save_path + 'comparisons.png', nrow=n)
                if save_reconstructions:
                    save_image(recon_batch.cpu(), save_path + "reconstructions.png", nrow=n)
                if save_latent_space:
                    save_image(mu.cpu(), save_path + "latent_space.png", nrow=n)
                if save_samples:
                    z = torch.randn(n, vae.latent_dim).to(device)
                    sample = vae.decode(z).cpu()
                    save_image(sample, save_path + "samples.png", nrow=n)
            else:
                break
    return

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape, device="cpu"):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(device)

def forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

def set_diffusion_params(T = 300):
  
    betas = linear_beta_schedule(timesteps=T)

    # Pre-calculate different terms for closed form
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    dict_params = {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance
    }
    return dict_params