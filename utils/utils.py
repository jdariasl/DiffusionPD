import torch
from torchvision.utils import save_image

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