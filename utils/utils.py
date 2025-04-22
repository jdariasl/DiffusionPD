import torch
from torchvision.utils import save_image
import torch.nn.functional as F
import json
from diffusers import DDPMScheduler
from train.sampler_diffusion import Class_DDPMPipeline


def read_config(file_path):
    """
    Reads the JSON configuration file and returns the parsed dictionary.

    Args:
        file_path (str): Path to the JSON configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(file_path, "r") as file:
        config = json.load(file)
    return config


def test_vae(
    vae,
    test_loader,
    device,
    save_path,
    n_samples=10,
    save_reconstructions=False,
    save_latent_space=False,
    save_samples=False,
):
    print(save_path)
    vae.eval()
    with torch.no_grad():
        for i, (data_o,data, _, _, _) in enumerate(test_loader):
            data = data.to(device)
            data_o = data_o.to(device)
            mu, _ = vae.encode(data)
            recon_batch = vae.decode(mu)
            recon_batch = torch.nn.functional.sigmoid(recon_batch)
            if i == 0:
                n = min(data.size(0), n_samples)
                comparison = torch.cat(
                    [data_o[:n], recon_batch.view(data.size(0), 1, 65, 41)[:n]]
                )
                save_image(comparison.cpu(), save_path + "vae_comparisons.png", nrow=n)
                if save_reconstructions:
                    save_image(
                        recon_batch.cpu(), save_path + "vae_reconstructions.png", nrow=n
                    )
                if save_latent_space:
                    save_image(mu.cpu(), save_path + "vae_latent_space.png", nrow=n)
                if save_samples:
                    z = torch.randn(n, vae.latent_dim).to(device)
                    sample = vae.decode(z).cpu()
                    save_image(sample, save_path + "vae_samples.png", nrow=n)
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


def forward_diffusion_sample(
    x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device="cpu"
):
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
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(
        device
    ) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


def set_diffusion_params(T=300):

    betas = linear_beta_schedule(timesteps=T)

    # Pre-calculate different terms for closed form
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    dict_params = {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance,
    }
    return dict_params


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


@torch.no_grad()
def sample_plot_image_scheduler(vae, norm, model, T, latent_dim, device, save_path, n=10, return_image=False):
    vae.eval()
    norm.eval()
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=T,
        beta_schedule="linear",
        beta_start=0.00015,
        beta_end=0.0195,
    )
    pipeline = Class_DDPMPipeline(unet=model, scheduler=noise_scheduler)
    image = pipeline(
        batch_size=n,
        latent_dim=latent_dim,
        generator=torch.Generator(device="cpu").manual_seed(1234),
        device=device,
        num_inference_steps=T,
    )
    #print(image.max(), image.min(), image.mean())
    #print(norm.batch_norm.bias)
    image = ((image-norm.batch_norm.bias)/norm.batch_norm.weight)*torch.sqrt(norm.batch_norm.running_var +norm.batch_norm.eps) + norm.batch_norm.running_mean
    #print(image.max(), image.min(), image.mean())
    img = vae.decode(image)
    for j in range(n):
        img_j = img[j].unsqueeze(0)
        img_j = torch.clamp(img_j, -1.0, 1.0)

        save_image(img_j.cpu(), save_path + "diff_samples_" + str(j) + ".png")
        if return_image and j==n-1:
            return img_j.cpu()         
    return


@torch.no_grad()
def sample_plot_image(vae, model, T, latent_dim, device, save_path, n=10):
    vae.eval()
    # set constant diffusion parameters
    diff_params = set_diffusion_params(T=T)
    betas = diff_params["betas"]
    sqrt_one_minus_alphas_cumprod = diff_params["sqrt_one_minus_alphas_cumprod"]
    sqrt_recip_alphas = diff_params["sqrt_recip_alphas"]
    posterior_variance = diff_params["posterior_variance"]

    num_images = 12
    stepsize = int(T / num_images)

    for j in range(n):
        # Sample noise
        emb_vec = torch.randn((1, latent_dim), device=device)

        # Random class label
        class_label = torch.randint(0, 4, (emb_vec.shape[0],)).to(emb_vec.device)

        for i in range(0, T)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            emb_vec = sample_timestep(
                model,
                emb_vec,
                t,
                class_label,
                betas,
                sqrt_one_minus_alphas_cumprod,
                sqrt_recip_alphas,
                posterior_variance,
            )
            # Edit: This is to maintain the natural range of the distribution
            # img = torch.clamp(img, -1.0, 1.0)
            if i % stepsize == 0:
                img = vae.decode(emb_vec)
                save_image(
                    img.cpu(),
                    save_path + "diff_samples_" + str(j) + ".png",
                    nrow=num_images,
                )

    return


@torch.no_grad()
def reverse_diff(model, img, class_label, T, diff_step, device):

    # set constant diffusion parameters
    diff_params = set_diffusion_params(T=T)
    betas = diff_params["betas"]
    sqrt_one_minus_alphas_cumprod = diff_params["sqrt_one_minus_alphas_cumprod"]
    sqrt_recip_alphas = diff_params["sqrt_recip_alphas"]
    posterior_variance = diff_params["posterior_variance"]

    for i in range(0, diff_step)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(
            model,
            img,
            t,
            class_label,
            betas,
            sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas,
            posterior_variance,
        )
        # Edit: This is to maintain the natural range of the distribution
        # img = torch.clamp(img, -1.0, 1.0)
    return img


@torch.no_grad()
def sample_timestep(
    model,
    x,
    t,
    class_label,
    betas,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
    posterior_variance,
):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, class_label, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def eval_class_pred_diff_scheduler(test_loader, vae, model, time_steps, device, pred_T=50
):
    model.eval()
    
    pred_labels = []
    true_labels = []
    speakers = []
    scores = []
    
    # set constant diffusion parameters
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=time_steps,
        beta_schedule="linear",
        beta_start=0.0015,
        beta_end=0.0195,
    )
    
    pipeline = Class_DDPMPipeline(unet=model, scheduler=noise_scheduler)
    test_loss = 0
    with torch.no_grad():
        for i, (data, _, label, speaker_id, _) in enumerate(test_loader):
            data = data.to(device)
            mu, _ = vae.encode(data)
            mu = mu.to(device)
            #mu = Norm(mu)
            true_labels.append(label)
            speakers.append(speaker_id)
            label = torch.ones(mu.shape[0], dtype=torch.int64).to(device)
            recover_spec = pipeline(
                init_samples=mu,
                class_labels=label,
                num_inference_steps=pred_T,
                generator=torch.Generator(device="cpu").manual_seed(1234),
                device=device
            )

            label_not = (~label.bool()).long().to(device)
            recover_spec_not = pipeline(
                init_samples=mu,
                class_labels=label_not,
                num_inference_steps=pred_T,
                generator=torch.Generator(device="cpu").manual_seed(1234),
                device=device
            )

            pred_score = -1 * (
                torch.mean(torch.abs(recover_spec - mu), axis=1)
                - torch.mean(torch.abs(recover_spec_not - mu), axis=1)
            )
            scores.append(pred_score)
            pred_labels.append((pred_score > 0).long())
    true_labels = torch.cat(true_labels)
    pred_labels = torch.cat(pred_labels)
    speakers = torch.cat(speakers)
    scores = torch.cat(scores)
    n_speakers = torch.unique(speakers)
    pred_labels_speaker = []
    true_labels_speaker = []
    scores_speaker = []
    for i, speaker in enumerate(n_speakers):
        idx = speakers == speaker
        scores_speaker.append(torch.mean(scores[idx]))
        true_labels_speaker.append(true_labels[idx][0])
        pred_labels_speaker.append((scores_speaker[i] > 0).long())
    true_labels_speaker = torch.stack(true_labels_speaker)
    pred_labels_speaker = torch.stack(pred_labels_speaker)

    return true_labels, pred_labels, true_labels_speaker, pred_labels_speaker

@torch.no_grad()
def eval_class_pred_diff(test_loader, vae, norm, model, T, device, pred_T=50):
    vae.eval()
    norm.eval()
    model.eval()

    pred_labels = []
    true_labels = []
    speakers = []
    scores = []
    with torch.no_grad():
        for i, (data, _, label, speaker_id, _) in enumerate(test_loader):

            diff_params = set_diffusion_params(T=T)
            speakers.append(speaker_id)
            data = data.to(device)
            mu, _ = vae.encode(data)
            mu = norm(mu)
            true_labels.append(label)
            label = torch.ones(mu.shape[0], dtype=torch.int64).to(device)
            t = pred_T * torch.ones(mu.shape[0], dtype=torch.int64).to(device)
            x_noisy, _ = forward_diffusion_sample(
                mu,
                t,
                diff_params["sqrt_alphas_cumprod"],
                diff_params["sqrt_one_minus_alphas_cumprod"],
                device,
            )
            recover_spec = reverse_diff(model, x_noisy, label, T, t[0], device)
            label_not = (~label.bool()).long()
            recover_spec_not = reverse_diff(model, x_noisy, label_not, T, t[0], device)

            pred_score = -1 * (
                torch.mean(torch.abs(recover_spec - mu), axis=1)
                - torch.mean(torch.abs(recover_spec_not - mu), axis=1)
            )
            scores.append(pred_score)
            pred_labels.append((pred_score > 0).long())
    true_labels = torch.cat(true_labels)
    pred_labels = torch.cat(pred_labels)
    speakers = torch.cat(speakers)
    scores = torch.cat(scores)
    n_speakers = torch.unique(speakers)
    pred_labels_speaker = []
    true_labels_speaker = []
    scores_speaker = []
    for i, speaker in enumerate(n_speakers):
        idx = speakers == speaker
        scores_speaker.append(torch.mean(scores[idx]))
        true_labels_speaker.append(true_labels[idx][0])
        pred_labels_speaker.append((scores_speaker[i] > 0).long())
    true_labels_speaker = torch.stack(true_labels_speaker)
    pred_labels_speaker = torch.stack(pred_labels_speaker)

    return true_labels, pred_labels, true_labels_speaker, pred_labels_speaker
