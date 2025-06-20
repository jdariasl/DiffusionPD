import torch
from torchvision.utils import save_image
import torch.nn.functional as F
import json
from diffusers import DDPMScheduler
from train.sampler_diffusion import Class_DDPMPipeline
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.manifold import TSNE
import numpy as np
import umap
import subprocess
import re

def get_idle_gpu():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    
    try:
        # Execute nvidia-smi to get GPU information
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"], encoding="utf-8")
        memory_free = [int(x.strip()) for x in output.strip().split('\n')]
        
        # Find the GPU with the most free memory
        idle_gpu_index = memory_free.index(max(memory_free))
        return torch.device(f"cuda:{idle_gpu_index}")
    
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return torch.device("cuda:0") # Default to first GPU

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
        for i, (data_o, data, _, _, _) in enumerate(test_loader):
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
def sample_plot_image_scheduler(
    vae, model, T, latent_dim, device, save_path, n=10, return_image=False
):
    vae.eval()
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

    img = vae.decode(image)
    for j in range(n):
        img_j = img[j].unsqueeze(0)
        img_j = torch.clamp(img_j, -1.0, 1.0)

        save_image(img_j.cpu(), save_path + "diff_samples_" + str(j) + ".png")
        if return_image and j == n - 1:
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
def eval_class_pred_diff_scheduler(
    test_loader, vae, model, time_steps, device, pred_T=50
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

            # noise = torch.randn_like(mu).to(device)
            # mu = noise_scheduler.add_noise(
            #    mu,
            #    noise,
            #    pred_T * torch.ones(mu.shape[0], dtype=torch.int64).to(device),
            # )
            # mu = Norm(mu)
            true_labels.append(label)
            speakers.append(speaker_id)
            label = torch.ones(mu.shape[0], dtype=torch.int64).to(device)
            recover_spec = pipeline(
                init_samples=mu,
                class_labels=label,
                num_inference_steps=pred_T,
                generator=torch.Generator(device="cpu").manual_seed(1234),
                device=device,
            )

            label_not = (~label.bool()).long().to(device)
            recover_spec_not = pipeline(
                init_samples=mu,
                class_labels=label_not,
                num_inference_steps=pred_T,
                generator=torch.Generator(device="cpu").manual_seed(1234),
                device=device,
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
    scores_speaker = torch.stack(scores_speaker)

    return (
        true_labels,
        pred_labels,
        scores,
        true_labels_speaker,
        pred_labels_speaker,
        scores_speaker,
    )


@torch.no_grad()
def eval_class_pred_diff(test_loader, vae, model, T, device, pred_T=50):
    vae.eval()
    # norm.eval()
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
            # mu = norm(mu)
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
    scores_speaker = torch.stack(scores_speaker)

    return (
        true_labels,
        pred_labels,
        scores,
        true_labels_speaker,
        pred_labels_speaker,
        scores_speaker,
    )


def plot_kde_and_roc(
    true_labels,
    pred_labels,
    true_labels_speaker,
    pred_labels_speaker,
    filename="performance_plot.png",
):
    sns.set_theme(style="whitegrid")

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Instance-level KDE and histogram
    ax1 = axes[0, 0]
    pos_scores = [p for t, p in zip(true_labels, pred_labels) if t == 1]
    neg_scores = [p for t, p in zip(true_labels, pred_labels) if t == 0]

    sns.histplot(
        pos_scores,
        kde=True,
        stat="density",
        bins=30,
        color="blue",
        label="Positive",
        ax=ax1,
    )
    sns.histplot(
        neg_scores,
        kde=True,
        stat="density",
        bins=30,
        color="red",
        label="Negative",
        ax=ax1,
    )
    ax1.set_title("Instance-level KDE and Histogram")
    ax1.set_xlabel("Prediction Score")
    ax1.legend()

    # Speaker-level KDE and histogram
    ax2 = axes[0, 1]
    pos_scores_speaker = [
        p for t, p in zip(true_labels_speaker, pred_labels_speaker) if t == 1
    ]
    neg_scores_speaker = [
        p for t, p in zip(true_labels_speaker, pred_labels_speaker) if t == 0
    ]

    sns.histplot(
        pos_scores_speaker,
        kde=True,
        stat="density",
        bins=30,
        color="blue",
        label="Positive",
        ax=ax2,
    )
    sns.histplot(
        neg_scores_speaker,
        kde=True,
        stat="density",
        bins=30,
        color="red",
        label="Negative",
        ax=ax2,
    )
    ax2.set_title("Speaker-level KDE and Histogram")
    ax2.set_xlabel("Prediction Score")
    ax2.legend()

    # Instance-level ROC curve
    ax3 = axes[1, 0]
    fpr, tpr, _ = roc_curve(true_labels, pred_labels)
    auc_score = roc_auc_score(true_labels, pred_labels)
    ax3.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {auc_score:.2f})")
    ax3.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax3.set_title("Instance-level ROC Curve")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.legend()

    # Speaker-level ROC curve
    ax4 = axes[1, 1]
    fpr_speaker, tpr_speaker, _ = roc_curve(true_labels_speaker, pred_labels_speaker)
    auc_speaker = roc_auc_score(true_labels_speaker, pred_labels_speaker)
    ax4.plot(
        fpr_speaker,
        tpr_speaker,
        color="blue",
        label=f"ROC curve (AUC = {auc_speaker:.2f})",
    )
    ax4.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax4.set_title("Speaker-level ROC Curve")
    ax4.set_xlabel("False Positive Rate")
    ax4.set_ylabel("True Positive Rate")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def pred_T_effect(test_loader, vae, model, T, device):
    AUC = []
    AUC_speaker = []
    Accuracy = []
    Accuracy_speaker = []
    for pred_T in range(51, 101):
        (
            true_labels,
            pred_labels,
            scores,
            true_labels_speaker,
            pred_labels_speaker,
            scores_speaker,
        ) = eval_class_pred_diff_scheduler(
            test_loader, vae, model, T, device, pred_T=pred_T
        )

        Accuracy.append((true_labels == pred_labels).float().mean())
        Accuracy_speaker.append(
            (true_labels_speaker == pred_labels_speaker).float().mean()
        )
        AUC.append(roc_auc_score(true_labels, scores))
        AUC_speaker.append(roc_auc_score(true_labels_speaker, scores_speaker))
        print(f"Pred_T:{pred_T}, done!")
    return AUC, AUC_speaker, Accuracy, Accuracy_speaker


@torch.no_grad()
def get_bottleneck_embeddings(data_loader, vae, model, time_steps, device, pred_T=50):
    model.eval()

    embeddings = []
    generated_vectors = []
    true_vectors = []
    true_labels = []
    speakers = []
    database_ids = []

    # set constant diffusion parameters
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=time_steps,
        beta_schedule="linear",
        beta_start=0.0015,
        beta_end=0.0195,
    )

    pipeline = Class_DDPMPipeline(unet=model, scheduler=noise_scheduler)
    with torch.no_grad():
        for i, (data, _, label, speaker_id, db_id) in enumerate(data_loader):
            data = data.to(device)
            mu, _ = vae.encode(data)
            mu = mu.to(device)
            database_ids.append(db_id)
            # noise = torch.randn_like(mu).to(device)
            # mu = noise_scheduler.add_noise(
            #    mu,
            #    noise,
            #    pred_T * torch.ones(mu.shape[0], dtype=torch.int64).to(device),
            # )
            # mu = Norm(mu)
            label = label.long().to(device)
            speakers.append(speaker_id.detach().numpy())
            recover_spec, bottleneck = pipeline(
                init_samples=mu,
                class_labels=label,
                num_inference_steps=pred_T,
                generator=torch.Generator(device="cpu").manual_seed(1234),
                return_bottleneck=True,
                device=device,
            )
            true_labels.append(label.detach().numpy())
            true_vectors.append(mu.detach().numpy())
            generated_vectors.append(recover_spec.detach().numpy())
            embeddings.append(bottleneck.detach().numpy())

    true_labels = np.concatenate(true_labels)
    speakers = np.concatenate(speakers)
    embeddings = np.concatenate(embeddings)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)  # Flatten if needed
    true_vectors = np.concatenate(true_vectors)
    generated_vectors = np.concatenate(generated_vectors)
    database_ids = np.concatenate(database_ids)

    return (
        embeddings,
        true_vectors,
        generated_vectors,
        true_labels,
        speakers,
        database_ids,
    )


def plot_embeddings(
    train_embeddings,
    train_true_labels,
    train_sources,
    test_embeddings,
    test_true_labels,
    test_sources,
    method="umap",
    filename="embedding_plot.png",
):
    """
    Plots UMAP or t-SNE projections of train and test embeddings.
    Each (class, source) pair is assigned a unique color.

    Args:
        train_embeddings (np.ndarray): Training embeddings.
        train_true_labels (np.ndarray): Training labels.
        train_sources (np.ndarray): Dataset sources for training data.
        test_embeddings (np.ndarray): Test embeddings.
        test_true_labels (np.ndarray): Test labels.
        test_sources (np.ndarray): Dataset sources for test data.
        method (str): "umap" or "tsne".
        filename (str): Path to save the figure.
    """

    # Step 1: Project data
    if method.lower() == "umap":
        reducer = umap.UMAP(random_state=42)
        train_proj = reducer.fit_transform(train_embeddings)
        test_proj = reducer.transform(test_embeddings)
    elif method.lower() == "tsne":
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        combined_embeddings = np.vstack((train_embeddings, test_embeddings))
        combined_proj = tsne.fit_transform(combined_embeddings)
        train_proj = combined_proj[:len(train_embeddings)]
        test_proj = combined_proj[len(train_embeddings):]
    else:
        raise ValueError("Invalid method. Choose 'umap' or 'tsne'.")

    # Step 2: Create composite (class, source) labels
    train_combo_labels = np.array([f"{label}_{source}" for label, source in zip(train_true_labels, train_sources)])
    test_combo_labels = np.array([f"{label}_{source}" for label, source in zip(test_true_labels, test_sources)])

    all_combo_labels = np.concatenate([train_combo_labels, test_combo_labels])
    unique_combos = np.unique(all_combo_labels)
    combo_to_index = {combo: idx for idx, combo in enumerate(unique_combos)}
    cmap = get_cmap("tab20", len(unique_combos))
    color_map = [cmap(combo_to_index[label]) for label in all_combo_labels]
    train_colors = color_map[:len(train_proj)]
    test_colors = color_map[len(train_proj):]

    # Step 3: Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].scatter(train_proj[:, 0], train_proj[:, 1], color=train_colors, alpha=0.7)
    axes[0].set_title(f"Train Embeddings ({method.upper()})")
    axes[0].set_xlabel(f"{method.upper()}-1")
    axes[0].set_ylabel(f"{method.upper()}-2")

    axes[1].scatter(test_proj[:, 0], test_proj[:, 1], color=test_colors, alpha=0.7)
    axes[1].set_title(f"Test Embeddings ({method.upper()})")
    axes[1].set_xlabel(f"{method.upper()}-1")
    axes[1].set_ylabel(f"{method.upper()}-2")

    # Legend
    handles = []
    for combo, idx in combo_to_index.items():
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=cmap(idx), label=combo, markersize=8))
    fig.legend(handles=handles, title="Class_Source", loc="upper center", ncol=5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename)
    plt.close()

def linear_probe_classifier(
    train_embeddings,
    train_labels,
    test_embeddings,
    test_labels,
    solver='lbfgs',
    max_iter=1000,
    penalty='l2',
    C=1.0,
):
    """
    Trains a linear classifier (logistic regression) on precomputed embeddings (linear probe).
    Evaluates on the test set and returns a classification report.

    Args:
        train_embeddings (np.ndarray): Training embeddings.
        train_labels (np.ndarray): Ground truth labels for training.
        test_embeddings (np.ndarray): Test embeddings.
        test_labels (np.ndarray): Ground truth labels for testing.
        solver (str): Solver used by LogisticRegression. Default is 'lbfgs'.
        max_iter (int): Maximum iterations for the solver.
        penalty (str): Regularization penalty.
        C (float): Inverse regularization strength.

    Returns:
        report (str): Text classification report.
    """
    # Initialize linear classifier
    clf = LogisticRegression(
        solver=solver,
        max_iter=max_iter,
        penalty=penalty,
        C=C,
        multi_class='auto',
        random_state=42
    )

    # Train on embeddings
    clf.fit(train_embeddings, train_labels)

    # Predict on test embeddings
    test_preds = clf.predict(test_embeddings)

    # Generate classification report
    report = classification_report(test_labels, test_preds)
    print("=== Classification Report ===")
    print(report)

    return report