import torch
from models.unet import UNet
from utils.utils import set_diffusion_params
import torch.optim as optim
from losses.losses import diff_loss

def train_diffusion(vae, time_steps, train_loader, test_loader, x_dim, z_dim, epochs, lr, device):
    #best_loss_val = 1e10
    
    #backbone model
    model = UNet(in_channels=x_dim, out_channels=1, num_classes=4, init_features=z_dim).to(device)
    
    #set constant diffusion parameters
    diff_params = set_diffusion_params(T = time_steps)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    vae.eval()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, label, _, _) in enumerate(train_loader):
            data = data.to(device)
            label = label.long().to(device)
            with torch.no_grad():
                mu, _ = vae.encode(data)

            optimizer.zero_grad()
            t = torch.randint(0, time_steps, (data.shape[0],)).to(device)
            loss = diff_loss(model, mu, label, t, diff_params['sqrt_alphas_cumprod'], diff_params['sqrt_one_minus_alphas_cumprod'],device)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('====> Epoch: {} Average diff loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))
        
    return model
