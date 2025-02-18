from models.vae import VAE
from losses.losses import vae_loss
import torch
import torch.nn as nn
import torch.optim as optim



def train_vae(train_loader, x_dim, z_dim, epochs, lr, device):
    
    model = VAE(x_dim, z_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, _, _, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))
    return model