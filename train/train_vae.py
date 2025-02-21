from models.vae import VAE
from losses.losses import vae_loss
import torch
import torch.nn as nn
import torch.optim as optim



def train_vae(train_loader, test_loader, x_dim, z_dim, epochs, lr, device):
    best_loss_val = 1e10
    model = VAE(x_dim, z_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, _, _, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('====> Epoch: {} Average vae loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))
        loss_val = test_vae(test_loader, model, device)

        if loss_val < best_loss_val:
            best_loss_val = loss_val
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_val,
            }, 'saved_models/vae.pth')
            print('Model saved as vae.pth')
    return model

def test_vae(test_loader, model, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _, _, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += vae_loss(recon_batch, data, mu, logvar).item()
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss