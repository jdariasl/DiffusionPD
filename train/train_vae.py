from models.vae import VAE, VAE_DA
from losses.losses import vae_loss, vae_loss_da
import torch
import torch.nn as nn
import torch.optim as optim

# from torch.cuda.amp import GradScaler

db_dir = {
    "Gita": 0,
    "Neurovoz": 1,
    "Saarbruecken": 2,
}


def train_vae(
    train_loader,
    test_loader,
    x_dim,
    z_dim,
    epochs,
    lr,
    device,
    resume_training=False,
    vae=None,
):
    best_loss_val = 1e10
    # scaler = GradScaler()
    if resume_training:
        model = vae
    else:
        # Initialize the VAE model
        model = VAE(x_dim, z_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data_o, data, _, _, _) in enumerate(train_loader):
            data = data.to(device)
            data_o = data_o.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data_o, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(
            "====> Epoch: {} Average vae loss: {:.4f}".format(
                epoch, train_loss / len(train_loader.dataset)
            )
        )
        loss_val = test_vae(test_loader, model, device)

        if loss_val < best_loss_val:
            best_loss_val = loss_val
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_val,
                },
                "saved_models/vae.pth",
            )
            print("Model saved as vae.pth")
    return model


def train_vae_da(
    train_loader,
    test_loader,
    x_dim,
    z_dim,
    epochs,
    lr,
    device,
    resume_training=False,
    vae=None,
):
    best_loss_val = 1e10
    # scaler = GradScaler()
    if resume_training:
        model = vae
    else:
        # Initialize the VAE model
        model = VAE_DA(x_dim, z_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data_o, data, _, _, db_group) in enumerate(train_loader):
            data = data.to(device)
            data_o = data_o.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            db_logit = model.db_classifier(mu)
            # Map db_group to dic index
            db = [db_dir[db_group[i]] for i in range(len(db_group))]
            db_group = torch.tensor(db, dtype=torch.long).to(device)
            db = db_group.to(device)
            # Use the domain adaptation loss function
            loss = vae_loss_da(recon_batch, data_o, mu, logvar, db_logit, db)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(
            "====> Epoch: {} Average vae loss: {:.4f}".format(
                epoch, train_loss / len(train_loader.dataset)
            )
        )
        loss_val = test_vae_da(test_loader, model, device)

        if loss_val < best_loss_val:
            best_loss_val = loss_val
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_val,
                },
                "saved_models/vae_da.pth",
            )
            print("Model saved as vae.pth")
    return model


def test_vae(test_loader, model, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data_o, data, _, _, _) in enumerate(test_loader):
            data = data.to(device)
            data_o = data_o.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += vae_loss(recon_batch, data_o, mu, logvar).item()
    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))
    return test_loss

def test_vae_da(test_loader, model, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data_o, data, _, _, db_group) in enumerate(test_loader):
            data = data.to(device)
            data_o = data_o.to(device)
            recon_batch, mu, logvar = model(data)
            db_logit = model.db_classifier(mu)
            # Map db_group to dic index
            db = [db_dir[db_group[i]] for i in range(len(db_group))]
            db_group = torch.tensor(db, dtype=torch.long).to(device)
            db = db_group.to(device)
            test_loss += vae_loss_da(recon_batch, data_o, mu, logvar, db_logit, db).item()
    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))
    return test_loss
