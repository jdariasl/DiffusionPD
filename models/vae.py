import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, 
                 x_dim,
                 z_dim = 32,
                 hidden_dims_spectrogram=[64, 1024, 64],):
        super(VAE, self).__init__()
        self.x_dim = x_dim
        self.latent_dim = z_dim
        self.hidden_dims_spectrogram = hidden_dims_spectrogram        
        # Encoder
        self.enc_conv1 = nn.Conv2d(
            self.x_dim, 
            self.hidden_dims_spectrogram[0], 
            kernel_size=3, 
            stride=2, 
            padding=1)
        self.enc_conv2 = nn.Conv2d(
            self.hidden_dims_spectrogram[0], 
            self.hidden_dims_spectrogram[1], 
            kernel_size=3, 
            stride=2, 
            padding=1)
        self.enc_conv3 = nn.Conv2d(
            self.hidden_dims_spectrogram[1], 
            self.hidden_dims_spectrogram[2], 
            kernel_size=3, 
            stride=2, 
            padding=1)
        self.fc_mu = nn.Linear(64 * 9 * 6, self.latent_dim)
        self.fc_logvar = nn.Linear(64 * 9 * 6, self.latent_dim)
        
        self.spec_enc = torch.nn.Sequential(
            self.enc_conv1,
            nn.ReLU(),
            self.enc_conv2,
            nn.ReLU(),
            self.enc_conv3,
            nn.ReLU(),
        )


        # Decoder
        self.fc_dec = nn.Linear(self.latent_dim, 64 * 9 * 6)
        self.dec_conv1 = nn.ConvTranspose2d(self.hidden_dims_spectrogram[2], self.hidden_dims_spectrogram[1], kernel_size=3, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(self.hidden_dims_spectrogram[1], self.hidden_dims_spectrogram[0], kernel_size=3, stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(self.hidden_dims_spectrogram[0], self.x_dim, kernel_size=3, stride=2, padding=1)
        self.spec_dec = torch.nn.Sequential(
            self.dec_conv1,
            nn.ReLU(),
            self.dec_conv2,
            nn.ReLU(),
            self.dec_conv3
        )
        
    def encode(self, x):
        x = self.spec_enc(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = F.relu(self.fc_dec(z))
        x = x.view(x.size(0), 64, 9, 6)
        x = self.spec_dec(x)
        #if train:
        #    x = torch.sigmoid(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar