import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset 
from itertools import count
from tqdm import trange

class Encoder(nn.Module):
    def __init__(self, Ny, Nx, Nchannel, latent_dim, device, dtype):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        
        self.conv1_conv = nn.Conv2d(Nchannel, 32, kernel_size=3, padding=1, device=device, dtype=dtype)
        self.conv1_bn   = nn.BatchNorm2d(32, device=device)
        self.conv1_act  = nn.Tanh()
        self.conv1_pool = nn.AvgPool2d(kernel_size=2)
        
        self.conv2_conv = nn.Conv2d(32, 64, kernel_size=3, padding=1, device=device, dtype=dtype)
        self.conv2_bn   = nn.BatchNorm2d(64, device=device)
        self.conv2_act  = nn.Tanh()
        self.conv2_pool = nn.AvgPool2d(kernel_size=2)
        
        self.conv3_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1, device=device, dtype=dtype)
        self.conv3_bn   = nn.BatchNorm2d(128, device=device)
        self.conv3_act  = nn.Tanh()
        
        self.flat_size = (Ny // 4) * (Nx // 4) * 128  # after two pooling operations
        self.fc_mu      = nn.Linear(self.flat_size, latent_dim, device=device, dtype=dtype)
        self.fc_log_var = nn.Linear(self.flat_size, latent_dim, device=device, dtype=dtype)

    def forward(self, x):

        x = self.conv1_conv(x)
        x = self.conv1_bn(x)
        x = self.conv1_act(x)
        #x = torch.sin(x)
        x = self.conv1_pool(x)
        
        x = self.conv2_conv(x)
        x = self.conv2_bn(x)
        x = self.conv2_act(x)
        #x = torch.sin(x)
        x = self.conv2_pool(x)
        
        x = self.conv3_conv(x)
        x = self.conv3_bn(x)
        x = self.conv3_act(x)
        #x = torch.sin(x)
        
        x = x.view(x.size(0), -1)
        z_mu = self.fc_mu(x)
        z_log_var = self.fc_log_var(x)
        
        return z_mu, z_log_var

class Decoder(nn.Module):
    def __init__(self, Ny, Nx, Nchannel, latent_dim, device, dtype):
        super(Decoder, self).__init__()
        self.Ny, self.Nx = Ny, Nx
        self.flat_size = (Ny // 4) * (Nx // 4) * 128  

        self.fc = nn.Linear(latent_dim, self.flat_size, device=device, dtype=dtype)
        
        self.deconv1_deconv = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, 
                                                 device=device, dtype=dtype)
        self.deconv1_bn     = nn.BatchNorm2d(64, device=device)
        self.deconv1_act    = nn.Tanh()
        
        self.deconv2_deconv = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, 
                                                  output_padding=1, device=device, dtype=dtype)
        self.deconv2_bn     = nn.BatchNorm2d(32, device=device)
        self.deconv2_act    = nn.Tanh()

        self.deconv3_deconv = nn.ConvTranspose2d(32, Nchannel, kernel_size=3, stride=2, padding=1, 
                                                  output_padding=1, device=device, dtype=dtype)

    def forward(self, z):

        z = self.fc(z)
        z = z.view(z.size(0), 128, self.Ny // 4, self.Nx // 4)
        
        z = self.deconv1_deconv(z)
        z = self.deconv1_bn(z)
        z = self.deconv1_act(z)
        #z = torch.sin(z)
        
        z = self.deconv2_deconv(z)
        z = self.deconv2_bn(z)
        z = self.deconv2_act(z)
        #z = torch.sin(z)

        z = self.deconv3_deconv(z)
        
        return z

class VAE(nn.Module):
    def __init__(self, Ny, Nx, Nchannel, latent_dim, gamma, lr, mask, device, dtype):
        
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.Ny = Ny
        self.Nx = Nx
        self.Nchannel = Nchannel
        self.encoder = Encoder(Ny, Nx, Nchannel, latent_dim, device, dtype)
        self.decoder = Decoder(Ny, Nx, Nchannel, latent_dim, device, dtype)
        self.gamma = gamma
        self.mask = mask[np.newaxis, ...] # add one channel
        self.itercount = count()
        self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-07)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        
    def sample(self, Nsample, eps=None):
        if eps is None:
            eps = torch.randn((Nsample, self.latent_dim))
        return self.decoder(eps)

    def reparameterize(self, mean, logvar):
        eps = torch.randn_like(mean)
        return eps * torch.exp(0.5 * logvar) + mean

    def mask_recon_loss(self, x_recon, x):
        
        batch_size = x_recon.shape[0]
        x_recon_masked = x_recon[np.tile(self.mask, (batch_size, 1, 1, 1))].view(batch_size, -1)
        x_masked = x[np.tile(self.mask, (batch_size, 1, 1, 1))].view(batch_size, -1)
        return torch.sum(-torch.square(x_recon_masked - x_masked), dim=1)

    def log_normal_pdf(self, sample, mean, logvar):
        
        log2pi = np.log(2. * np.pi)
        return torch.sum(-0.5 * ((sample - mean) ** 2 * torch.exp(-logvar) + logvar + log2pi), dim=1)

    def loss_elbo(self, x):
        
        mean_z, logvar_z = self.encoder(x)
        z = self.reparameterize(mean_z, logvar_z)
        x_recon = self.decoder(z)
        recon_loss = self.mask_recon_loss(x_recon, x)
        kl_regu = self.gamma * (0.5 * torch.sum(-1 - logvar_z + mean_z.pow(2) + logvar_z.exp(), dim=1))
        return -torch.mean(recon_loss - kl_regu)

    def train_vae(self, nIter, num_print, batch_size, xtrain, xtest):
        
        train_loss_log = []
        test_loss_log = []
        train_loader = DataLoader(TensorDataset(xtrain,), batch_size = batch_size, shuffle=True, drop_last=True)
        pbar = trange(nIter)
    
        for it in pbar:
            self.train()
            train_loss = 0.
            self.current_count = next(self.itercount)
    
            # Iterate over batches using DataLoader
            for batch_x, in train_loader:
                self.optimizer.zero_grad()
                loss_batch = self.loss_elbo(batch_x)
                loss_batch.backward()
                self.optimizer.step()
                train_loss += loss_batch.item()
            
            #self.scheduler.step()
            train_loss /= len(train_loader)
    
            if it % num_print == 0:
                with torch.inference_mode():
                    self.eval()
                    test_loss = self.loss_elbo(xtest)
                    train_loss_log.append(train_loss)
                    test_loss_log.append(test_loss.item())
                    pbar.set_postfix({'Train Loss': train_loss,
                                      'Test Loss': test_loss.item()})
    
        return np.array(train_loss_log), np.array(test_loss_log)