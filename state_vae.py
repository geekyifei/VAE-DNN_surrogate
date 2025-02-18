import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset 
from itertools import count
from tqdm import trange

class Encoder(nn.Module):
    
    def __init__(self, Ny, Nx, Nt, Nchannel, latent_dim, device, dtype):
        
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        #self.inputshape = (Nchannel, Nt, Ny, Nx)  # PyTorch uses (C, D, H, W) format

        self.conv1 = nn.Conv3d(Nchannel, 16, kernel_size=3, padding=1, device = device, dtype = dtype)
        self.bn1   = nn.BatchNorm3d(16, device=device)
        #self.act1 = nn.Tanh()
        self.act1 = nn.LeakyReLU(0.2)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        #self.pool1 = nn.AvgPool3d(kernel_size=2)
        
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1, device = device, dtype = dtype)
        self.bn2   = nn.BatchNorm3d(32, device=device)
        #self.act2 = nn.Tanh()
        self.act2 = nn.LeakyReLU(0.2)
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        #self.pool2 = nn.AvgPool3d(kernel_size=2)
        
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1, device = device, dtype = dtype)
        self.bn3   = nn.BatchNorm3d(64, device=device)
        #self.act3 = nn.Tanh()
        self.act3 = nn.LeakyReLU(0.2)
        
        self.flat_size = (Ny // 4) * (Nx // 4) * (Nt // 4) * 64
        self.fc_mu = nn.Linear(self.flat_size, self.latent_dim, device = device, dtype = dtype)
        self.fc_log_var = nn.Linear(self.flat_size, self.latent_dim, device = device, dtype = dtype)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.act3(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        z_mu = self.fc_mu(x)
        z_log_var = self.fc_log_var(x)
        
        return z_mu, z_log_var

class Decoder(nn.Module):
    def __init__(self, Ny, Nx, Nt, Nchannel, latent_dim, device, dtype):
        
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        #self.inputshape = (Nchannel, Nt, Ny, Nx)  # PyTorch uses (C, D, H, W) format
        
        self.flat_size = (Ny // 4) * (Nx // 4) * (Nt // 4) * 64
        self.fc1 = nn.Linear(self.latent_dim, self.flat_size, device = device, dtype = dtype)
        #self.act_fc1 = nn.Tanh()
        self.act_fc1 = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv3d(64, 64, kernel_size=3, padding=1, device = device, dtype = dtype)
        self.bn1   = nn.BatchNorm3d(64, device=device)
        #self.act1 = nn.Tanh()
        self.act1 =  nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv3d(64, 32, kernel_size=3, padding=1, device = device, dtype = dtype)
        self.bn2   = nn.BatchNorm3d(32, device=device)
        #self.act2 = nn.Tanh()
        self.act2 =  nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv3d(32, 16, kernel_size=3, padding=1, device = device, dtype = dtype)
        self.bn3   = nn.BatchNorm3d(16, device=device)
        #self.act3 = nn.Tanh()
        self.act3 =  nn.LeakyReLU(0.2)
        self.conv_out = nn.Conv3d(16, Nchannel, kernel_size=3, padding=1, device = device, dtype = dtype)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.Nt = Nt
        self.Nx = Nx
        self.Ny = Ny

    def forward(self, z):
        
        z = self.fc1(z)
        z = self.act_fc1(z)
        z = z.view(-1, 64, self.Nt//4, self.Ny//4, self.Nx//4)
        z = self.up(z)
        z = self.conv1(z)
        z = self.bn1(z)
        z = self.act1(z)
        z = self.up(z)
        z = self.conv2(z)
        z = self.bn2(z)
        z = self.act2(z)
        z = self.conv3(z)
        z = self.bn3(z)
        z = self.act3(z)
        x_recon = self.conv_out(z)
        
        return x_recon


class VAE(nn.Module):
    def __init__(self, Ny, Nx, Nt, Nchannel, latent_dim, gamma, lr, mask_t, device, dtype):
        
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.Ny = Ny
        self.Nx = Nx
        self.Nt = Nt
        self.Nchannel = Nchannel
        self.encoder = Encoder(Ny, Nx, Nt, Nchannel, latent_dim, device, dtype)
        self.decoder = Decoder(Ny, Nx, Nt, Nchannel, latent_dim, device, dtype)
        self.gamma = gamma
        self.mask_t = mask_t[np.newaxis, ...]
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
        x_recon_masked = x_recon[np.tile(self.mask_t, (batch_size, 1, 1, 1, 1))].view(batch_size, -1)
        x_masked = x[np.tile(self.mask_t, (batch_size, 1, 1, 1, 1))].view(batch_size, -1)
        return torch.sum(-torch.square(x_recon_masked - x_masked), dim=1)

    def log_normal_pdf(self, sample, mean, logvar):
        
        log2pi = np.log(2. * np.pi)
        return torch.sum(-0.5 * ((sample - mean) ** 2 * torch.exp(-logvar) + logvar + log2pi), dim=1)

    # def loss_elbo(self, x, dip = False):
        
    #     mean_z, logvar_z = self.encoder(x)
    #     z = self.reparameterize(mean_z, logvar_z)
    #     x_recon = self.decoder(z)
    #     recon_loss = self.mask_recon_loss(x_recon, x)
    #     kl_regu = self.gamma * (0.5 * torch.sum(-1 - logvar_z + mean_z.pow(2) + logvar_z.exp(), dim=1))
        
    #     if dip == 'dip1':
    #         cov_matrix = torch.cov(mean_z.T) 
    #         #exp_cov = torch.diag(torch.mean(torch.exp(logvar_z), axis=0))
    #         diff_matrix = cov_matrix - torch.eye(cov_matrix.shape[0], device= cov_matrix.device)  
    #         alpha_loss = self.alpha * torch.sum(torch.square(diff_matrix)) 
    #     else:
    #         alpha_loss = 0
    #     return -torch.mean(recon_loss - kl_regu - alpha_loss)

    def loss_elbo(self, x, dip = False):
    
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
    

# Decoder using Conv3DTranspose to upsample

# class Decoder(nn.Module):
#     def __init__(self, Ny, Nx, Nt, Nchannel, latent_dim, device, dtype):
#         super(Decoder, self).__init__()
#         self.latent_dim = latent_dim
        
#         # Compute flattened feature size (assuming two-fold downsampling in each dimension)
#         self.flat_size = (Ny // 4) * (Nx // 4) * (Nt // 4) * 64
#         self.fc1 = nn.Linear(latent_dim, self.flat_size, device=device, dtype=dtype)
#         self.act_fc1 = nn.Tanh()
        
#         # First transposed convolution upsamples by a factor of 2:
#         self.deconv1 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2, device=device, dtype=dtype)
#         self.bn1 = nn.BatchNorm3d(64, device=device)
#         self.act1 = nn.Tanh()
        
#         # Second transposed convolution upsamples by another factor of 2:
#         self.deconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, device=device, dtype=dtype)
#         self.bn2 = nn.BatchNorm3d(32, device=device)
#         self.act2 = nn.Tanh()
        
#         # An additional convolution block to refine features:
#         self.conv3 = nn.Conv3d(32, 16, kernel_size=3, padding=1, device=device, dtype=dtype)
#         self.bn3 = nn.BatchNorm3d(16, device=device)
#         self.act3 = nn.Tanh()
        
#         # Final convolution to produce the output reconstruction
#         self.conv_out = nn.Conv3d(16, Nchannel, kernel_size=3, padding=1, device=device, dtype=dtype)
        
#         # Save spatial dimensions for reshaping
#         self.Nt = Nt
#         self.Ny = Ny
#         self.Nx = Nx

#     def forward(self, z):
#         # Expand the latent vector to the flattened feature map and reshape
#         z = self.fc1(z)
#         z = self.act_fc1(z)
#         # Note: here we use the ordering: depth = Nt, height = Ny, width = Nx.
#         # The encoder computed flat_size as (Ny//4)*(Nx//4)*(Nt//4)*64 so we reshape accordingly.
#         z = z.view(-1, 64, self.Nt // 4, self.Ny // 4, self.Nx // 4)
        
#         # First upsampling: from (Nt//4, Ny//4, Nx//4) to (Nt//2, Ny//2, Nx//2)
#         z = self.deconv1(z)
#         z = self.bn1(z)
#         z = self.act1(z)
        
#         # Second upsampling: from (Nt//2, Ny//2, Nx//2) to (Nt, Ny, Nx)
#         z = self.deconv2(z)
#         z = self.bn2(z)
#         z = self.act2(z)
        
#         # Further refine the features
#         z = self.conv3(z)
#         z = self.bn3(z)
#         z = self.act3(z)
        
#         # Produce the final reconstruction (the output shape will be [batch, Nchannel, Nt, Ny, Nx])
#         x_recon = self.conv_out(z)
        
#         return x_recon
