import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset 
from itertools import count
from tqdm import trange

# def weights_init(net, init_type='normal', init_gain=0.02):
#     def init_func(m):
#         classname = m.__class__.__name__
#         if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#             if init_type == 'normal':
#                 init.normal_(m.weight.data, 0.0, init_gain)
#             elif init_type == 'xavier':
#                 init.xavier_normal_(m.weight.data, gain=init_gain)
#             elif init_type == 'kaiming':
#                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#             elif init_type == 'orthogonal':
#                 init.orthogonal_(m.weight.data, gain=init_gain)
#             else:
#                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#             if hasattr(m, 'bias') and m.bias is not None:
#                 init.constant_(m.bias.data, 0.0)
#         elif classname.find('BatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
#             init.normal_(m.weight.data, 1.0, init_gain)
#             init.constant_(m.bias.data, 0.0)
#     net.apply(init_func)

class ParameterEncoder(nn.Module):

    def __init__(self, Ny, Nx, Nchannel, latent_dim, 
                 conv_base_channel, activation, norm, device, dtype):
        super(ParameterEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.conv_base_channel = conv_base_channel
        self.activation = activation
        self.norm = norm

        self.conv1_conv = nn.Conv2d(Nchannel, self.conv_base_channel, 
                                    kernel_size=3, padding=1, 
                                    device=device, dtype=dtype)
        self.conv1_bn   = self.norm(self.conv_base_channel, device=device)
        self.conv1_act  = self.activation
        self.conv1_pool = nn.AvgPool2d(kernel_size=2)
        
        self.conv2_conv = nn.Conv2d(self.conv_base_channel, self.conv_base_channel*2, 
                                    kernel_size=3, padding=1, 
                                    device=device, dtype=dtype)
        self.conv2_bn   = self.norm(self.conv_base_channel*2, device=device)
        self.conv2_act  = self.activation
        self.conv2_pool = nn.AvgPool2d(kernel_size=2)
        
        self.conv3_conv = nn.Conv2d(self.conv_base_channel*2, self.conv_base_channel*4, 
                                    kernel_size=3, padding=1, 
                                    device=device, dtype=dtype)
        self.conv3_bn   = self.norm(self.conv_base_channel*4, device=device)
        self.conv3_act  = self.activation
        
        self.flat_size = (Ny // 4) * (Nx // 4) * self.conv_base_channel*4  
        self.fc_mu      = nn.Linear(self.flat_size, latent_dim, 
                                    device=device, dtype=dtype)
        self.fc_log_var = nn.Linear(self.flat_size, latent_dim, 
                                    device=device, dtype=dtype)

    def forward(self, x):

        x = self.conv1_conv(x)
        x = self.conv1_bn(x)
        x = self.conv1_act(x)
        x = self.conv1_pool(x)
        
        x = self.conv2_conv(x)
        x = self.conv2_bn(x)
        x = self.conv2_act(x)
        x = self.conv2_pool(x)
        
        x = self.conv3_conv(x)
        x = self.conv3_bn(x)
        x = self.conv3_act(x)
        
        x = x.view(x.size(0), -1)
        z_mu = self.fc_mu(x)
        z_log_var = self.fc_log_var(x)
        
        return z_mu, z_log_var

class ParameterDecoder(nn.Module):

    def __init__(self, Ny, Nx, Nchannel, latent_dim, 
                 conv_base_channel, activation, norm, device, dtype):
        super(ParameterDecoder, self).__init__()
        self.Ny, self.Nx = Ny, Nx
        self.conv_base_channel = conv_base_channel
        self.activation = activation
        self.norm = norm
        self.flat_size = (Ny // 4) * (Nx // 4) * self.conv_base_channel*4

        self.fc = nn.Linear(latent_dim, self.flat_size, 
                            device=device, dtype=dtype)
        
        self.deconv1_deconv = nn.ConvTranspose2d(self.conv_base_channel*4, self.conv_base_channel*2, 
                                                 kernel_size=3, stride=1, padding=1, 
                                                 device=device, dtype=dtype)
        self.deconv1_bn     = self.norm(self.conv_base_channel*2, device=device)
        self.deconv1_act    = self.activation
        
        self.deconv2_deconv = nn.ConvTranspose2d(self.conv_base_channel*2, self.conv_base_channel, 
                                                 kernel_size=3, stride=2, padding=1, 
                                                 output_padding=1, device=device, dtype=dtype)
        self.deconv2_bn     = self.norm(self.conv_base_channel, device=device)
        self.deconv2_act    = self.activation

        self.deconv3_deconv = nn.ConvTranspose2d(self.conv_base_channel, Nchannel, 
                                                 kernel_size=3, stride=2, padding=1, 
                                                 output_padding=1, device=device, dtype=dtype)

    def forward(self, z):

        z = self.fc(z)
        z = z.view(z.size(0), self.conv_base_channel*4, self.Ny // 4, self.Nx // 4)
        
        z = self.deconv1_deconv(z)
        z = self.deconv1_bn(z)
        z = self.deconv1_act(z)
        
        z = self.deconv2_deconv(z)
        z = self.deconv2_bn(z)
        z = self.deconv2_act(z)

        z = self.deconv3_deconv(z)
        
        return z

class ParameterVAE(nn.Module):
    def __init__(self, Ny, Nx, Nchannel, 
                 latent_dim, gamma, lr, mask,
                 device, 
                 conv_base_channel = 32, 
                 activation = 'tanh', 
                 norm = None, 
                 dtype = torch.float32):
        
        super(ParameterVAE, self).__init__()
        self.latent_dim = latent_dim
        self.Ny = Ny
        self.Nx = Nx
        self.Nchannel = Nchannel
        self.conv_base_channel = conv_base_channel

        if norm == 'bn':
            self.norm =  nn.BatchNorm2d
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d
        # elif norm == 'ln':
        #     # For LayerNorm the normalized shape must be specified.
        #     # In a conv3d output with shape (B, C, D, H, W), one common choice is to normalize
        #     # over (C, D, H, W). This requires knowing the spatial dimensions.
        #     # If spatial_dims is provided as a tuple (D, H, W):
        #     if spatial_dims is None:
        #         raise ValueError("For LayerNorm you must provide spatial_dims, e.g. (D, H, W)")
        #     normalized_shape = (num_features, ) + spatial_dims
        #     return nn.LayerNorm(normalized_shape, device=device, dtype=dtype)
        elif norm == 'gn':
            # For GroupNorm you need to choose a number of groups.
            # A common default is 8 groups (but it must divide num_features)
            self.norm = lambda num_features: nn.GroupNorm(8, num_features)
        elif norm == 'none':
            self.norm = nn.Identity()
        else:
            raise ValueError("Unsupported normalization: {}".format(norm))

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.encoder = ParameterEncoder(Ny, Nx, Nchannel, latent_dim, 
                                        self.conv_base_channel, self.activation, self.norm, device, dtype)
        self.decoder = ParameterDecoder(Ny, Nx, Nchannel, latent_dim, 
                                        self.conv_base_channel, self.activation, self.norm, device, dtype)
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
    

class StateEncoder(nn.Module):
    
    def __init__(self, Ny, Nx, Nt, Nchannel, latent_dim, 
                 conv_base_channel, activation, norm,
                 device, dtype):
        
        super(StateEncoder, self).__init__()
        self.latent_dim = latent_dim
        # inputshape = (Nchannel, Nt, Ny, Nx)  # PyTorch uses (C, D, H, W) format
        self.conv_base_channel = conv_base_channel
        self.activation = activation
        self.norm = norm

        self.conv1 = nn.Conv3d(Nchannel, self.conv_base_channel, 
                               kernel_size=3, padding=1, device = device, dtype = dtype)
        self.bn1   = self.norm(self.conv_base_channel, device=device)
        self.act1 = self.activation
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        
        self.conv2 = nn.Conv3d(self.conv_base_channel, self.conv_base_channel*2, 
                               kernel_size=3, padding=1, device = device, dtype = dtype)
        self.bn2   = self.norm(self.conv_base_channel*2, device=device)
        self.act2 = self.activation
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        
        self.conv3 = nn.Conv3d(self.conv_base_channel*2, self.conv_base_channel*4, 
                               kernel_size=3, padding=1, device = device, dtype = dtype)
        self.bn3   = self.norm(self.conv_base_channel*4, device=device)
        self.act3 = self.activation
        
        self.flat_size = (Ny // 4) * (Nx // 4) * (Nt // 4) * self.conv_base_channel*4
        self.fc_mu = nn.Linear(self.flat_size, self.latent_dim, 
                               device = device, dtype = dtype)
        self.fc_log_var = nn.Linear(self.flat_size, self.latent_dim, 
                                    device = device, dtype = dtype)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.act3(x)
        x = x.view(x.size(0), -1) 
        
        z_mu = self.fc_mu(x)
        z_log_var = self.fc_log_var(x)
        
        return z_mu, z_log_var

class StateDecoder(nn.Module):

    def __init__(self, Ny, Nx, Nt, Nchannel, latent_dim, 
                 conv_base_channel, activation, norm,
                 device, dtype):
        # Decoder using transpose conv
        super(StateDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.conv_base_channel = conv_base_channel
        self.activation = activation
        self.norm = norm

        self.flat_size = (Ny // 4) * (Nx // 4) * (Nt // 4) * self.conv_base_channel*4
        self.fc1 = nn.Linear(self.latent_dim, self.flat_size, device=device, dtype=dtype)
        self.act_fc1 = self.activation
        
        self.deconv1 = nn.ConvTranspose3d(self.conv_base_channel*4, self.conv_base_channel*2, 
                                          kernel_size=4, stride=2, padding=1, device=device, dtype=dtype)
        self.bn1 = self.norm(conv_base_channel*2, device=device)
        self.act1 = self.activation
        
        self.deconv2 = nn.ConvTranspose3d(self.conv_base_channel*2, self.conv_base_channel, 
                                          kernel_size=4, stride=2, padding=1, device=device, dtype=dtype)
        self.bn2 = self.norm(conv_base_channel, device=device)
        self.act2 = self.activation
        
        self.deconv3 = nn.ConvTranspose3d(self.conv_base_channel, Nchannel, 
                                          kernel_size=3, stride=1, padding=1, device=device, dtype=dtype)

        self.Nt = Nt
        self.Nx = Nx
        self.Ny = Ny

    def forward(self, z):
        z = self.fc1(z)
        z = self.act_fc1(z)
        z = z.view(-1, self.conv_base_channel*4, self.Nt // 4, self.Ny // 4, self.Nx // 4)
        
        z = self.deconv1(z)
        #z = self.bn1(z)
        z = self.act1(z)
        
        z = self.deconv2(z)
        #z = self.bn2(z)
        z = self.act2(z)
        
        x_recon = self.deconv3(z)

        return x_recon

class StateVAE(nn.Module):
    def __init__(self, Ny, Nx, Nt, Nchannel, latent_dim, gamma, lr, mask_t,
                 device, 
                 conv_base_channel = 32, 
                 activation = 'tanh', 
                 norm = None, 
                 dtype = torch.float32):
        
        super(StateVAE, self).__init__()
        self.latent_dim = latent_dim
        self.Ny = Ny
        self.Nx = Nx
        self.Nt = Nt
        self.Nchannel = Nchannel
        self.conv_base_channel = conv_base_channel

        if norm == 'bn':
            self.norm =  nn.BatchNorm3d
        elif norm == 'in':
            self.norm = nn.InstanceNorm3d
        # elif norm == 'ln':
        #     # For LayerNorm the normalized shape must be specified.
        #     # In a conv3d output with shape (B, C, D, H, W), one common choice is to normalize
        #     # over (C, D, H, W). This requires knowing the spatial dimensions.
        #     # If spatial_dims is provided as a tuple (D, H, W):
        #     if spatial_dims is None:
        #         raise ValueError("For LayerNorm you must provide spatial_dims, e.g. (D, H, W)")
        #     normalized_shape = (num_features, ) + spatial_dims
        #     return nn.LayerNorm(normalized_shape, device=device, dtype=dtype)
        elif norm == 'gn':
            # For GroupNorm you need to choose a number of groups.
            # A common default is 8 groups (but it must divide num_features)
            self.norm = lambda num_features: nn.GroupNorm(8, num_features)
        elif norm == 'none':
            self.norm = nn.Identity()
        else:
            raise ValueError("Unsupported normalization: {}".format(norm))

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.encoder = StateEncoder(Ny, Nx, Nt, Nchannel, latent_dim, 
                                    self.conv_base_channel, self.activation, self.norm, 
                                    device, dtype)
        self.decoder = StateDecoder(Ny, Nx, Nt, Nchannel, latent_dim, 
                                    self.conv_base_channel, self.activation, self.norm, 
                                    device, dtype)
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
        train_loader = DataLoader(TensorDataset(xtrain,), 
                                  batch_size = batch_size, 
                                  shuffle=True, drop_last=True)
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
    