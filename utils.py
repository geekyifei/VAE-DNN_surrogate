# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:18:46 2024

@author: yifei
"""

import numpy as np
import scipy.linalg as spl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

Nx = 20
Ny = 40
Nt = 24
Ncells_raw = Nx * Ny
Ncells = 706

rl2e = lambda yest, yref : spl.norm(yest - yref, 2) / spl.norm(yref , 2) 
infe = lambda yest, yref : spl.norm(yest - yref, np.inf) 
lpp = lambda h, href, sigma: np.sum( -(h - href)**2/(2*sigma**2) - 1/2*np.log( 2*np.pi) - 2*np.log(sigma))
normalize = lambda data, vmax, vmin: 2*(data - vmin)/ (vmax - vmin) - 1 # scale data to [-1, 1]
back_normalize = lambda data, vmax, vmin: (data + 1)*(vmax - vmin)/2 + vmin # scale data back

def plot_func(recon_mesh, real_mesh, fig_save_name, **kwargs):

    figsize = kwargs.get('figsize', (12, 8))
    dpi = kwargs.get('dpi', 300)
    fontsize = kwargs.get('fontsize', 20)
    #labelsize = kwargs.get('labelsize', 16)
    cmap = kwargs.get('cmap', 'turbo')

    fig,ax = plt.subplots(1, 3, figsize=figsize, dpi = dpi)
    err = real_mesh - recon_mesh
    mask = ~np.isnan(real_mesh)
    vmin = real_mesh[mask].min()
    vmax = real_mesh[mask].max()
    
    pred = ax[0].imshow(recon_mesh, vmin  = vmin, vmax = vmax, cmap = cmap)
    true = ax[1].imshow(real_mesh, vmin  = vmin, vmax = vmax, cmap = cmap)
    diff = ax[2].imshow(err, vmin  = err[mask].min(), vmax = err[mask].max(), cmap = cmap)

    divider1 = make_axes_locatable(ax[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    divider2 = make_axes_locatable(ax[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    divider3 = make_axes_locatable(ax[2])
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(pred, cax=cax1)
    fig.colorbar(true, cax=cax2)
    fig.colorbar(diff, cax=cax3)

    ax[0].set_title('Prediction',
                    fontsize = fontsize, 
                    fontweight ="bold")
    ax[1].set_title('Reference', 
                fontsize = fontsize,
                fontweight ="bold")
    ax[2].set_title('Point Error',
                fontsize = fontsize, 
                 fontweight ="bold")
    fig.tight_layout()
    plt.savefig(fig_save_name)

def plot_field(h_pred_masked, fig_save_name, inactive_mask):
    fig, axs = plt.subplots(4, 6, figsize=(20, 20), dpi = 300)  
    for t in range(0, Nt):
        row = t // 6
        col = t % 6
        h_pred_recon = mesh_recon(h_pred_masked.reshape((Nt, -1))[t,:], inactive_mask)
        vmin = h_pred_recon[inactive_mask].min()
        vmax = h_pred_recon[inactive_mask].max()
        im = axs[row, col].imshow(h_pred_recon, vmin  = vmin, vmax = vmax)
        axs[row, col].set_title(f'Timestep {t+1}')
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])
        fig.colorbar(im, ax=axs[row, col], fraction=0.046, pad=0.04)
    plt.tight_layout()
    if fig_save_name is not None:
        plt.savefig(fig_save_name)

def mesh_recon(a, inactive_mask):
    mesh = np.zeros(Ncells_raw)
    mesh[np.where(inactive_mask.ravel())] = a
    mesh[np.where(~inactive_mask.ravel())] = np.nan
    mesh = mesh.reshape(Ny, Nx)
    return mesh

def plot_loss(train_loss, test_loss, epochs, num_print, save_fig_to):
    t = np.arange(0, epochs, num_print)
    fig = plt.figure(constrained_layout=False, figsize=(4, 4), dpi = 300)
    ax = fig.add_subplot()
    ax.plot(t, train_loss, color='blue', label='Training Loss')
    ax.plot(t, test_loss, color='red', label='Testing Loss')
    ax.set_yscale('log')
    ax.set_ylabel('Loss',  fontsize = 16)
    ax.set_xlabel('Epochs', fontsize = 16)
    ax.legend(loc='upper right', fontsize = 14)
    fig.tight_layout()
    fig.savefig(save_fig_to + 'loss.png')
    
def reparameterize(mean, logvar):
    eps = np.random.normal(0, 1, mean.shape)
    return eps * np.exp(0.5 * logvar) + mean
