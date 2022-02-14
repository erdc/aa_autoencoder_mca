#! /usr/bin/env python

import numpy as np
import scipy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, ScalarFormatter, FormatStrFormatter

from matplotlib import animation
matplotlib.rc('animation', html='html5')
from IPython.display import display
import matplotlib.ticker as ticker
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText


# Plot parameters
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 20,
                     'lines.linewidth': 2,
                     'lines.markersize':10,
                     'axes.labelsize': 16, 
                     'axes.titlesize': 20,
                     'xtick.labelsize': 16,
                     'ytick.labelsize': 16,
                     'legend.fontsize': 16,
                     'axes.linewidth': 2})

import itertools
colors = itertools.cycle(['r','g','b','m','y','c'])
markers = itertools.cycle(['p','d','o','^','s','x','D','H','v','*'])


def set_label(key):
    if key == 'pulse':
        ky = 'u'
    elif key == 'S_dep':
        ky = 'h'
    elif key == 'S_vx':
        ky = 'v_x'
    elif key == 'S_vy':
        ky == 'v_y'
    
    return ky


def compare_soln(uh,utrue,iplot,times_pred,times_true,Nx,Ny,key,**kwargs):
    """
    Plot the true and predicted solution fields
    """
    tn_index = np.searchsorted(times_pred,times_true[iplot])
    urom = uh[key][:,tn_index].reshape((Ny,Nx))
    uplot = utrue[key][:,iplot].reshape((Ny,Nx))
    err = urom-uplot
    try:
        flag = kwargs['flag']
    except:
        flag = 'ROM'
   
    ax1 = plt.subplot(1,3,1); ax1.axis('off')
    surf1 = ax1.imshow(urom, extent=[-100,100,0,500], origin="lower")
    ax1.set_title("%s Solution\n $%1.3f<\mathbf{%s}<%1.3f$"%(flag,np.amin(urom),
                                                    set_label(key),
                                                    np.amax(urom)),fontsize=18)
    ax2 = plt.subplot(1,3,2); ax2.axis('off')
    surf2 = ax2.imshow(uplot, extent=[-100,100,0,500], origin="lower")
    ax2.set_title("True Solution\n $%1.3f<\mathbf{%s}<%1.3f$"%(np.amin(uplot),
                                                    set_label(key),
                                                    np.amax(uplot)),fontsize=18)
    plt.colorbar(surf1, ax=[ax1, ax2], shrink=0.8, aspect=40, pad = 0.03, location='right')
    

    
    ax3 = plt.subplot(1,3,3); ax3.axis('off')
    surf3 = ax3.imshow(err, extent=[-100,100,0,500], origin="lower")
    ax3.set_title("$%1.5f$< error$<%1.5f$\n Rel. Err. = $%1.5f$ "%(np.amin(err),
                                         np.amax(err),
                                         np.linalg.norm(err)/np.linalg.norm(uplot)),
                                         fontsize=18)
    plt.colorbar(surf3, shrink=0.8,aspect=40, pad = 0.03)
    
    
    
def plot_training(epochs, loss, vloss, lr):
    """
    Visualize the training trajectory
    """
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(14,4),constrained_layout=True)
    ax[0].semilogy(epochs,loss,label='train_loss',marker='v',markevery=128)
    ax[0].semilogy(epochs,vloss,label='val_loss',marker='o',markevery=156)
    ax[0].set_title('Training and Validation losses')

    ax[1].plot(epochs,lr,label='LR',marker='p',markevery=128)
    ax[1].set_title('Learning rate decay')
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

    for i in range(2):
        ax[i].legend(); 
        ax[i].set_xlabel('Epochs')
        

def plot_rel_err(x_indx, rel_err_shift, rel_err_recon, val_dict, ky):
    """
    Visualize the evolution of the shift and true 
    loss components
    """
    validation_data = val_dict['v_data']
    val_mark = val_dict['v_mark']
    val_skip = val_dict['v_skip']
    show_legend = val_dict['legend']
        
    ax1 = plt.subplot(1,2,1); 
    ax1.scatter(x_indx[:val_mark:val_skip],rel_err_shift[ky][:val_mark:val_skip],label='%s:train'%(set_label(ky)) ) 
    if validation_data:
        ax1.scatter(x_indx[1:val_mark:val_skip],rel_err_shift[ky][1:val_mark:val_skip],label='%s:test'%(set_label(ky)) ) 
    ax1.set_title('Prediction of shifted snapshots',fontsize=18)
    ax1.set_xlabel('Time (mins)')
    if show_legend:
        ax1.legend()

    ax2 = plt.subplot(1,2,2); 
    ax2.scatter(x_indx[:val_mark:val_skip],rel_err_recon[ky][:val_mark:val_skip],label='%s:train'%(set_label(ky)) ) 
    if validation_data:
        ax2.scatter(x_indx[1:val_mark:val_skip],rel_err_recon[ky][1:val_mark:val_skip],label='%s:test'%(set_label(ky)) ) 
    ax2.set_title('Prediction of true snapshots',fontsize=18)
    ax2.set_xlabel('Time (mins)')
    if show_legend:
        ax2.legend()


    