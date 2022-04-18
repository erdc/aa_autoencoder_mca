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
import matplotlib.gridspec as gridspec


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
    elif key == 'burgers':
        ky = 'u'
    elif key == 'ade':
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



def compare_soln_1d(uh,utrue,times_pred,times_true,xx,key,**kwargs):
    """
    Plot the true and predicted solution fields
    """
    try:
        flag = kwargs['flag']
    except:
        flag = 'ROM'

    try:
        tr_end = kwargs['tr_end']
    except:
        tr_end = 0.9

    nt_start = np.maximum(np.searchsorted(times_true, times_pred[0]),50)
    nt_list = [int(x) for x in np.linspace(nt_start, int(times_true.size*tr_end),4)]
    ax1 = plt.subplot(1,2,1);
    for nt in nt_list:
        ax1.plot(xx, utrue[key][:,nt], marker='v',markevery=17,label='TRUE, t=%.2f'%(times_true[nt]))
        nt_plot = np.searchsorted(times_pred,times_true[nt])
        ax1.plot(xx, uh[key][:,nt_plot], marker='o',markevery=23,label='%s, t=%.2f'%(flag,times_pred[nt_plot]))
        ax1.legend(ncol=2,bbox_to_anchor=(0.99,1.42))

    ax2 = plt.subplot(1,2,2);
    for nt in nt_list:
        nt_plot = np.searchsorted(times_pred,times_true[nt])
        err = utrue[key][:,nt] - uh[key][:,nt_plot]
        ax2.plot(xx, err, label='t = %.2f'%(times_true[nt]), marker = next(markers),markevery=23)
        ax2.legend(ncol=2,bbox_to_anchor=(0.9,1.25))



def plot_soln_1d(uh_shift,utrue_shift,uh,utrue,times_pred,times_true,xx,key,**kwargs):
    """
    Plot the true and predicted solution fields together
    """
    try:
        flag = kwargs['flag']
    except:
        flag = 'ROM'

    try:
        tr_end = kwargs['tr_end']
    except:
        tr_end = 0.9

    nt_start = np.maximum(np.searchsorted(times_true, times_pred[0]),50)
    nt_list = [int(x) for x in np.linspace(nt_start, int(times_true.size*tr_end),4)]
    ax1 = plt.subplot(1,2,1);
    for nt in nt_list:
        ax1.plot(xx, utrue_shift[key][:,nt], marker='v',markevery=17,label='TRUE, t=%.2f'%(times_true[nt]))
        nt_plot = np.searchsorted(times_pred,times_true[nt])
        ax1.plot(xx, uh_shift[key][:,nt_plot], marker='o',markevery=23,label='%s, t=%.2f'%(flag,times_pred[nt_plot]))
    ax1.set_xlabel('x',fontsize=18,)

    ax2 = plt.subplot(1,2,2);
    for nt in nt_list:
        ax2.plot(xx, utrue[key][:,nt], marker='v',markevery=17, )
        nt_plot = np.searchsorted(times_pred,times_true[nt])
        ax2.plot(xx, uh[key][:,nt_plot], marker='o',markevery=23,)
        ax2.set_xlabel('x',fontsize=18,)


def plot_decoded_1d(uh,utrue,times_pred,times_true,xx,key,**kwargs):
    """
    Plot the true and predicted solution fields together
    """
    try:
        flag = kwargs['flag']
    except:
        flag = 'ROM'

    try:
        tr_end = kwargs['tr_end']
    except:
        tr_end = 0.9

    nt_start = np.maximum(np.searchsorted(times_true, times_pred[0]),50)
    nt_list = [int(x) for x in np.linspace(nt_start, int(times_true.size*tr_end),4)]

    ax1 = plt.subplot(1,1,1);
    for nt in nt_list:
        ax1.plot(xx, utrue[key][:,nt], marker='v',markevery=17,label='TRUE, t=%.2f'%(times_true[nt]))
        nt_plot = np.searchsorted(times_pred,times_true[nt])
        ax1.plot(xx, uh[key][:,nt_plot], marker='o',markevery=23,label='%s, t=%.2f'%(flag,times_pred[nt_plot]))
    ax1.set_xlabel('x',fontsize=18,)
    ax1.legend(ncol=2,bbox_to_anchor=(1.03, 1.49))


def plot_spcaetime_1d(p1,p2,p4,p5,label1=None, label2=None):
    """
    Plot space-time 2d plots of 1D solutions
    Row1 : Predicted, True, Error for Soln1
    Row2 : Predicted, True, Error for Soln2
    """
    f = plt.figure(figsize=(16,7))
    gs = gridspec.GridSpec(2, 3, )
    gs.update(wspace=0.12, hspace=0.26) # set the spacing between axes.


    p3 = p2-p1
    p6 = p5-p4
    vmin1 = np.amin([p1.min(), p2.min(), p4.min(), p5.min()])
    vmax1 = np.amax([p1.max(), p2.max(), p4.max(), p5.max()])
    vmin2 = np.amin([p3.min(), p6.min()])
    vmax2 = np.amax([p3.max(), p6.max()])

    ax1 = plt.subplot(gs[0, 0]);
    f1= ax1.imshow(p1,cmap='jet',origin='lower',vmin=vmin1, vmax=vmax1, extent=(0,2,0,1), aspect = 0.79)
    ax1.yaxis.set_ticks(np.arange(0,1.1,1))

    ax2 = plt.subplot(gs[0, 1]);
    f2 = ax2.imshow(p2,cmap='jet',origin='lower',vmin=vmin1, vmax=vmax1, extent=(0,2,0,1), aspect = 0.79)
    ax2.set_yticklabels([])

    ax3 = plt.subplot(gs[0, 2]);
    f3 = ax3.imshow(p3,cmap='Spectral',origin='lower',vmin=vmin2,vmax=vmax2,extent=(0,2,0,1), aspect = 0.79)
    ax3.set_yticklabels([])

    ax4 = plt.subplot(gs[1, 0]);
    f4= ax4.imshow(p4,cmap='jet',origin='lower',vmin=vmin1, vmax=vmax1, extent=(0,2,0,1), aspect = 0.79)
    ax4.set_xticklabels([]); ax4.yaxis.set_ticks(np.arange(0,1.1,1))

    ax5 = plt.subplot(gs[1, 1]);
    f5 = ax5.imshow(p5,cmap='jet',origin='lower',vmin=vmin1, vmax=vmax1, extent=(0,2,0,1), aspect = 0.79)
    cbar1 = f.colorbar(f5, ax=list((ax1, ax2, ax4,ax5)),orientation='horizontal',aspect=50, pad=0.1,shrink=0.9)
    ax5.set_xticklabels([]); ax5.set_yticklabels([])

    ax6 = plt.subplot(gs[1, 2]);
    f6 = ax6.imshow(p6,cmap='Spectral',origin='lower',vmin=vmin2,vmax=vmax2,extent=(0,2,0,1), aspect = 0.79)
    cbar2 = f.colorbar(f6, ax=list((ax3,ax6)), orientation='horizontal', aspect=25, pad=0.1,
                       ticks=[vmin2,(vmin2+vmax2)/2,vmax2])
    cbar2.ax.set_xticklabels(['%.4f'%vmin2, '%.4f'%(.5*vmin2+.5*vmax2), '%.4f'%vmax2])
    ax6.set_xticklabels([]); ax6.set_yticklabels([])

    # ax1.set_xlabel('$t$',fontsize=16); ax2.set_xlabel('$t$',fontsize=16); ax3.set_xlabel('$t$',fontsize=16)
    ax1.set_ylabel('$x$',fontsize=18);

    ax4.set_xlabel('$t$',fontsize=18); ax5.set_xlabel('$t$',fontsize=18); ax6.set_xlabel('$t$',fontsize=18)
    ax4.set_ylabel('$x$',fontsize=18);

    # place a text box in upper left in axes coords
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    if label1 is not None:
        ax1.text(-0.22, 0.5, label1, bbox=props, horizontalalignment='center', rotation=90,
             verticalalignment='center', transform=ax1.transAxes)
    if label2 is not None:
        ax4.text(-0.22, 0.5, label2, bbox=props, horizontalalignment='center', rotation=90,
             verticalalignment='center', transform=ax4.transAxes)




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


    
