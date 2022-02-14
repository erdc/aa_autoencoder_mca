#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
try:
    os.listdir(base_dir)
except:
    base_dir = os.getcwd()
data_dir = os.path.join(base_dir,'../data/')


## Runtime Options
visualize = False
save_data = True


### Create a uniform mesh
x = np.arange(-100,101,1)
y = np.arange(0,501,1)
xx, yy = np.meshgrid(x,y)

def plot_mesh(xx,yy):
    fig = plt.figure(figsize=(3,7))
    plt.scatter(xx, yy)
    segs1 = np.stack((xx,yy), axis=2)
    segs2 = segs1.transpose(1,0,2)
    plt.gca().add_collection(LineCollection(segs1))
    plt.gca().add_collection(LineCollection(segs2))


if visualize:
    plot_mesh(xx,yy)


## Analytic function to generate a Gaussian pulse moving with time
def Gaussian2D(x, y, t, c, **kwargs):

    try:
        x0 = kwargs['x0']
    except:
        x0 = 0
    try:
        y0 = kwargs['y0']
    except:
        y0 = 40

    try:
        amp = kwargs['amp']
    except:
        amp = 1.0

    try:
        sigma_x = kwargs['sigma_x']
    except:
        sigma_x = 8

    try:
        sigma_y = kwargs['sigma_y']
    except:
        sigma_y = 8

    arg = (x - x0)**2/(2*sigma_x**2) + (y - c*t - y0)**2/(2*sigma_y**2)
    out = amp*np.exp(-arg)

    return out

x0 = 0; y0 = 40


## Generate parametric snapshots with varying speeds of propagation
print("********Generating snapshots for variable speeds of propagation*******\n")
pulse1 = {}; time1 = {}
sigma = 8
speed_list = [1, 2, 3, 4, 5, 6, 8] #
for ix,speed in enumerate(speed_list):
    time1[ix] = np.arange((y.max()-2*y0)//speed)
    pulse1[ix] = np.zeros((xx.shape[0],xx.shape[1],time1[ix].size))
    for nt in np.arange(time1[ix].size):
        pulse1[ix][:,:,nt] = Gaussian2D(xx,yy, nt, c=speed,
                                    sigma_x = y0/sigma, sigma_y = y0/sigma)
    print("Generating solutions for Sigma = %.4f, Speed = %d"%(y0/sigma, speed))

    if save_data:
        np.savez_compressed(data_dir+'Gaussian2d_pulse_500x200_c%.2f_sigma%.4f'%(speed,y0/sigma),
                pulse = pulse1[ix], x = xx, y = yy, t = time1[ix], c = speed, sigma = y0/sigma)

if visualize:
    nt=time1[ix][-2]
    fig, ax = plt.subplots(nrows=1,ncols=len(speed_list),figsize=(16,6),constrained_layout=True)
    for ix,speed in enumerate(speed_list):
        ax[ix].imshow(pulse1[ix][:,:,nt],
                   extent=[min(x),max(x),min(y),max(y)],
                   origin="lower")
        ax[ix].set_title('Speed = %d, \nTime = %d'%(speed,nt))
        ax[ix].axis('off')
    plt.suptitle("Snapshots of a linearly advecting 2D pulse at different speeds of propagation",fontsize=20)
    # plt.show()
    fig.savefig(data_dir+'Pulse2D_variable_speed.png',dpi=300,bbox_inches='tight')


print("\n********Generating snapshots with different initial locations*******\n")
## Generate parametric snapshots for varying speeds
## with different starting points
pulse2 = {}
time2 = {}
sigma = 8
param_list = [(2,41), (3,41), (3,42), (4,41), (4,42), (4,43)]
for ix,(speed,y0_var) in enumerate(param_list):
    time2[ix] = np.arange((y.max()-2*y0)//speed)
    pulse2[ix] = np.zeros((xx.shape[0],xx.shape[1],time2[ix].size))
    for nt in np.arange(time2[ix].size):
        pulse2[ix][:,:,nt] = Gaussian2D(xx,yy, nt, c=speed, y0 = y0_var,
                                        sigma_x = y0/sigma, sigma_y = y0/sigma)
    print("Generating solutions for Sigma = %.4f, Speed = %d, (x0,y0) = (0,%d)"%(y0/sigma, speed,y0_var))

    if save_data:
        np.savez_compressed(data_dir+'Gaussian2d_pulse_500x200_c%.2f_sigma%.4f_y0%d'%(speed,y0/sigma,y0_var),
                            pulse = pulse2[ix], x = xx, y = yy, t = time2[ix], c = speed, sigma = y0/sigma,
                           y0 = y0_var)


print("\n********Generating snapshots for variable pulse sizes*******\n")
## Generate parametric snapshots for pulse
## of varying sizes
speed = 1
pulse3 = {}; time3 = {}
sigma_list = [2, 4, 8, 5, 2.5]  #
for ix,sigma in enumerate(sigma_list):
    time3[ix] = np.arange(y.max()-2*y0)
    pulse3[ix] = np.zeros((xx.shape[0],xx.shape[1],time3[ix].size))
    for nt in np.arange(time3[ix].size):
        pulse3[ix][:,:,nt] = Gaussian2D(xx, yy, nt, c=speed, sigma_x = y0/sigma,
                                        sigma_y = y0/sigma)
    print("Generating solutions for Sigma = %.4f, Speed = %d"%(y0/sigma, speed))

    if save_data:
        np.savez_compressed(data_dir+'Gaussian2d_pulse_500x200_c%.2f_sigma%.4f'%(speed,y0/sigma),
                            pulse = pulse3[ix], x = xx, y = yy, t = time3[ix], c = speed, sigma = y0/sigma)

if visualize:
    nt=449
    fig, ax = plt.subplots(nrows=1,ncols=len(sigma_list),figsize=(11,5),constrained_layout=True)
    for ix,sigma in enumerate(sigma_list):
        ax[ix].imshow(pulse3[ix][:,:,nt],
                   extent=[min(x),max(x),min(y),max(y)],
                   origin="lower")
        ax[ix].set_title('Sigma = %d, \nTime = %d'%(y0/sigma,nt))
        ax[ix].axis('off')
    plt.suptitle("Snapshots of a linearly advecting 2D pulse with different pulse sizes",fontsize=20)
    # plt.show()
    fig.savefig(data_dir+'Pulse2D_variable_size.png',dpi=300,bbox_inches='tight')


print("\n********Generating shifted snapshots for parametric variations*******\n")
## Generate parametric "shifted" snapshots for pulse
## of varying sizes
vel = 1
pulse4 = {}
shift_param_list = [(1,2), (1,4), (1,8), (1,5), (1,2.5), (2,8), (3,8), (4,8), (5,8), (6,8), (8,8)]
for ix,(speed,sigma) in enumerate(shift_param_list):
    print('Generating shifted solutions for Sigma=%.4f, Speed = %d'%(y0/sigma, speed))
    pulse4[ix] = Gaussian2D(xx, yy, 240/speed, c=speed, sigma_x = y0/sigma,
                            sigma_y = y0/sigma)
    if save_data:
        np.savez_compressed(data_dir+'Shift_Gaussian2d_pulse_500x200_c%.2f_sigma%.4f'%(speed, y0/sigma),
                            pulse = pulse4[ix], x = xx, y = yy, t = 240/speed, c = speed, sigma = y0/sigma)

if visualize:
    nt=240/vel
    fig, ax = plt.subplots(nrows=1,ncols=len(shift_param_list),figsize=(21,6))
    fig.tight_layout()
    fig.subplots_adjust(top=1.00)
    for ix,(speed,sigma) in enumerate(shift_param_list):
        ax[ix].imshow(pulse4[ix],
                   extent=[min(x),max(x),min(y),max(y)],
                   origin="lower")
        ax[ix].set_title('Sigma = %d, \nSpeed = %d, \nTime = %d'%(y0/sigma,speed,nt))
        ax[ix].axis('off')
    plt.suptitle("Shifted snapshots of a linearly advecting 2D pulse with different parametric variations",fontsize=20,y=0.96)
    plt.show()

    
    
