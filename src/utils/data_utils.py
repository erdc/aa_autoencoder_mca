#! /usr/bin/env python

import numpy as np
import scipy



"""
Simple utilities for managing snapshots and
creating training, testing data
"""

def read_data(data, soln_names, **kwargs): 
    """
    Extract snapshot data into a dictionary
    indexed by component keys
    data: datafile loaded from numpy file
    """
    XX = data['x']; YY = data['y']
    Nx = XX.shape[1]; Ny = YY.shape[0]
    Nn = XX.shape[1]*YY.shape[0]
    nodes = np.c_[np.reshape(XX,Nn), np.reshape(YY,Nn)]
    assert Nn == nodes.shape[0]

    try:
        T_end = kwargs['t_end']
    except:
        T_end = data['t'].max()

    snap_start = 0
    snap_end = np.count_nonzero(data['t'][data['t'] <= T_end])

    snap = {}
    for key in soln_names:
        snap[key] = data[key].reshape((Nn,-1))[:,snap_start:]

    times = data['t'][snap_start:]
    
    print('Loaded {0} snapshots of dimension {1} for {2}  spanning times [{3:.3f}, {4:.3f}] mins'.format(
                        snap[soln_names[0]].shape[1],snap[soln_names[0]].shape[0], 
                        soln_names, times[0]/60, times[-1]/60))

    return snap, times, nodes, Nx, Ny, snap_start, snap_end


def data_scaler(u1, u2, umax=None, umin=None):
    """
    Takes two data arrays and 
    normalizes both to (0,1)
    using the same scaling
    """
    if umax == None and umin == None:
        u_max = np.maximum(u1.max(), u2.max())
        u_min = np.minimum(u1.min(), u2.min())
    else:
        u_max = umax; u_min = umin
        
    u1 = (u1 - u_min)/(u_max - u_min)
    u2 = (u2 - u_min)/(u_max - u_min)
    
    return u1, u2, u_max, u_min


def scaler_inverse(u, umax, umin, scaling=False):
    """
    Applies an inverse scaling transform to 
    a scaled data array, using scaling factors
    previously defined
    """
    if scaling:
        out = (umax - umin)*u + umin
    else:
        out = u
    return out


def augment_state(state, param_list, Nt, scaled=True):
    """
    Utility function to augment a given 2D state array
    with the parameter value associated with the 
    dynamics
    Input::
    state = [NT x NS] array, 
            NS is spatial/latent dimension of 
                state array,
            NT is sum of all time snapshots for
            each parameter value. 
    param_list = Dictionary of parameter values
    Nt = Dictionary with # of snapshots for each 
        parameter value
    scaled = Boolean, True if augmented values 
            are scaled.
    Output::
    aug_state = [NT x (NS+1)] array, augmented 
                with parameter values
    """
    aug_state = np.hstack((state, np.zeros((state.shape[0],1)) ))
    ctr = 0
    if len(param_list) == 1:
        p_max = 1.0
    else:
        p_max = np.asarray(param_list).max()
    for indx,val in enumerate(np.asarray(param_list)/p_max):
        aug_state[ctr:ctr+Nt[indx],-1] = val*np.ones(Nt[indx]) 
        ctr+= Nt[indx]
        
    return aug_state


