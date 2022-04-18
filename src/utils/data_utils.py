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
    XX = data['x'];
    if XX.ndim > 1:
        Nx = XX.shape[1];
    else:
        Nx = XX.shape[0]
    try:
        YY = data['y']; Ny = YY.shape[0]
        Nn = XX.shape[1]*YY.shape[0]
        nodes = np.c_[np.reshape(XX,Nn), np.reshape(YY,Nn)]
    except:
        Ny = 0;
        Nn = Nx
        nodes = np.reshape(XX,Nn)

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

    for indx,val in enumerate(np.asarray(param_list)):
        aug_state[ctr:ctr+Nt[indx],-1] = val*np.ones(Nt[indx])
        ctr+= Nt[indx]

    return aug_state



def compute_pod_multicomponent(S_pod,subtract_mean=True,subtract_initial=False,full_matrices=False):
    """
    Compute standard SVD [Phi,Sigma,W] for all variables stored in dictionary S_til
     where S_til[key] = Phi . Sigma . W is an M[key] by N[key] array
    Input:
    :param: S_pod -- dictionary of snapshots
    :param: subtract_mean -- remove mean or not
    :param: full_matrices -- return Phi and W as (M,M) and (N,N) [True] or (M,min(M,N)) and (min(M,N),N)
    Returns:
    S      : perturbed snapshots if requested, otherwise shallow copy of S_pod
    S_mean : mean of the snapshots
    Phi : left basis vector array
    sigma : singular values
    W   : right basis vectors
    """
    S, S_mean = {},{}
    Phi,sigma,W = {},{},{}

    for key in list(S_pod.keys()):
        if subtract_mean:
            S_mean[key] = np.mean(S_pod[key],1);
            S[key] = S_pod[key].copy();
            S[key]-= np.tile(S_mean[key],(S_pod[key].shape[1],1)).T
            Phi[key],sigma[key],W[key] = np.linalg.svd(S[key][:,1:],full_matrices=full_matrices)

        elif subtract_initial:
            S_mean[key] = S_pod[key][:,0]
            S[key] = S_pod[key].copy()
            S[key]-= np.tile(S_mean[key],(S_pod[key].shape[1],1)).T
            Phi[key],sigma[key],W[key] = np.linalg.svd(S[key][:,:],full_matrices=full_matrices)
        else:
            S_mean[key] = np.mean(S_pod[key],1)
            S[key] = S_pod[key]
            Phi[key],sigma[key],W[key] = np.linalg.svd(S[key][:,:],full_matrices=full_matrices)

    return S,S_mean,Phi,sigma,W


def compute_trunc_basis(D,U,eng_cap = 0.999999,user_rank={}):
    """
    Compute the number of modes and truncated basis to use based on getting 99.9999% of the 'energy'
    Input:
    D -- dictionary of singular values for each system component
    U -- dictionary of left singular basis vector arrays
    eng_cap -- fraction of energy to be captured by truncation
    user_rank -- user-specified rank to over-ride energy truncation (Empty dict means ignore)
    Output:
    nw -- list of number of truncated modes for each component
    U_r -- truncated left basis vector array as a list (indexed in order of dictionary keys in D)
    """

    nw = {}
    for key in list(D.keys()):
        if key in user_rank:
            nw[key] = user_rank[key]
            nw[key] = np.minimum(nw[key], D[key].shape[0]-2)
            print('User specified truncation level = {0} for {1}\n'.format(nw[key],key))
        else:
            nw[key] = 0
            total_energy = (D[key]**2).sum(); assert total_energy > 0.
            energy = 0.
            while energy/total_energy < eng_cap and nw[key] < D[key].shape[0]-2:
                nw[key] += 1
                energy = (D[key][:nw[key]]**2).sum()
            print('{3} truncation level for {4}% = {0}, \sigma_{1} = {2}'.format(nw[key],nw[key]+1,
                                                            D[key][nw[key]+1],key,eng_cap*100) )

    U_r = {}
    for key in list(D.keys()):
        U_r[key] = U[key][:,:nw[key]]

    return nw, U_r
