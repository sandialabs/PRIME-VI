import sys, os
import numpy as np
from numpy.linalg import inv
from numpy import diag
from scipy import stats
from Ypred import Ypred
from utils import theta_samples, fGamma
from transforms import phi_hat_to_phi, theta_t

def Ypred_pushforward(phi_hat,params,Ns=30,forecast=True):
    phi = phi_hat_to_phi(phi_hat)
    theta_hat_samples, _ = theta_samples(phi,params,Ns)
    theta_hat_samples_rshp = []
    for i in range(Ns):
        theta_hat_samples_rshp.append([theta_hat_samples[0][i,...],theta_hat_samples[1][i,...]])
    thetas = list(map(theta_t,theta_hat_samples_rshp))
    Y_preds = []
    for i in range(Ns):
        Y_preds.append(Ypred(theta_hat_samples_rshp[i],params,forecast=forecast))
    return np.array(Y_preds), thetas

def fGamma_pushforward(phi_hat,params,Ns=30,forecast=True):
    if forecast:
        x = np.concatenate([params['t'],params['t_forecast']])
    else:
        x = params['t']
    phi = phi_hat_to_phi(phi_hat)
    theta_hat_samples, _ = theta_samples(phi,params,Ns)
    theta_hat_samples_rshp = []
    theta_mean = theta_t([phi_hat[0][...,0],phi_hat[1][...,1]])
    for i in range(Ns):
        theta_hat_samples_rshp.append([theta_hat_samples[0][i,...],theta_hat_samples[1][i,...]])
    thetas = list(map(theta_t,theta_hat_samples_rshp))
    fvals = np.zeros([Ns,params['Nreg'],x.shape[0]])
    fmean = np.zeros([params['Nreg'],x.shape[0]])
    for i in range(Ns):
        for r in range(params['Nreg']):
            t0, N, k, theta_scale = thetas[i][0][r,:]
            f = lambda tau:N*fGamma(tau-t0,k,theta_scale)
            fvals[i,r,:] = f(x)  
    for r in range(params['Nreg']):
        t0, N, k, theta_scale = theta_mean[0][r,:]
        f = lambda tau:N*fGamma(tau-t0,k,theta_scale)
        fmean[r,:] = f(x)
    return fmean, fvals, x

def Ypred_and_fGamma_pf(phi_hat,params,Ns=30,Ns_theta=30,forecast=True):
    phi = phi_hat_to_phi(phi_hat)
    theta_hat_samples, _ = theta_samples(phi,params,Ns_theta)
    theta_hat_samples_rshp = []
    for i in range(Ns_theta):
        theta_hat_samples_rshp.append([theta_hat_samples[0][i,...],theta_hat_samples[1][i,...]])
    thetas = list(map(theta_t,theta_hat_samples_rshp))
    Y_preds = []
    for i in range(Ns):
        Y_preds.append(Ypred(theta_hat_samples_rshp[i],params,forecast=forecast))
    Y_preds = np.array(Y_preds)
    if forecast:
        x = np.concatenate([params['t'],params['t_forecast']])
    else:
        x = params['t']
    fvals = np.zeros([Ns,params['Nreg'],x.shape[0]])
    fmean = np.zeros([params['Nreg'],x.shape[0]])
    for i in range(Ns):
        for r in range(params['Nreg']):
            t0, N, k, theta_scale = thetas[i][0][r,:]
            f = lambda tau:N*fGamma(tau-t0,k,theta_scale)
            fvals[i,r,:] = f(x)  
    return Y_preds, fvals, theta_hat_samples_rshp, thetas


    if forecast:
        tau = np.concatenate([params['t'],params['t_forecast']])
    else:
        tau = params['t']
    fGamma_pf = np.zeros([Ns,params['Nreg'],tau.shape[0]])
    
    def fGamma_wrapper(t0,N,k,theta_scale):
        return N*fGamma(tau-t0,k,theta_scale)
        
    for i in range(Ns):
        for r in range(params['Nreg']):
            t0, N, k, theta_scale = thetas[i][0][r,:]
            fGamma_pf[i,r,:] = fGamma_wrapper(t0,N,k,theta_scale) 

def Ypred_posterior_predictive(phi_hat,params,Ns=30,thetas=None,forecast=True,save_every=None,suffix=''):
    if thetas is None:
        phi = phi_hat_to_phi(phi_hat)
        theta_hat_samples, _ = theta_samples(phi,params,Ns)
        theta_hat_samples_rshp = []
        for i in range(Ns):
            theta_hat_samples_rshp.append([theta_hat_samples[0][i,...],theta_hat_samples[1][i,...]])
        thetas = list(map(theta_t,theta_hat_samples_rshp))
        
        Y_preds = []
        for i in range(Ns):
            Y_preds.append(Ypred(theta_hat_samples_rshp[i],params,forecast=forecast))
            if save_every is not None:
                if i % save_every == 0:
                    np.save(f'Y_preds_pf_{suffix}.npy',np.array(Y_preds))
    else:
        Y_preds = []
        for i in range(Ns):
            Y_preds.append(Ypred(thetas[i],params,forecast=forecast,transform=False))
            if save_every is not None:
                if i % save_every == 0:
                    np.save(f'Y_preds_pf_{suffix}.npy',np.array(Y_preds))
                
    Y_preds_pf = np.array(Y_preds)
    Y_preds_pp = np.zeros_like(Y_preds_pf)

    Nd = Y_preds_pf.shape[-1]
    sigmas = np.zeros_like(Y_preds_pf)
    noises = np.zeros_like(Y_preds_pf)
    for i in range(Ns):
        tauphi,lbdphi,siga,sigm = thetas[i][1]
        for d in range(Nd):
            y_pred = Y_preds_pf[i,:,d]
            Sigma_d = diag((siga + sigm*y_pred)**2)
            sigmas[i,:,d] = diag(Sigma_d)
            noise = stats.multivariate_normal.rvs(cov=Sigma_d)
            Y_preds_pp[i,:,d] = Y_preds_pf[i,:,d] + noise
            noises[i,:,d] = noise
        if save_every is not None:
            if i % save_every == 0:
                np.save(f'Y_preds_pp_{suffix}.npy',Y_preds_pp[:i])
 
    return thetas, Y_preds_pf, Y_preds_pp
    