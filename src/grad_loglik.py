import numpy as np
from Ypred import Ypred
from grad_Ypred import dYpred_dtheta
from utils import fGamma, F_LN, qform
from loglik import loglik_gaussian
from scipy.special import gamma, digamma, erf
from numpy.linalg import inv
from numpy import trace as tr, dot, diag, log, sqrt, power, exp
from transforms import theta_t, theta_t_var
import matplotlib.pyplot as plt

def dloglik_dtheta(theta_hat,params,grad_model=True,grad_noise=True):
    """Computes the derivative of the log likelihood with respect to theta_hat.

    :param theta_hat: unconstrained model parameters
    :type theta_hat: list
    :param params: additional model data
    :type params: dict
    :param grad_model: whether to compute model parameter gradients, defaults to True
    :type grad_model: bool, optional
    :param grad_noise: whether to compute noise parameter gradients, defaults to True
    :type grad_noise: bool, optional
    :return: model and noise parameter gradients
    :rtype: list
    """
    # Parameters
    Y_obs                       = np.array(params['daily_counts'])
    Nd                          = params['t'].shape[0]
    Nreg                        = params['Nreg']
    W                           = params['W']
    theta                       = theta_t(theta_hat)
    tauphi,lbdphi,siga,sigm     = theta[1][:]
    D                           = params['D']
    
    # Compute model predictions, "inner derivatives"
    if grad_model:
        Y_pred, dY_pred_dtheta = dYpred_dtheta(theta_hat,params,compute_preds=True)
    else:
        Y_pred = Ypred(theta_hat,params)
        
    # Arrays to hold derivatives
    grad_t0_N_k_theta = np.zeros([Nreg,4,Nd])
    grad_tau_phi_siga_sigm = np.zeros([Nd,4])
    if Nreg > 1:
        Sigma_GMRF_cov = inv(D - lbdphi*W)
    else:
        Sigma_GMRF_cov = np.zeros_like(W)
        
    # Grad loops
    for t in range(Nd):
        y_pred_d = Y_pred[:,t]
        y_obs_d = Y_obs[:,t]
        dy = y_obs_d - y_pred_d
        Sigma_d = tauphi*Sigma_GMRF_cov + diag((siga + sigm*y_pred_d)**2)
        Sigma_d_inv = inv(Sigma_d)
        
        # Compute grad_t0_N_k_theta regions, days
        if grad_model:  
            for r in range(Nreg):
                # Compute: grad log det(Sigma_d)
                grad_t0_N_k_theta[r,:,t] = 2*sigm*Sigma_d_inv[r,r]*(siga + sigm*y_pred_d[r])*dY_pred_dtheta[r,:,t]
                # Compute: grad (y_obs - y_pred)^T Sigma (y_obs - y_pred)
                for i in range(4):
                    dy_d_dvi = np.zeros_like(y_pred_d)
                    dy_d_dvi[r] = dY_pred_dtheta[r,i,t]
                    dSigma_d_dvi = 2*sigm*diag((siga + sigm*y_pred_d)*dy_d_dvi)
                    grad_t0_N_k_theta[r,i,t] += -qform(dy,Sigma_d_inv @ dSigma_d_dvi @ Sigma_d_inv,dy) - 2*qform(dy,Sigma_d_inv,dy_d_dvi)
        # Compute grad_tau_phi
        if Nreg > 1:
            grad_tau_phi_siga_sigm[t,0] = theta_t_var['tauphi']['dt'](theta_hat[1][0])*(tr(Sigma_d_inv @ Sigma_GMRF_cov) - qform(dy, Sigma_d_inv @ Sigma_GMRF_cov @ Sigma_d_inv, dy))
            grad_tau_phi_siga_sigm[t,1] = theta_t_var['lbdphi']['dt'](theta_hat[1][1])*tauphi*(tr(Sigma_d_inv @ Sigma_GMRF_cov @ W @ Sigma_GMRF_cov) - qform(dy, Sigma_d_inv @ Sigma_GMRF_cov @ W @ Sigma_GMRF_cov @ Sigma_d_inv, dy))
        grad_tau_phi_siga_sigm[t,2] = theta_t_var['siga']['dt'](theta_hat[1][2])*2*(tr(Sigma_d_inv @ diag(siga + sigm*y_pred_d)) - qform(dy, Sigma_d_inv @ diag(siga + sigm*y_pred_d) @ Sigma_d_inv, dy))
        grad_tau_phi_siga_sigm[t,3] = theta_t_var['sigm']['dt'](theta_hat[1][3])*2*(tr(Sigma_d_inv @ diag((siga + sigm*y_pred_d)*y_pred_d)) - qform(dy, Sigma_d_inv @ diag((siga + sigm*y_pred_d)*y_pred_d) @ Sigma_d_inv, dy))
        
    # Sum over days
    grad_t0_N_k_theta = -0.5*np.sum(grad_t0_N_k_theta,axis=2)   
    grad_tau_phi_siga_sigm = -0.5*np.sum(grad_tau_phi_siga_sigm,axis=0) 
    return [grad_t0_N_k_theta,grad_tau_phi_siga_sigm]

###################################################################################################################################
####################################################### Numerical gradients #######################################################
###################################################################################################################################

# First-order derivatives
def dloglik_dtheta_num_4th_ord(theta,params):
    h = params['h']
    Nreg = theta[0].shape[0]
    theta_1 = theta[0]
    theta_2 = theta[1]
    grad_1 = np.zeros([Nreg,4])
    grad_2 = np.zeros_like(theta_2)
    for r in range(Nreg):
        for v in range(4):
            theta_m2h = np.copy(theta_1)
            theta_m2h[r,v] -= 2*h
            theta_m1h = np.copy(theta_1)
            theta_m1h[r,v] -= h
            theta_p1h = np.copy(theta_1)
            theta_p1h[r,v] += h
            theta_p2h = np.copy(theta_1)
            theta_p2h[r,v] += 2*h
            ll_m2h = loglik_gaussian([theta_m2h,theta_2], params)
            ll_m1h = loglik_gaussian([theta_m1h,theta_2], params)
            ll_p1h = loglik_gaussian([theta_p1h,theta_2], params)
            ll_p2h = loglik_gaussian([theta_p2h,theta_2], params)
            grad_1[r,v] = ((1/12)*ll_m2h + (-2/3)*ll_m1h + (2/3)*ll_p1h + (-1/12)*ll_p2h)/h
    for i in range(theta_2.shape[0]):
        theta_m2h = np.copy(theta_2)
        theta_m2h[i] -= 2*h
        theta_m1h = np.copy(theta_2)
        theta_m1h[i] -= h
        theta_p1h = np.copy(theta_2)
        theta_p1h[i] += h
        theta_p2h = np.copy(theta_2)
        theta_p2h[i] += 2*h
        ll_m2h = loglik_gaussian([theta_1,theta_m2h], params)
        ll_m1h = loglik_gaussian([theta_1,theta_m1h], params)
        ll_p1h = loglik_gaussian([theta_1,theta_p1h], params)
        ll_p2h = loglik_gaussian([theta_1,theta_p2h], params)
        grad_2[i] = ((1/12)*ll_m2h + (-2/3)*ll_m1h + (2/3)*ll_p1h + (-1/12)*ll_p2h)/h
    return [grad_1,grad_2]

def dloglik_dtheta_num_2nd_ord(theta,params):
    h = params['h']
    Nreg = theta[0].shape[0]
    theta_1 = theta[0]
    theta_2 = theta[1]
    grad_1 = np.zeros([Nreg,4])
    grad_2 = np.zeros_like(theta_2)
    for r in range(Nreg):
        for v in range(4):
            theta_m1h = np.copy(theta_1)
            theta_m1h[r,v] -= h
            theta_p1h = np.copy(theta_1)
            theta_p1h[r,v] += h
            ll_m1h = loglik_gaussian([theta_m1h,theta_2], params)
            ll_p1h = loglik_gaussian([theta_p1h,theta_2], params)
            grad_1[r,v] = ((1/2)*ll_p1h + (-1/2)*ll_m1h)/h
    for i in range(theta_2.shape[0]):
        theta_m1h = np.copy(theta_2)
        theta_m1h[i] -= h
        theta_p1h = np.copy(theta_2)
        theta_p1h[i] += h
        ll_m1h = loglik_gaussian([theta_1,theta_m1h], params)
        ll_p1h = loglik_gaussian([theta_1,theta_p1h], params)
        grad_2[i] = ((1/2)*ll_p1h + (-1/2)*ll_m1h)/h
    return [grad_1,grad_2]

# Second-order derivatives
def dloglik_dtheta_sq_diag_2nd_ord(theta,params):
    h = params['h']
    Nreg = theta[0].shape[0]
    theta_1 = theta[0]
    theta_2 = theta[1]
    hess_diag = []
    theta_p1h = [np.copy(theta[i])+h for i in range(len(theta))]
    theta_m1h = [np.copy(theta[i])-h for i in range(len(theta))]
    grad_p1h = dloglik_dtheta(theta_p1h,params)
    grad_m1h = dloglik_dtheta(theta_m1h,params)
    hess_diag_1 = (grad_p1h[0] - grad_m1h[0])/(2*h)
    hess_diag_2 = (grad_p1h[1] - grad_m1h[1])/(2*h)
    return [hess_diag_1,hess_diag_2]
    
