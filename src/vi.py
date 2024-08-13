import numpy as np
import sys,os
import math
import ray
from Ypred import Ypred
from utils import block_decompose_1D_tensor, fGamma, F_LN, qform, load_obj, save_obj, theta_samples
from transforms import  theta_to_vec, vec_to_theta, sigma, sigma_inv, grad_sigma,\
                        vec_to_phi, phi_to_vec, phi_hat_to_phi, phi_hat_to_phi
from scipy.special import gamma, digamma, erf
from numpy.linalg import inv, norm
from numpy import trace as tr, dot, diag, log, sqrt, power, exp
from numpy.random import randn
from loglik import loglik_gaussian
from prior import log_prior
from grad_loglik import dloglik_dtheta, dloglik_dtheta_num_4th_ord
from gradient_descent import adam

# phi defines parameters of mean-field Gaussian surrogate posterior
# Format: [ (Nreg x 4 x 2), 2 x 2 ]
# Innermost dimension gives (mu,rho) where sig = log(exp(rho) + 1)
# so that (mu,sig) are the mean, std of a Gaussian component

def dlog_gauss_dmu(theta,mu,sig):
    return (theta-mu)/(sig**2)

def dlog_gauss_dsig(theta,mu,sig):
    return -(1/sig)+((theta-mu)**2)*(sig**(-3))

def grad_log_gaussian(theta_hat,phi_hat):
    phi = phi_hat_to_phi(phi_hat)
    mu_1 = phi[0][...,0]
    mu_2 = phi[1][...,0]
    sig_1 = phi[0][...,1]
    sig_2 = phi[1][...,1]
    grad_mu_1 = np.expand_dims(dlog_gauss_dmu(theta_hat[0],mu_1,sig_1),axis=2)
    grad_mu_2 = np.expand_dims(dlog_gauss_dmu(theta_hat[1],mu_2,sig_2),axis=1)
    grad_sig_1 = np.expand_dims(dlog_gauss_dsig(theta_hat[0],mu_1,sig_1)*grad_sigma(phi_hat[0][...,1]),axis=2)
    grad_sig_2 = np.expand_dims(dlog_gauss_dsig(theta_hat[1],mu_2,sig_2)*grad_sigma(phi_hat[1][...,1]),axis=1)
    
    grad_1 = np.concatenate([grad_mu_1,grad_sig_1],axis=2)
    grad_2 = np.concatenate([grad_mu_2,grad_sig_2],axis=1)

    return [grad_1,grad_2]

# Entropy of mean-field Gaussian
def gaussian_entropy(phi_hat):
    phi = phi_hat_to_phi(phi_hat)
    d = np.prod(phi[0].shape) + np.prod(phi[1].shape)
    return 0.5*(d*log(2*np.pi*np.e) + np.sum(2*log(phi[0][:,:,1])) + np.sum(2*log(phi[1][:,1])))

# Gradient of entropy of mean-field gaussian w.r.t. phi
def grad_phi_gaussian_entropy(phi_hat):
    phi = phi_hat_to_phi(phi_hat)
    grad_1 = np.zeros_like(phi[0])
    grad_2 = np.zeros_like(phi[1])
    grad_1[...,1] = (1/phi[0][...,1])*grad_sigma(phi_hat[0][...,1])
    grad_2[...,1] = (1/phi[1][...,1])*grad_sigma(phi_hat[1][...,1])
    return [grad_1,grad_2]

@ray.remote
def ELBO_mc_sample(phi_hat,Ns,params):
    phi = phi_hat_to_phi(phi_hat)
    theta_hats,eps = theta_samples(phi,params,Ns)
    ELBO_mc_samples = np.zeros(Ns)
    for i in range(Ns):
        sample = loglik_gaussian([theta_hats[0][i],theta_hats[1][i]], params) + log_prior([theta_hats[0][i],theta_hats[1][i]], params)
        ELBO_mc_samples[i] = sample
    s = np.sum(ELBO_mc_samples)
    return s

# MC estimate of ELBO objective  
def ELBO_mc_est(phi_hat,Ns,params):
    N_mc_blocks = params['N_mc_blocks']
    Ns_block = int(Ns/N_mc_blocks)     # Number of samples per process
    entropy = gaussian_entropy(phi_hat)
    futures = [ELBO_mc_sample.remote(phi_hat,Ns_block,params) for i in range(N_mc_blocks)]
    ELBO_mc_samples = ray.get(futures)
    ELBO_mc_samples = np.array(ELBO_mc_samples)
    est = -entropy - (1/Ns)*np.sum(ELBO_mc_samples)
    return est

@ray.remote
def grad_phi_ELBO_mc_sample(phi_hat,theta_hats,eps,params):
    Ns = theta_hats[0].shape[0]
    grad_phi_ELBO_mc_samples = [np.zeros([Ns]+list(phi_hat[0].shape)),\
                                np.zeros([Ns]+list(phi_hat[1].shape))]
    for i in range(Ns):
        grad_phi_ELBO_mc_sample = [np.zeros(phi_hat[0].shape),np.zeros(phi_hat[1].shape)]
        grad_theta_loglik_1,grad_theta_loglik_2 = dloglik_dtheta([theta_hats[0][i],theta_hats[1][i]],params)             # gradient of log-lik w.r.t. theta
        grad_theta_logprior_1,grad_theta_logprior_2 = log_prior([theta_hats[0][i],theta_hats[1][i]],params,grad=True)   # gradient of log-prior w.r.t. theta
        # grad mu
        grad_phi_ELBO_mc_sample[0][:,:,0] = -(grad_theta_loglik_1 + grad_theta_logprior_1)
        grad_phi_ELBO_mc_sample[1][:,0] = -(grad_theta_loglik_2 + grad_theta_logprior_2)
        # grad rho
        grad_phi_ELBO_mc_sample[0][:,:,1] = -(grad_theta_loglik_1 + grad_theta_logprior_1)*grad_sigma(phi_hat[0][:,:,1])*eps[0][i]
        grad_phi_ELBO_mc_sample[1][:,1] = -(grad_theta_loglik_2 + grad_theta_logprior_2)*grad_sigma(phi_hat[1][:,1])*eps[1][i]
        
        grad_phi_ELBO_mc_samples[0][i] = grad_phi_ELBO_mc_sample[0]
        grad_phi_ELBO_mc_samples[1][i] = grad_phi_ELBO_mc_sample[1]
    return grad_phi_ELBO_mc_samples
    
# MC estimate of gradient w.r.t. phi of ELBO objective
def grad_phi_ELBO_mc_est(phi_hat,Ns,params):
    phi = phi_hat_to_phi(phi_hat)
    theta_hats,eps = theta_samples(phi,params,Ns)
    N_mc_blocks = params['N_mc_blocks']
    Ns_block = int(Ns/N_mc_blocks)     # Number of samples per process
    grad_phi_entropy = grad_phi_gaussian_entropy(phi_hat)       # Gradient of entropy w.r.t. phi has closed form for Gaussian distributions
    futures = [grad_phi_ELBO_mc_sample.remote(  phi_hat,\
                                                [theta_hats[0][i*Ns_block:(i+1)*Ns_block],theta_hats[1][i*Ns_block:(i+1)*Ns_block]],\
                                                [eps[0][i*Ns_block:(i+1)*Ns_block],eps[1][i*Ns_block:(i+1)*Ns_block]],\
                                                params) for i in range(N_mc_blocks)]
    grad_samples = ray.get(futures)
    
    grad_phi_ELBO_mc_samples_1 = np.vstack([grad_samples[i][0] for i in range(N_mc_blocks)])
    grad_phi_ELBO_mc_samples_2 = np.vstack([grad_samples[i][1] for i in range(N_mc_blocks)])
    grad_phi_ELBO_mc = [-grad_phi_entropy[0] + ((1/Ns)*np.sum(grad_phi_ELBO_mc_samples_1,axis=0)),\
                        -grad_phi_entropy[1] + ((1/Ns)*np.sum(grad_phi_ELBO_mc_samples_2,axis=0))]
    return grad_phi_ELBO_mc

@ray.remote
def grad_phi_ELBO_mc_score_sample(phi_hat,theta_hats,eps,params):
    Ns = theta_hats[0].shape[0]
    grad_phi_ELBO_mc_samples = [np.zeros([Ns]+list(phi_hat[0].shape)),\
                                np.zeros([Ns]+list(phi_hat[1].shape))]
    for i in range(Ns):
        grad_phi_ELBO_mc_sample = [np.zeros(phi_hat[0].shape),np.zeros(phi_hat[1].shape)]
        grad_gauss_1,grad_gauss_2 = grad_log_gaussian([theta_hats[0][i],theta_hats[1][i]],phi_hat)
        logpost = loglik_gaussian([theta_hats[0][i],theta_hats[1][i]], params) + log_prior([theta_hats[0][i],theta_hats[1][i]], params, grad=False)

        grad_phi_ELBO_mc_samples[0][i] = logpost*grad_gauss_1
        grad_phi_ELBO_mc_samples[1][i] = logpost*grad_gauss_2
    return grad_phi_ELBO_mc_samples
    
# Score estimate of gradient w.r.t. phi of ELBO objective
def grad_phi_ELBO_mc_score_est(phi_hat,Ns,params):
    phi = phi_hat_to_phi(phi_hat)
    theta_hats,eps = theta_samples(phi,params,Ns)
    N_mc_blocks = params['N_mc_blocks']
    Ns_block = int(Ns/N_mc_blocks)     # Number of samples per process
    grad_phi_entropy = grad_phi_gaussian_entropy(phi_hat)       # Gradient of entropy w.r.t. phi has closed form for Gaussian distributions
    futures = [grad_phi_ELBO_mc_score_sample.remote(  phi_hat,\
                                                [theta_hats[0][i*Ns_block:(i+1)*Ns_block],theta_hats[1][i*Ns_block:(i+1)*Ns_block]],\
                                                [eps[0][i*Ns_block:(i+1)*Ns_block],eps[1][i*Ns_block:(i+1)*Ns_block]],\
                                                params) for i in range(N_mc_blocks)]
    grad_samples = ray.get(futures)
    
    grad_phi_ELBO_mc_samples_1 = np.vstack([grad_samples[i][0] for i in range(N_mc_blocks)])
    grad_phi_ELBO_mc_samples_2 = np.vstack([grad_samples[i][1] for i in range(N_mc_blocks)])
    grad_phi_ELBO_mc = [-grad_phi_entropy[0] - ((1/Ns)*np.sum(grad_phi_ELBO_mc_samples_1,axis=0)),\
                        -grad_phi_entropy[1] - ((1/Ns)*np.sum(grad_phi_ELBO_mc_samples_2,axis=0))]
    return grad_phi_ELBO_mc

################################################# Return gradient samples #################################################
# Score estimate of gradient w.r.t. phi of ELBO objective
def grad_phi_ELBO_mc_score_est_2(phi_hat,Ns,params):
    phi = phi_hat_to_phi(phi_hat)
    theta_hats,eps = theta_samples(phi,params,Ns)
    N_mc_blocks = params['N_mc_blocks']
    Ns_block = int(Ns/N_mc_blocks)     # Number of samples per process
    grad_phi_entropy = grad_phi_gaussian_entropy(phi_hat)       # Gradient of entropy w.r.t. phi has closed form for Gaussian distributions
    futures = [grad_phi_ELBO_mc_score_sample.remote(  phi_hat,\
                                                [theta_hats[0][i*Ns_block:(i+1)*Ns_block],theta_hats[1][i*Ns_block:(i+1)*Ns_block]],\
                                                [eps[0][i*Ns_block:(i+1)*Ns_block],eps[1][i*Ns_block:(i+1)*Ns_block]],\
                                                params) for i in range(N_mc_blocks)]
    grad_samples = ray.get(futures)
    
    
    grad_phi_ELBO_mc_samples_1 = np.vstack([grad_samples[i][0] for i in range(N_mc_blocks)])
    grad_phi_ELBO_mc_samples_2 = np.vstack([grad_samples[i][1] for i in range(N_mc_blocks)])
    
    grad_samples_vec = np.zeros([Ns,phi_to_vec(phi_hat).shape[0]])
    for i in range(Ns):
        grad_samples_vec[i,:] = phi_to_vec([grad_phi_ELBO_mc_samples_1[i],grad_phi_ELBO_mc_samples_2[i]])
    return grad_samples_vec

# MC estimate of gradient w.r.t. phi of ELBO objective
def grad_phi_ELBO_mc_est_2(phi_hat,Ns,params):
    phi = phi_hat_to_phi(phi_hat)
    theta_hats,eps = theta_samples(phi,params,Ns)
    N_mc_blocks = params['N_mc_blocks']
    Ns_block = int(Ns/N_mc_blocks)     # Number of samples per process
    grad_phi_entropy = grad_phi_gaussian_entropy(phi_hat)       # Gradient of entropy w.r.t. phi has closed form for Gaussian distributions
    futures = [grad_phi_ELBO_mc_sample.remote(  phi_hat,\
                                                [theta_hats[0][i*Ns_block:(i+1)*Ns_block],theta_hats[1][i*Ns_block:(i+1)*Ns_block]],\
                                                [eps[0][i*Ns_block:(i+1)*Ns_block],eps[1][i*Ns_block:(i+1)*Ns_block]],\
                                                params) for i in range(N_mc_blocks)]
    grad_samples = ray.get(futures)
    
    grad_phi_ELBO_mc_samples_1 = np.vstack([grad_samples[i][0] for i in range(N_mc_blocks)])
    grad_phi_ELBO_mc_samples_2 = np.vstack([grad_samples[i][1] for i in range(N_mc_blocks)])
    
    grad_samples_vec = np.zeros([Ns,phi_to_vec(phi_hat).shape[0]])
    for i in range(Ns):
        grad_samples_vec[i,:] = phi_to_vec([grad_phi_ELBO_mc_samples_1[i],grad_phi_ELBO_mc_samples_2[i]])
    return grad_samples_vec

# Score estimate of gradient w.r.t. phi of ELBO objective
def grad_phi_ELBO_mc_score_est_3(phi_hat,Ns,params):
    phi = phi_hat_to_phi(phi_hat)
    theta_hats,eps = theta_samples(phi,params,Ns)
    N_mc_blocks = params['N_mc_blocks']
    Ns_block = int(Ns/N_mc_blocks)     # Number of samples per process
    grad_phi_entropy = grad_phi_gaussian_entropy(phi_hat)       # Gradient of entropy w.r.t. phi has closed form for Gaussian distributions
    futures = [grad_phi_ELBO_mc_score_sample.remote(  phi_hat,\
                                                [theta_hats[0][i*Ns_block:(i+1)*Ns_block],theta_hats[1][i*Ns_block:(i+1)*Ns_block]],\
                                                [eps[0][i*Ns_block:(i+1)*Ns_block],eps[1][i*Ns_block:(i+1)*Ns_block]],\
                                                params) for i in range(N_mc_blocks)]
    grad_samples = ray.get(futures)
    
    grad_phi_ELBO_mc_samples_1 = np.vstack([grad_samples[i][0] for i in range(N_mc_blocks)])
    grad_phi_ELBO_mc_samples_2 = np.vstack([grad_samples[i][1] for i in range(N_mc_blocks)])
    grad_phi_ELBO_mc = [-((1/Ns)*np.sum(grad_phi_ELBO_mc_samples_1,axis=0)),\
                        - ((1/Ns)*np.sum(grad_phi_ELBO_mc_samples_2,axis=0))]
    return grad_phi_ELBO_mc
############################################################################################################################

def adam_VI(phi_hat_0,
            params,
            max_its,
            tol,
            alpha_0=0.001,
            alpha_schedule=None,
            log_freq=1,
            log_header='',
            obj_freq=1,
            Ns_grad=10,
            Ns_obj=10):
    """Carries out variational inference by optimizing the ELBO with the reparametrization trick and ADAM

    :param phi_hat_0: initial outbreak model and noise parameters
    :type phi_hat_0: list
    :param params: additional model parameters
    :type params: dict
    :param max_its: max number of iterations
    :type max_its: int
    :param tol: zero gradient tolerance
    :type tol: float
    :param alpha_0: initial learning rate, defaults to 0.001
    :type alpha_0: float, optional
    :param alpha_schedule: learning rate schedule, defaults to None
    :type alpha_schedule: function, optional
    :param obj_freq: how frequently to compute objective function values, defaults to 1
    :type obj_freq: int, optional
    :param log_freq: how frequently to output logs, defaults to 1
    :type log_freq: int, optional
    :param log_header: header for log file, defaults to ''
    :type log_header: str, optional
    :param Ns_grad: number of Monte Carlo samples to use to approximate reparametrization gradient, defaults to 10
    :type Ns_grad: int, optional
    :param Ns_obj: number of Monte Carlo samples to use to approximate ELBO objective, defaults to 10
    :type Ns_obj: int, optional
    :return: optimized model and noise parameters
    :rtype: np.ndarray
    """
    
    obj_f = lambda phi_hat:ELBO_mc_est(vec_to_phi(phi_hat), Ns_obj, params)
    grad_f = lambda phi_hat:phi_to_vec(grad_phi_ELBO_mc_est(vec_to_phi(phi_hat), Ns_grad, params))
    return adam(theta_0=phi_hat_0,
                params=params,
                max_its=max_its,
                tol=tol,
                obj_f=obj_f,
                grad_f=grad_f,
                alpha_0=alpha_0,
                alpha_schedule=alpha_schedule,
                log_freq=log_freq,
                log_header=log_header,
                obj_freq=obj_freq
                )

def adam_bbox_VI(phi_hat_0,
            params,
            max_its,
            tol,
            alpha_0=0.001,
            alpha_schedule=None,
            log_freq=1,
            log_header='',
            obj_freq=1,
            Ns_grad=10,
            Ns_obj=10):
    """Carries out variational inference by optimizing the ELBO with the score / black-box approach and ADAM

    :param phi_hat_0: initial outbreak model and noise parameters
    :type phi_hat_0: list
    :param params: additional model parameters
    :type params: dict
    :param max_its: max number of iterations
    :type max_its: int
    :param tol: zero gradient tolerance
    :type tol: float
    :param alpha_0: initial learning rate, defaults to 0.001
    :type alpha_0: float, optional
    :param alpha_schedule: learning rate schedule, defaults to None
    :type alpha_schedule: function, optional
    :param obj_freq: how frequently to compute objective function values, defaults to 1
    :type obj_freq: int, optional
    :param log_freq: how frequently to output logs, defaults to 1
    :type log_freq: int, optional
    :param log_header: header for log file, defaults to ''
    :type log_header: str, optional
    :param Ns_grad: number of Monte Carlo samples to use to approximate score/black-box gradient, defaults to 10
    :type Ns_grad: int, optional
    :param Ns_obj: number of Monte Carlo samples to use to approximate ELBO objective, defaults to 10
    :type Ns_obj: int, optional
    :return: optimized model and noise parameters
    :rtype: np.ndarray
    """
    
    obj_f = lambda phi_hat:ELBO_mc_est(vec_to_phi(phi_hat), Ns_obj, params)
    grad_f = lambda phi_hat:phi_to_vec(grad_phi_ELBO_mc_score_est(vec_to_phi(phi_hat), Ns_grad, params))
    return adam(theta_0=phi_hat_0,
                params=params,
                max_its=max_its,
                tol=tol,
                obj_f=obj_f,
                grad_f=grad_f,
                alpha_0=alpha_0,
                alpha_schedule=alpha_schedule,
                log_freq=log_freq,
                log_header=log_header,
                obj_freq=obj_freq
                )

def adam_MLE(theta_0,
             params,
             max_its,
             tol,
             alpha_0=0.001,
             alpha_schedule=None,
             log_freq=1,
             log_header='',
             obj_freq=1,
             grad_model=True,
             grad_noise=True):

    obj_f = lambda theta:loglik_gaussian(vec_to_theta(theta), params)+log_prior(vec_to_theta(theta), params)
    grad_f = lambda theta:-(theta_to_vec(dloglik_dtheta(vec_to_theta(theta), params, grad_model, grad_noise))+theta_to_vec(log_prior(vec_to_theta(theta), params, grad=True)))
    return adam(theta_0=theta_0,
                params=params,
                max_its=max_its,
                tol=tol,
                obj_f=obj_f,
                grad_f=grad_f,
                alpha_0=alpha_0,
                alpha_schedule=alpha_schedule,
                log_freq=log_freq,
                log_header=log_header,
                obj_freq=obj_freq
                )

##############################################################################################################################
################################################ DEBUG versions ##############################################################
##############################################################################################################################
# @ray.remote
# def grad_phi_ELBO_mc_sample(phi_hat,theta_hats,eps,params):
#     Ns = theta_hats[0].shape[0]
#     grad_phi_ELBO_mc_samples = [np.zeros([Ns]+list(phi_hat[0].shape)),\
#                                 np.zeros([Ns]+list(phi_hat[1].shape))]
#     for i in range(Ns):
#         grad_phi_ELBO_mc_sample = [np.zeros(phi_hat[0].shape),np.zeros(phi_hat[1].shape)]
#         grad_theta_loglik_1,grad_theta_loglik_2 = dloglik_dtheta([theta_hats[0][i],theta_hats[1][i]],params)             # gradient of log-lik w.r.t. theta
#         grad_theta_logprior_1,grad_theta_logprior_2 = log_prior([theta_hats[0][i],theta_hats[1][i]],params,grad=True)   # gradient of log-prior w.r.t. theta
#         # grad mu
#         grad_phi_ELBO_mc_sample[0][:,:,0] = -(grad_theta_loglik_1 + grad_theta_logprior_1)
#         grad_phi_ELBO_mc_sample[1][:,0] = -(grad_theta_loglik_2 + grad_theta_logprior_2)
#         # grad rho
#         grad_phi_ELBO_mc_sample[0][:,:,1] = -(grad_theta_loglik_1 + grad_theta_logprior_1)*grad_sigma(phi_hat[0][:,:,1])*eps[0][i]
#         grad_phi_ELBO_mc_sample[1][:,1] = -(grad_theta_loglik_2 + grad_theta_logprior_2)*grad_sigma(phi_hat[1][:,1])*eps[1][i]
        
#         grad_phi_ELBO_mc_samples[0][i] = grad_phi_ELBO_mc_sample[0]
#         grad_phi_ELBO_mc_samples[1][i] = grad_phi_ELBO_mc_sample[1]
#     return grad_phi_ELBO_mc_samples
    
# # MC estimate of gradient w.r.t. phi of ELBO objective
# def grad_phi_ELBO_mc_est(phi_hat,Ns,params):
#     phi = phi_hat_to_phi(phi_hat)
#     theta_hats,eps = theta_samples(phi,params,Ns)
    
#     if not(os.path.exists('theta_hats_hist.pkl')):
#         save_obj([theta_hats],'theta_hats_hist.pkl')
#     else:
#         theta_hats_hist = load_obj('theta_hats_hist.pkl')
#         theta_hats_hist.append(theta_hats)
#         save_obj(theta_hats_hist,'theta_hats_hist.pkl')
        
    
#     N_mc_blocks = params['N_mc_blocks']
#     Ns_block = int(Ns/N_mc_blocks)     # Number of samples per process
#     grad_phi_ELBO_mc_samples = [np.zeros([Ns]+list(phi_hat[0].shape)),\
#                                 np.zeros([Ns]+list(phi_hat[1].shape))]
#     grad_phi_entropy = grad_phi_gaussian_entropy(phi_hat)       # Gradient of entropy w.r.t. phi has closed form for Gaussian distributions
#     futures = [grad_phi_ELBO_mc_sample.remote(  phi_hat,\
#                                                 [theta_hats[0][i*Ns_block:(i+1)*Ns_block],theta_hats[1][i*Ns_block:(i+1)*Ns_block]],\
#                                                 [eps[0][i*Ns_block:(i+1)*Ns_block],eps[1][i*Ns_block:(i+1)*Ns_block]],\
#                                                 params) for i in range(N_mc_blocks)]
    
    
#     grad_samples = ray.get(futures)
    
#     grad_phi_ELBO_mc_samples_1 = np.vstack([grad_samples[i][0] for i in range(N_mc_blocks)])
#     grad_phi_ELBO_mc_samples_2 = np.vstack([grad_samples[i][1] for i in range(N_mc_blocks)])
#     grad_phi_ELBO_mc = [-grad_phi_entropy[0] + ((1/Ns)*np.sum(grad_phi_ELBO_mc_samples_1,axis=0)),\
#                         -grad_phi_entropy[1] + ((1/Ns)*np.sum(grad_phi_ELBO_mc_samples_2,axis=0))]
#     return grad_phi_ELBO_mc
