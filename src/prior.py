import numpy as np
import ray
from Ypred import Ypred
from utils import block_decompose_1D_tensor, fGamma, F_LN, qform
from scipy.special import gamma, digamma, erf
from numpy.linalg import inv
from numpy import trace as tr, dot, diag, log, sqrt, power, exp

######################### Basic priors #########################
def log_prior_gauss(theta,mu,sig):
    return -log(sig)-0.5*(log(2*np.pi) + ((theta-mu)/sig)**2)

def grad_log_prior_gauss(theta,mu,sig):
    return -(theta-mu)/(sig**2)

def log_prior_bump_uniform(theta,a,b,n):
    # return 0.
    mu = (b+a)/2
    sig = (b-a)/2
    return np.where(np.abs((theta-mu)/sig)<1,exp(1/(((theta-mu)/sig)**(2*n)-1)),0)
    
def grad_log_prior_bump_uniform(theta,a,b,n):
    mu = (b+a)/2
    sig = (b-a)/2
    return np.where(np.abs((theta-mu)/sig)<1,-((2*n)/sig) * (((theta-mu)/sig)**(2*n) - 1)**(-2) * ((theta-mu)/sig)**(2*n - 1),0)

def log_improper_uniform(theta):
    return 0.

def grad_log_improper_uniform(theta):
    return 0.

##################### Independent joint prior ########################
def log_prior(theta,params,grad=False):
    # Prior factors over each of the (Nreg x 4) + 2 theta variables
    Nreg = params['Nreg']
    logp1 = np.zeros([Nreg,4])
    logp2 = np.zeros(4)
    if grad:
        gauss = grad_log_prior_gauss
        bump_uniform = grad_log_prior_bump_uniform
        uniform = grad_log_improper_uniform
    else:
        gauss = log_prior_gauss
        bump_uniform = log_prior_bump_uniform
        uniform = log_improper_uniform
    for r in range(Nreg):
        for v in range(4): 
            if params['prior_types'][0][r,v] == 'gaussian':
                [mu,sig] = params['prior_params'][0][r,v]
                logp1[r,v] = gauss(theta[0][r,v],mu,sig)
            elif params['prior_types'][0][r,v] == 'uniform':
                [a,b,n] = params['prior_params'][0][r,v]
                logp1[r,v] = bump_uniform(theta[0][r,v],a,b,n)
            elif params['prior_types'][0][r,v] == 'none':
                logp1[r,v] = uniform(theta[0][r,v])
    for i in range(4):
        if params['prior_types'][1][i] == 'gaussian':
            [mu,sig] = params['prior_params'][1][i]
            logp2[i] = gauss(theta[1][i],mu,sig)
        elif params['prior_types'][1][i] == 'uniform':
            [a,b,n] = params['prior_params'][1][i]
            logp2[i] = bump_uniform(theta[1][i],a,b,n)
        elif params['prior_types'][1][i] == 'none':
            logp2[i] = uniform(theta[1][i])
    if grad:
        return [logp1,logp2]
    else:
        return np.sum(logp1)+np.sum(logp2)
