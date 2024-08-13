import numpy as np
from numpy import diag, log, dot, power
from numpy.linalg import det, inv
from Ypred import Ypred
from utils import block_decompose_1D_tensor, qform
from transforms import theta_t
import matplotlib.pyplot as plt

# Compute gaussian log likelihood
def loglik_gaussian(theta_hat,params):
    """This computes Gaussian log-likelihood with spatial correlations

    :param theta_hat: unconstrained model parameters
    :type theta_hat: list
    :param params: additional model data
    :type params: dict
    :return: log likelihood
    :rtype: float
    """
    # Parameters
    Y_obs                   = np.array(params['daily_counts'])
    Nd                      = params['t'].shape[0]
    Nreg                    = params['Nreg']
    theta                   = theta_t(theta_hat)
    W                       = params['W']
    tauphi,lbdphi,siga,sigm = theta[1][:]
    D                       = params['D']
    
    Y_pred = Ypred(theta_hat,params)
    
    ll = 0.
    if Nreg > 1:
        Sigma_GMRF_cov = inv(D - lbdphi*W)
    else:
        Sigma_GMRF_cov = np.zeros_like(W)
    for d in range(Nd):
        y_pred = Y_pred[:,d]
        y_obs = Y_obs[:,d]
        dy = y_obs-y_pred
        Sigma_d = tauphi*Sigma_GMRF_cov + diag((siga + sigm*y_pred)**2)
        ld = log(det(Sigma_d))
        Sigma_d_inv = inv(Sigma_d)
        ll += log(det(Sigma_d))
        ll += qform(dy,inv(Sigma_d),dy)
    ll *= -0.5
    return ll
    