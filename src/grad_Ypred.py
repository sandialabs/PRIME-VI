import numpy as np
import ray
from Ypred import Ypred
from utils import fGamma, F_LN, qform
from loglik import loglik_gaussian
from scipy.special import gamma, digamma, erf
from scipy import integrate
from numpy.linalg import inv
from numpy import trace as tr, dot, diag, log, sqrt, power, exp
from transforms import theta_t, theta_t_var
import matplotlib.pyplot as plt

def dfGamma_dt0_factor(tau,k,theta):
    return np.where(tau > 0,(tau - k*theta + theta)/(theta*tau),0)

def dfGamma_dt0_func(tau,k,theta):
    return (tau+(1-k)*theta)*(theta**(-k-1))*(tau**(k-2))*exp(-tau/theta)/gamma(k)

# digamma is logarithmic derivative Gamma'(k)/Gamma(k)
def dfGamma_dk_factor(tau,k,theta):
    return np.where(tau > 1e-5, np.log(tau) - np.log(theta) - digamma(k),0)

def dfGamma_dtheta_factor(tau,k,theta):
    return (tau - k*theta)/theta**2

###################################################################################################################################
###################################################### Quadrature gradients #######################################################
###################################################################################################################################
   
# Derivatives of y_d w.r.t. theta for each day d
# theta is (Nregions x 4), columns are (t0, N, qshape, qscale)
def dYpred_dtheta(theta_hat,params,compute_grads=True,compute_preds=False):
    """Computes the derivative of the model predictions with respect to theta_hat.
    Derivatives are approximated using quadrature.

    :param theta_hat: unconstrained model parameters
    :type theta_hat: list
    :param params: additional model data
    :type params: dict
    :param compute_grads: whether to compute gradients, defaults to True
    :type compute_grads: bool, optional
    :param compute_preds: whether to compute predictions, defaults to False
    :type compute_preds: bool, optional
    :return: model parameter gradients
    :rtype: np.ndarray
    """
    # Parameters
    inc_sigma           = params['incubation_sigma']
    inc_mu              = params['incubation_median']
    Nd                  = params['t'].shape[0]
    t                   = params['t']
    Nreg                = params['Nreg']
    nquad               = params['nquad_grad']
    theta               = theta_t(theta_hat)
    F_LN_l,F_LN_u       = params['F_LN_lims']
    theta               = theta[0]
    theta_hat           = theta_hat[0]
    
    # Generate quadrature weights, points
    x_quad,w_quad = np.polynomial.legendre.leggauss(nquad) # x in [-1,1]   
        
    if compute_preds:
        preds = np.zeros([Nreg,Nd])
    if compute_grads:
        grads = np.zeros([Nreg,4,Nd])
    for d in range(Nd):
        for r in range(Nreg):
            t0, N, k, theta_scale = theta[r]
            t0_hat, N_hat, k_hat, theta_scale_hat = theta_hat[r]
            # int_lb = t0
            # int_ub = t[d]
            int_lb = max(t0,t[d]-F_LN_u)
            int_ub = t[d] - F_LN_l
            c = 0.5*(int_ub-int_lb)
            if c > 0:
                wi = c*w_quad
                taui = c*(x_quad + 1) + int_lb
                fGamma_vals = fGamma(taui-t0,k,theta_scale)
                # F_LN evaluation - constant w.r.t. theta
                F_LN_factor = F_LN(t[d]-taui,inc_mu,inc_sigma)-F_LN((t[d]-1)-taui,inc_mu,inc_sigma)
                # Derivatives of f_Gamma
                dfGamma_dt0 = dfGamma_dt0_func(taui-t0,k,theta_scale)
                dfGamma_dk = dfGamma_dk_factor(taui-t0,k,theta_scale) * fGamma_vals
                dfGamma_dtheta = dfGamma_dtheta_factor(taui-t0,k,theta_scale) * fGamma_vals     
                # Derivatives of preds via quadrature
                dy_dt0 = N*(np.dot(wi,dfGamma_dt0*F_LN_factor))
                dy_dN = np.dot(wi,fGamma_vals*F_LN_factor)
                if compute_preds:
                    preds[r,d] = N*dy_dN
                dy_dk = N*np.dot(wi,dfGamma_dk*F_LN_factor)
                dy_dtheta = N*np.dot(wi,dfGamma_dtheta*F_LN_factor)
            
                
                # chain rule
                grads[r,:,d] = np.array([theta_t_var['t0']['dt'](t0_hat)*dy_dt0,
                                         theta_t_var['N']['dt'](N_hat)*dy_dN,
                                         theta_t_var['k']['dt'](k_hat)*dy_dk,
                                         theta_t_var['theta']['dt'](theta_scale_hat)*dy_dtheta])  
    ans = []
    if compute_preds:
        ans.append(preds)
    ans.append(grads)
    if len(ans) == 1:
        return ans[0]
    else:
        return ans

###################################################################################################################################
####################################################### Numerical gradients #######################################################
###################################################################################################################################

def dYpred_dtheta_num_2nd_ord(theta_in,params):
    theta = theta_in[0]
    Nreg = theta.shape[0]
    Nd = params['ts'].shape[0]
    h = params['h']
    grad = np.zeros([Nreg,Nd,4])
    for i in range(4):
        theta_pert_fwd = np.copy(theta)
        theta_pert_fwd[:,i] = theta_pert_fwd[:,i] + h
        theta_pert_bwd = np.copy(theta)
        theta_pert_bwd[:,i] = theta_pert_bwd[:,i] - h
        model_pred_fwd = Ypred(theta_pert_fwd,params)
        model_pred_bwd = Ypred(theta_pert_bwd,params)
        grad[:,:,i] = (model_pred_fwd - model_pred_bwd)/(2*h)
    return np.transpose(grad,(0,2,1))

def dYpred_dtheta_num_4th_ord(theta,params):
    theta1 = theta[0]
    theta2 = theta[1]
    h = params['h']
    Nreg = params['Nreg']
    Nd = params['ts'].shape[0]
    grad = np.zeros([Nreg,Nd,4])
    for i in range(4):
        theta1_m2h = np.copy(theta1)
        theta1_m2h[:,i] -= 2*h
        theta1_m1h = np.copy(theta1)
        theta1_m1h[:,i] -= h
        theta1_p1h = np.copy(theta1)
        theta1_p1h[:,i] += h
        theta1_p2h = np.copy(theta1)
        theta1_p2h[:,i] += 2*h
        grad[:,:,i] = ((1/12)*Ypred([theta1_m2h,theta2],params) + (-2/3)*Ypred([theta1_m1h,theta2],params) + (2/3)*Ypred([theta1_p1h,theta2],params) + (-1/12)*Ypred([theta1_p2h,theta2],params))/h
    return np.transpose(grad,(0,2,1))