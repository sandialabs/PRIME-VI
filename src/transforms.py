import numpy as np
from numpy import trace as tr, dot, diag, log, sqrt, power, exp

####################### Basic transformations #######################
def softplus(x):
    return log(exp(x)+1)

def softplus_inv(x):
    return log(exp(x)-1)

def grad_softplus(x):
    return exp(x)/(exp(x)+1)

def logistic(x):
    return 1/(1+exp(-x))

def logistic_inv(x):
    return log(x) - log(1-x)

def grad_logistic(x):
    return logistic(x)*(1-logistic(x))

def geq_one(x):
    return softplus(x) + 1

def geq_one_inv(x):
    return softplus_inv(x-1)

def grad_geq_one(x):
    return grad_softplus(x)

def geq_two(x):
    return softplus(x) + 2

def geq_two_inv(x):
    return softplus_inv(x-2)

def grad_geq_two(x):
    return grad_softplus(x)

eps = 1e-2
def geq_eps(x):
    return softplus(x) + eps

def geq_eps_inv(x):
    return softplus_inv(x-eps)

def grad_geq_eps(x):
    return grad_softplus(x)
    
####################################################################

def array_list_to_vec(l):
    n_arrays = len(l)
    vec = np.concatenate([np.reshape(l[i],np.prod(l[i].shape)) for i in range(n_arrays)])
    return vec

def vec_to_array_list(vec,dims):
    n_arrays = len(dims)
    array_list = []
    idx = 0
    for i in range(n_arrays):
        dim = dims[i]
        dim_prod = np.prod(dim)
        array_list.append(np.reshape(vec[idx:idx+dim_prod],dim))
        idx += dim_prod
    return array_list

def theta_to_vec(theta):
    return array_list_to_vec(theta)

def vec_to_theta(vec):
    Nreg = (len(vec)-4) // 4
    return vec_to_array_list(vec,[[Nreg,4],[4]])

theta_t_var =     {'t0':{'t':lambda x:x,'t_inv':lambda x:x,'dt':lambda x:1.},
                   'N':{'t':softplus,'t_inv':softplus_inv,'dt':grad_softplus},
                   'k':{'t':geq_two,'t_inv':geq_two_inv,'dt':grad_geq_two},
                   'theta':{'t':geq_eps,'t_inv':geq_eps_inv,'dt':grad_geq_eps},
                   'tauphi':{'t':exp,'t_inv':log,'dt':exp},
                   'lbdphi':{'t':logistic,'t_inv':logistic_inv,'dt':grad_logistic},
                   'siga':{'t':exp,'t_inv':log,'dt':exp},
                   'sigm':{'t':exp,'t_inv':log,'dt':exp}
                   }

def theta_t(theta_hat):
    theta = [np.zeros_like(theta_hat[0]),np.zeros_like(theta_hat[1])]
    theta[0][:,0] = theta_t_var['t0']['t'](theta_hat[0][:,0])
    theta[0][:,1] = theta_t_var['N']['t'](theta_hat[0][:,1])
    theta[0][:,2] = theta_t_var['k']['t'](theta_hat[0][:,2])
    theta[0][:,3] = theta_t_var['theta']['t'](theta_hat[0][:,3])
    theta[1][:] = np.array([theta_t_var['tauphi']['t'](theta_hat[1][0]),
                            theta_t_var['lbdphi']['t'](theta_hat[1][1]),
                            theta_t_var['siga']['t'](theta_hat[1][2]),
                            theta_t_var['sigm']['t'](theta_hat[1][3])])
    return theta
    
def theta_t_inv(theta):
    theta_hat = [np.zeros_like(theta[0]),np.zeros_like(theta[1])]
    theta_hat[0][:,0] = theta_t_var['t0']['t_inv'](theta[0][:,0])
    theta_hat[0][:,1] = theta_t_var['N']['t_inv'](theta[0][:,1])
    theta_hat[0][:,2] = theta_t_var['k']['t_inv'](theta[0][:,2])
    theta_hat[0][:,3] = theta_t_var['theta']['t_inv'](theta[0][:,3])
    theta_hat[1][:] = np.array([theta_t_var['tauphi']['t_inv'](theta[1][0]),
                            theta_t_var['lbdphi']['t_inv'](theta[1][1]),
                            theta_t_var['siga']['t_inv'](theta[1][2]),
                            theta_t_var['sigm']['t_inv'](theta[1][3])])
    return theta_hat

#################################### VI-specific ####################################
def sigma(rho):
    return softplus(rho)

def sigma_inv(sig):
    return softplus_inv(sig)

def grad_sigma(rho):
    return grad_softplus(rho)

def phi_hat_to_phi(phi_hat):
    phi = [np.copy(phi_hat[i]) for i in range(len(phi_hat))]
    for i in range(len(phi_hat)):
        phi[i][...,1] = sigma(phi[i][...,1])
    return phi
    
def phi_to_phi_hat(phi):
    phi_hat = [np.copy(phi[i]) for i in range(len(phi))]
    for i in range(len(phi)):
        phi_hat[i][...,1] = sigma_inv(phi_hat[i][...,1])
    return phi_hat

def phi_to_vec(phi):
    return array_list_to_vec(phi)

def vec_to_phi(vec):
    Nreg = (len(vec)-8) // 4 // 2
    return vec_to_array_list(vec,[[Nreg,4,2],[4,2]])

# Transform standard normal samples to theta samples from mean-field surrogate posterior
def t(eps,phi):
    thetas = [phi[i][...,0] + phi[i][...,1]*eps[i] for i in range(len(eps))]
    return thetas
