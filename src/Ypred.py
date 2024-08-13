import numpy as np
from utils import fGamma, F_LN
from transforms import theta_t

# Model predictions for all regions at a set of times
# Prediction a 2D array (Num regions x Num times)
# Prediction array is computed in parallel as (N_region_blocks) block matrix
# theta is (Nregions x 4), columns are (t0, N, qshape, qscale)

def quad_pred(theta, nquad, params, t):
    inc_mu          = params['incubation_median']
    inc_sigma       = params['incubation_sigma']
    
    Nreg = theta.shape[0]
    Nt = t.shape[0]
    x_quad,w_quad = np.polynomial.legendre.leggauss(nquad) # x in [-1,1]
    preds = np.zeros((Nreg,Nt))
    for r in range(Nreg):
        t0, N, k, theta_scale = theta[r]
        people_with_symptoms = np.zeros_like(t)
        for d in range(Nt):
            c = 0.5*(t[d]-t0)
            if c > 0:
                taui = c*(x_quad + 1) + t0
                wi = c*w_quad
                integrand_taui = fGamma(taui-t0,k,theta_scale)*(F_LN(t[d]-taui,inc_mu,inc_sigma)-F_LN((t[d]-1)-taui,inc_mu,inc_sigma))
                people_with_symptoms = np.dot(wi,integrand_taui)
                preds[r,d] = N*people_with_symptoms
    return preds

def Ypred(theta_hat, params, forecast=False, transform=True):
    # Parameters
    nquad           = params['nquad_pred']
    if transform:
        theta           = theta_t(theta_hat)[0]
    else:
        theta           = theta_hat[0]
        
    if forecast:
        preds = quad_pred(theta,nquad,params,np.concatenate([params['t'],params['t_forecast']]))
    else:
        preds = quad_pred(theta,nquad,params,params['t'])
    return preds
                