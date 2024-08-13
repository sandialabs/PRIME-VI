import numpy as np
from numpy.linalg import norm
from utils import Timer

# Performs gradient descent on theta_0 with ADAM (Adaptive Moment Estimation)
def adam(theta_0,
         params,
         max_its,
         tol,
         obj_f,
         grad_f,
         alpha_0=0.001,
         alpha_schedule=None,
         obj_freq=1,
         log_freq=1,
         log_header=''):
    """Gradient descent via Adaptive Moment Estimation (ADAM)

    :param theta_0: initial parameters
    :type theta_0: np.ndarray
    :param params: additional model parameters
    :type params: dict
    :param max_its: max number of iterations
    :type max_its: int
    :param tol: zero gradient tolerance
    :type tol: float
    :param obj_f: objective function
    :type obj_f: function
    :param grad_f: gradient function
    :type grad_f: function
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
    :return: optimized parameters theta
    :rtype: np.ndarray
    """
    # Parameters
    beta_1 = 0.9
    beta_2 = 0.999
    eps = 1e-8
    theta = theta_0
    dim = theta.shape[0]
    m,v,m_t,v_hat = np.zeros(dim),np.zeros(dim),np.zeros(dim),np.zeros(dim)
    alpha = alpha_0
    if alpha_schedule is None:
        alpha_schedule = lambda it,max_its:alpha_0
    t = 0
    grad_norm = 1.0
    
    theta_hist = [theta]
    grad_hist = [grad_f(theta)]
    grad_norm_hist = [norm(grad_hist[0])]
    obj_hist = [obj_f(theta)]
    its = [0]
    
    timer = Timer()
    grad_times = []
    obj_times = []
    
    while t < max_its and grad_norm > tol:
        t += 1
        timer.start()
        g = grad_f(theta)
        grad_time = timer.stop()
        grad_times.append(grad_time)
        g_norm = norm(g)
        m = beta_1*m + (1-beta_1)*g
        v = beta_2*v + (1-beta_2)*(g*g)
        m_hat = m/(1 - (beta_1**t))
        v_hat = v/(1 - (beta_2**t))
        alpha = alpha_schedule(t,max_its)
        theta = theta - alpha*m_hat/(v_hat**(0.5) + eps)
        
        theta_hist.append(theta)
        grad_norm_hist.append(g_norm)
        grad_hist.append(g)
        
        if t%log_freq == 0:
            its.append(t)
            timer.start()
            obj = obj_f(theta)
            obj_time = timer.stop()
            obj_times.append(obj_time)
            obj_hist.append(obj)
            # print('It: {0} Prog: %{3:1.1f} Obj: {1:.3e} Grad norm: {2:.3e}'.format(t,obj,g_norm,100*(t/max_its)),end='\r',flush=True)
            print('It: {0} Prog: %{3:1.1f} Obj: {1:.3e} Grad norm: {2:.3e}'.format(t,obj,g_norm,100*(t/max_its)),flush=True)
            print('Avg. grad comp. time: {0:.3f}. Avg obj comp. time: {1:.3f}\n'.format(np.mean(np.array(grad_times)),np.mean(np.array(obj_times))),flush=True)
            
            np.save('{h}theta_hist.npy'.format(h=log_header),np.array(theta_hist))
            np.save('{h}obj_hist.npy'.format(h=log_header),np.array(obj_hist))
            np.save('{h}grad_norm_hist.npy'.format(h=log_header),np.array(grad_norm_hist))
            np.save('{h}grad_hist.npy'.format(h=log_header),np.array(grad_hist))
            np.save('{h}its.npy'.format(h=log_header),np.array(its))
    return theta, np.array(obj_hist), np.array(theta_hist), np.array(grad_norm_hist)