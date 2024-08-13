import sys, os 
import numpy as np
import matplotlib.pyplot as plt
import ray
from vi import adam_MLE, adam_VI, adam_bbox_VI
from utils import load_obj
from Ypred import Ypred
from transforms import theta_to_vec, vec_to_theta, theta_t, theta_t_var, theta_t_inv, phi_to_vec, vec_to_phi, sigma, sigma_inv, phi_to_phi_hat, phi_hat_to_phi
from evaluate import evaluate_MLE, evaluate_VI

################################### Params ###################################
params = load_obj('params.pkl')
print('Parameters for VI run with counties:')
print(params['counties'],'\n\n')
for key in params.keys():
    if not key == 'daily_counts':
        print(key,params[key])
print('\n\n')

################################### MLE ###################################
if params['run_MLE']:
    theta_0 = params['theta_0']
    theta_hat_0 = theta_t_inv(theta_0)

    plt.figure()
    for i,key in enumerate(params['counties']):  
        plt.plot(params['daily_counts'][i])
    Y_pred = Ypred(theta_hat_0,params)
    for i in range(params['Nreg']):  
        plt.plot(Y_pred[i],color='black')
    plt.legend(params['counties'])
    plt.title(r'Day$_0$: {0}, Day$_f$: {1},  n. days: {2}'.format(params['day0'],params['dayf'],params['t'].shape[0]))
    plt.xlabel(r'Days since Day$_0$')
    plt.savefig('theta_0.pdf',bbox_inches='tight')

    def alpha_schedule(it,max_its):
        return params['MLE_alpha_0']
    theta_f,obj_hist,theta_hist, grad_norm_hist = adam_MLE( theta_to_vec(theta_hat_0),
                                                            params=params,
                                                            max_its=params['MLE_max_its'],
                                                            tol=params['MLE_tol'],
                                                            alpha_0=params['MLE_alpha_0'],
                                                            alpha_schedule=alpha_schedule,
                                                            log_freq=params['MLE_log_freq'],
                                                            log_header=params['MLE_log_header'],
                                                            grad_model=True,
                                                            grad_noise=True
                                                          )
    print('Done with MLE!\n\n',flush=True)

if params['plot_MLE']:
    print('Plotting MLE results...')
    evaluate_MLE()
################################### VI ###################################
if params['run_VI']:
    ray.init(num_cpus = params['num_cpus'])

    # Warmstart IC
    theta_vec_hist = np.load('{h}theta_hist.npy'.format(h=params['MLE_log_header']))
    n_epochs = theta_vec_hist.shape[0]
    theta_hat_hist = []
    theta_hist = []
    for i in range(n_epochs):
        theta_i = vec_to_theta(theta_vec_hist[i])
        theta_hat_hist.append(theta_i)
        theta_hist.append(theta_t(theta_i))
    theta_1_hist = np.array([theta_hist[i][0] for i in range(n_epochs)])
    theta_2_hist = np.array([theta_hist[i][1] for i in range(n_epochs)])
    
    if 'VI_warmstart_epoch' in params.keys():
        warmstart_epoch = params['VI_warmstart_epoch']
    else:
        warmstart_epoch = -1
    phi_0_mu = [np.copy(theta_hat_hist[warmstart_epoch][0]),np.copy(theta_hat_hist[warmstart_epoch][1])]
    print('Started VI from MLE epoch: {0}'.format(warmstart_epoch))

    sig_vi = 1e-2
    phi_0_sig = [sig_vi*np.ones([params['Nreg'],4]),sig_vi*np.ones(4)]
    phi_0 = [np.concatenate([np.expand_dims(phi_0_mu[0],axis=2),np.expand_dims(phi_0_sig[0],axis=2)],axis=2),
             np.concatenate([np.expand_dims(phi_0_mu[1],axis=1),np.expand_dims(phi_0_sig[1],axis=1)],axis=1)]
    phi_hat_0 = phi_to_phi_hat(phi_0)

    def alpha_schedule(it,max_its):
        return params['VI_alpha_0']
    if params['VI_type'] == 'pathwise':
        VI = adam_VI
    elif params['VI_type'] == 'black_box':
        VI = adam_bbox_VI
    phi_hat_f,obj_hist,phi_hat_hist, grad_norm_hist = VI(   phi_hat_0=phi_to_vec(phi_hat_0),
                                                            params=params,
                                                            max_its=params['VI_max_its'],
                                                            tol=params['VI_tol'],
                                                            alpha_0=params['VI_alpha_0'],
                                                            alpha_schedule=alpha_schedule,
                                                            log_freq=params['VI_log_freq'],
                                                            log_header=params['VI_log_header'],
                                                            Ns_grad=params['VI_Ns_grad'],
                                                            Ns_obj=params['VI_Ns_obj'])

if params['plot_VI']:
    print('Plotting VI results...')
    evaluate_VI()