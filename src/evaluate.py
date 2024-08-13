import sys, os 
import numpy as np
import matplotlib.pyplot as plt
import ray
from utils import load_obj, save_obj, thetas_rshp, theta_samples
from Ypred import Ypred
from pushforward import Ypred_pushforward, fGamma_pushforward, Ypred_and_fGamma_pf, Ypred_posterior_predictive
from transforms import theta_to_vec, vec_to_theta, theta_t, theta_t_var, theta_t_inv, phi_to_vec, vec_to_phi, sigma, sigma_inv, phi_to_phi_hat, phi_hat_to_phi

def evaluate_MLE(data_dir='',params_fname='params.pkl',params=None):
    if params is None:
        params = load_obj('{d}{f}'.format(d=data_dir,f=params_fname))
    # Plot MLE results
    MLE_log_header='{d}MLE_'.format(d=data_dir)
    # MLE_log_header='MLE_'
    fig, ax = plt.subplots(1,2,figsize=(10,4))
    grad_hist_vec = np.load('{h}grad_hist.npy'.format(h=MLE_log_header))
    grad_hist = list(map(vec_to_theta,grad_hist_vec))
    grad_norm_hist = list(map(np.linalg.norm,grad_hist_vec))
    grad_1_norm_hist = list(map(np.linalg.norm,[grad_hist[i][0] for i in range(len(grad_hist))]))
    grad_2_norm_hist = list(map(np.linalg.norm,[grad_hist[i][1] for i in range(len(grad_hist))]))
    obj_hist = np.load('{h}obj_hist.npy'.format(h=MLE_log_header))
    its = np.load('{h}its.npy'.format(h=MLE_log_header))
    ax[0].scatter(its,obj_hist)
    ax[0].set_title('Log Likelihood vs iteration')
    ax[0].set_ylabel('Log Likelihood')
    ax[0].set_xlabel('Gradient descent iteration')
    ax[1].plot(grad_norm_hist)
    ax[1].plot(grad_1_norm_hist)
    ax[1].plot(grad_2_norm_hist)
    ax[1].legend(['Full grad','Grad model','Grad noise'])
    ax[1].set_ylabel('Gradient norm')
    ax[1].set_xlabel('Gradient descent iteration')
    ax[1].set_yscale('log')
    plt.savefig('ll_and_gradient.pdf',bbox_inches='tight')

    theta_vec_hist = np.load('{h}theta_hist.npy'.format(h=MLE_log_header))
    n_epochs = theta_vec_hist.shape[0]
    theta_hat_hist = []
    theta_hist = []
    for i in range(n_epochs):
        theta_i = vec_to_theta(theta_vec_hist[i])
        theta_hat_hist.append(theta_i)
        theta_hist.append(theta_t(theta_i))
    theta_1_hist = np.array([theta_hist[i][0] for i in range(n_epochs)])
    theta_2_hist = np.array([theta_hist[i][1] for i in range(n_epochs)])
    grad_theta_1_hist = np.array([grad_hist[i][0] for i in range(n_epochs)])
    grad_theta_2_hist = np.array([grad_hist[i][1] for i in range(n_epochs)])


    if params['Nreg'] == 1:
        nrows = 2
    else:
        nrows = params['Nreg']
    fig, ax = plt.subplots(nrows,5,figsize=(30,5*(nrows)))
    if params['Nreg'] == 1:
        for i in range(5):
            ax[1,i].axis('off')
                
    pad = 5 
    for a, county in zip(ax[:,0], params['counties']):
        a.annotate(county, xy=(0, 0.5), xytext=(-a.yaxis.labelpad - pad, 0),
                    xycoords=a.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
    n_plots = 10
    plt_num = 0
    alpha_min = 0.2
    alpha_max = 0.8
    epochs = list(map(int,np.linspace(0,n_epochs-1,n_plots)))
    def alpha_map(x):
        return (x - (1/n_plots))*(alpha_max-alpha_min) + alpha_min
    for r in range(params['Nreg']):
        ax[r,0].plot(params['daily_counts'][r],color='black')
        ax[r,0].legend(['True daily counts'])
        ax[r,0].set_xlabel('Days since day zero')
        ax[r,0].set_ylabel('Normalized daily counts')
        plt_num = 0
        for i in epochs:
            plt_num += 1
            Y_pred = Ypred(theta_hat_hist[i],params,forecast=True)
            if plt_num == n_plots:
                ax[r,0].plot(Y_pred[r,:],color='red',alpha=alpha_map(plt_num/n_plots))
            else:
                ax[r,0].plot(Y_pred[r,:],color='blue',alpha=alpha_map(plt_num/n_plots))

        for i in range(1,4):
            ax[r,1].plot(np.arange(n_epochs),theta_1_hist[:,r,i])
        ax[r,1].legend([r'$N$',r'$k$',r'$\theta$'])
        ax[r,1].set_yscale('log')
        ax[r,1].set_xlabel('Epoch')
        ax[r,1].set_label('Parameter convergence')

        for i in range(1,4):
            ax[r,2].plot(np.arange(n_epochs),np.abs(grad_theta_1_hist[:,r,i]))
        ax[r,2].legend([r'$N$',r'$k$',r'$\theta$'])
        ax[r,2].set_yscale('log')
        ax[r,2].set_xlabel('Epoch')
        ax[r,2].set_label('Parameter Gradient')

        ax[r,3].plot(np.arange(n_epochs),theta_1_hist[:,r,0])
        ax[r,3].set_xlabel('Epoch')
        ax[r,3].legend([r'$t_0$'])

        ax[r,4].plot(np.arange(n_epochs),np.abs(grad_theta_1_hist[:,r,0]))
        ax[r,4].set_xlabel('Epoch')
        ax[r,4].legend([r'$t_0$'])
    
    ax[0,0].set_title('Model convergence')
    ax[0,1].set_title('Parameter convergence')
    ax[0,2].set_title('Parameter Gradient')
    ax[0,3].set_title('Parameter convergence')
    ax[0,4].set_title('Parameter Gradient')
    fig.savefig('convergence.pdf',bbox_inches='tight')

    fig, ax = plt.subplots(1,2,figsize=(10,4))
    for i in range(4):
        ax[0].plot(np.arange(n_epochs),theta_2_hist[:,i])
        ax[1].plot(np.arange(n_epochs),np.abs(grad_theta_2_hist[:,i]))
    for i in range(2):
        ax[i].set_xlabel('Epoch')
        ax[i].set_ylabel('Log value')
        ax[i].legend([r'$\tau_{\Phi}$',r'$\lambda_{\Phi}$',r'$\sigma_a$',r'$\sigma_m$'])
        ax[i].set_yscale('log')
    ax[0].set_title('Noise parameter value')
    ax[1].set_title('Noise parameter gradient')
    plt.savefig('noise_convergence.pdf',bbox_inches='tight')

    
def evaluate_VI(data_dir='',params_fname='params.pkl',params=None):
    if params is None:
        params = load_obj('{d}{f}'.format(d=data_dir,f=params_fname))
    nepochs_take = params['plot_VI_nepochs']
    vi_log_header = '{d}VI_'.format(d=data_dir)
    
    ############## Load data ##############
    phi_hat_vec_hist = np.load('{h}theta_hist.npy'.format(h=vi_log_header))
    if not(nepochs_take == 'all'):
        phi_hat_vec_hist = phi_hat_vec_hist[:nepochs_take]
    n_epochs = phi_hat_vec_hist.shape[0]
    phi_hat_hist = []
    phi_hist = []
    theta_mu_hist = []
    for i in range(n_epochs):
        phi_hat_i = vec_to_phi(phi_hat_vec_hist[i])
        phi_hat_hist.append(phi_hat_i)
        phi_i = phi_hat_to_phi(phi_hat_i)
        phi_hist.append(phi_i)
        theta_mu_hist.append(theta_t([phi_i[0][...,0],phi_i[1][...,0]]))
    phi_1_hist = np.array([phi_hist[i][0] for i in range(n_epochs)])
    phi_2_hist = np.array([phi_hist[i][1] for i in range(n_epochs)])
    theta_mu_1_hist = np.array([theta_mu_hist[i][0] for i in range(n_epochs)])
    theta_mu_2_hist = np.array([theta_mu_hist[i][1] for i in range(n_epochs)])
    theta_hat_sigma_1 = np.array([phi_hist[i][0][...,1] for i in range(n_epochs)])
    theta_hat_sigma_2 = np.array([phi_hist[i][1][...,1] for i in range(n_epochs)])

    grad_hist_vec = np.load('{h}grad_hist.npy'.format(h=vi_log_header))
    obj_hist = np.load('{h}obj_hist.npy'.format(h=vi_log_header))
    its = np.load('{h}its.npy'.format(h=vi_log_header))
    if not(nepochs_take == 'all'):
        grad_hist_vec = grad_hist_vec[:nepochs_take]
    grad_hist = list(map(vec_to_phi,grad_hist_vec))
    grad_theta_1_hist = np.array([grad_hist[i][0] for i in range(n_epochs)])
    grad_theta_2_hist = np.array([grad_hist[i][1] for i in range(n_epochs)])
    
    ############## Plot VI results ##############

    #################################### Objectives ####################################
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    grad_hist = list(map(vec_to_phi,grad_hist_vec))
    grad_norm_hist = list(map(np.linalg.norm,grad_hist_vec))
    grad_1_norm_hist = list(map(np.linalg.norm,[grad_hist[i][0] for i in range(len(grad_hist))]))
    grad_2_norm_hist = list(map(np.linalg.norm,[grad_hist[i][1] for i in range(len(grad_hist))]))
    ax[0].scatter(its,obj_hist)
    ax[0].set_title('ELBO vs iteration')
    ax[0].set_ylabel('ELBO')
    ax[0].set_xlabel('Gradient descent iteration')
    ax[1].plot(grad_norm_hist)
    ax[1].plot(grad_1_norm_hist)
    ax[1].plot(grad_2_norm_hist)
    ax[1].legend(['Full grad','Grad model','Grad noise'])
    ax[1].set_ylabel('Gradient norm')
    ax[1].set_xlabel('Gradient descent iteration')
    ax[1].set_yscale('log')
    plt.savefig('ELBO_and_gradient.pdf',bbox_inches='tight')
    
    #################################### Y_pred Pushforward ####################################
    if params['plot_Ypred_pf']:
    
        if params['Nreg'] == 1:
            nrows = 2
        else:
            nrows = params['Nreg']
        fig, ax = plt.subplots(nrows,1,figsize=(7,5*nrows))
        pad = 5 
        for a, county in zip(ax[:], params['counties']):
            a.annotate(county, xy=(0, 0.5), xytext=(-a.yaxis.labelpad - pad, 0),
                        xycoords=a.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')

        Ns = params['Ns_pf']
        Y_preds, thetas = Ypred_pushforward(phi_hat_hist[-1],params,Ns, forecast=True)
        np.save('Y_pred_pf.npy',Y_preds)
        save_obj(thetas,'thetas_pf.pkl')
        ax[0].set_title(r'VI $Y_{pred}$ Pushforward')
        for r in range(params['Nreg']):
            ax[r].set_xlabel('Days since day zero')
            ax[r].set_ylabel('Daily counts')
            plt_num = 0
            for i in range(Ns):
                Y_pred = Y_preds[i]
                ax[r].plot(params['populations'][r]*Y_pred[r,:],color='blue',alpha=0.02)
            Y_pred_mean = params['populations'][r]*Ypred([phi_hat_hist[-1][0][...,0],phi_hat_hist[-1][1][...,0]],params, forecast=True)
            ax[r].plot(Y_pred_mean[r,:],color='red',label='mean')
            ax[r].scatter(params['t'],params['populations'][r]*params['daily_counts'][r],marker='x',color='black',label='true')
            ax[r].scatter(params['t_forecast'],params['populations'][r]*params['daily_counts_forecast'][r],marker='x',color='orange',label='true forecast')
            ax[r].legend()
        fig.savefig('VI_Ypred_pushforward.png',bbox_inches='tight')

    #################################### fGamma Pushforward ####################################
    if params['plot_fGamma_pf']:
        if params['Nreg'] == 1:
            nrows = 2
        else:
            nrows = params['Nreg']
        fig, ax = plt.subplots(nrows,1,figsize=(7,5*nrows))
        pad = 5 
        for a, county in zip(ax[:], params['counties']):
            a.annotate(county, xy=(0, 0.5), xytext=(-a.yaxis.labelpad - pad, 0),
                        xycoords=a.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')

        Ns = params['Ns_pf']
        fGamma_mean, fGamma_vals, x_vals = fGamma_pushforward(phi_hat_hist[-1],params,Ns,forecast=True)
        np.save('fGamma_mean.npy',fGamma_mean)
        np.save('fGamma_vals.npy',fGamma_vals)
        np.save('fGamma_xvals.npy',x_vals)
        ax[0].set_title(r'VI $f_\gamma$ Pushforward')
        for r in range(params['Nreg']):
            ax[r].set_xlabel('Days since day zero')
            ax[r].set_ylabel('Daily counts')
            plt_num = 0
            for i in range(Ns):
                ax[r].plot(x_vals, params['populations'][r]*fGamma_vals[i,r,:],color='blue',alpha=0.02)
            ax[r].plot(x_vals, params['populations'][r]*fGamma_mean[r,:],color='red',label='mean')
            ax[r].scatter(params['t'],params['populations'][r]*params['daily_counts'][r],marker='x',color='black',label='true')
            ax[r].scatter(params['t_forecast'],params['populations'][r]*params['daily_counts_forecast'][r],marker='x',color='orange',label='true forecast')
            ax[r].legend()
        fig.savefig('VI_fGamma_pushforward.png',bbox_inches='tight')

    #################################### Parameter convergence ####################################
    if params['plot_VI_params']:
        if params['Nreg'] == 1:
            nrows = 2
        else:
            nrows = params['Nreg']
        fig, ax = plt.subplots(nrows,5,figsize=(35,5*nrows))

        pad = 5 
        for a, county in zip(ax[:,0], params['counties']):
            a.annotate(county, xy=(0, 0.5), xytext=(-a.yaxis.labelpad - pad, 0),
                        xycoords=a.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')
        n_plots = 10
        plt_num = 0
        alpha_min = 0.2
        alpha_max = 0.8
        def alpha_map(x):
            return (x - (1/n_plots))*(alpha_max-alpha_min) + alpha_min
        for r in range(params['Nreg']):
            ax[r,0].plot(params['daily_counts'][r],color='black')
            ax[r,0].legend(['True daily counts'])
            ax[r,0].set_xlabel('Days since day zero')
            ax[r,0].set_ylabel('Normalized daily counts')
            plt_num = 0
            for i in range(n_epochs):
                if i%int(n_epochs/n_plots) == 0: 
                    plt_num += 1
                    Y_pred = Ypred([phi_hat_hist[i][0][...,0],phi_hat_hist[i][1][...,0]],params)
                    if plt_num == n_plots:
                        ax[r,0].plot(Y_pred[r,:],color='red')
                    else:
                        ax[r,0].plot(Y_pred[r,:],color='blue',alpha=alpha_map(plt_num/n_plots))
            Y_pred = Ypred([phi_hat_hist[-1][0][...,0],phi_hat_hist[-1][1][...,0]],params)
            ax[r,0].plot(Y_pred[r,:],color='red')
            for i in range(4):
                ax[r,1].plot(np.arange(n_epochs),theta_mu_1_hist[:,r,i])
            ax[r,1].legend([r'$t_0$',r'$N$',r'$k$',r'$\theta$'])
            ax[r,1].set_yscale('log')
            ax[r,1].set_xlabel('Epoch')
            ax[r,1].set_ylabel(r'$\mu$')

            for i in range(4):
                ax[r,2].plot(np.arange(n_epochs),np.abs(grad_theta_1_hist[:,r,i,0]))
            ax[r,2].legend([r'$t_0$',r'$N$',r'$k$',r'$\theta$'])
            ax[r,2].set_yscale('log')
            ax[r,2].set_xlabel('Epoch')
            ax[r,2].set_label('Parameter Gradient')

            for i in range(4):
                ax[r,3].plot(np.arange(n_epochs),theta_hat_sigma_1[:,r,i])
            ax[r,3].legend([r'$t_0$',r'$N$',r'$k$',r'$\theta$'])
            ax[r,3].set_yscale('log')
            ax[r,3].set_xlabel('Epoch')
            ax[r,3].set_ylabel(r'$\sigma$')

            for i in range(4):
                ax[r,4].plot(np.arange(n_epochs),np.abs(grad_theta_1_hist[:,r,i,1]))
            ax[r,4].legend([r'$t_0$',r'$N$',r'$k$',r'$\theta$'])
            ax[r,4].set_yscale('log')
            ax[r,4].set_xlabel('Epoch')
            ax[r,4].set_label('Parameter Gradient')


        ax[0,0].set_title('Model convergence')
        ax[0,1].set_title(r'Parameter $\mu$')
        ax[0,2].set_title(r'Parameter $\mu$ gradient')
        ax[0,3].set_title(r'Parameter $\sigma$') 
        ax[0,4].set_title(r'Parameter $\sigma$ gradient') 
        fig.savefig('VI_convergence.png',bbox_inches='tight')


    #################################### Noise ####################################
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    plt.figure()
    for i in range(4):
        ax[0].plot(np.arange(n_epochs),theta_mu_2_hist[:,i])
        sigma_hist = [phi_hist[j][1][i,1] for j in range(len(phi_hist))]
        ax[1].plot(np.arange(n_epochs),sigma_hist)
    ax[0].legend([r'$\tau_{\Phi}$',r'$\lambda_{\Phi}$',r'$\sigma_a$',r'$\sigma_m$'])
    ax[1].legend([r'$\tau_{\Phi}$',r'$\lambda_{\Phi}$',r'$\sigma_a$',r'$\sigma_m$'])
    ax[0].set_xlabel('Epoch')
    ax[1].set_ylabel('Epoch')
    ax[0].set_title(r'Noise params $\mu$')
    ax[1].set_title(r'Noise params $\sigma$')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    fig.savefig('VI_noise_convergence.png',bbox_inches='tight')

    
# Prepare results for paper
def prepare_results(data_dir='',save_dir='',names={},params_fname='params.pkl',epoch=-1,noise_type='coupled'):
    params = load_obj('{d}{f}'.format(d=data_dir,f=params_fname))
    M = 100    # Number of pushforward posterior runs
    N = int(1e5)
    K = 4 # No. params per county
    # No. noise params
    if noise_type == 'coupled':
        k = 4
    elif noise_type == 'independent':
        k = 2
        
    ############################ Load data ############################
    vi_log_header = '{d}VI_'.format(d=data_dir)
    phi_hat_vec_hist = np.load('{h}theta_hist.npy'.format(h=vi_log_header))
    phi_hat_final = vec_to_phi(phi_hat_vec_hist[epoch])
    phi_final = phi_hat_to_phi(phi_hat_final)
    theta_hat_mu_final = [phi_final[0][...,0],phi_final[1][...,0]]
    theta_mu_final = theta_t(theta_hat_mu_final)
    ############################ Prepare data for paper ############################
    # data_types = ['forecast','infection','ppparams','posterior']
    data_types = ['forecast','infection','mean_and_std']
    counties = params['counties']
    Ndays = params['t'].shape[0] + params['t_forecast'].shape[0]
    t_forecast = np.concatenate([params['t'],params['t_forecast']])
    dates_forecast = np.concatenate([params['dates'],params['dates_forecast']])
    true_counts = np.hstack([params['daily_counts'],params['daily_counts_forecast']])
    for i in range(len(params['counties'])):
        true_counts[i,:] *= params['populations'][i]
    _, fvals, _, _ = Ypred_and_fGamma_pf(phi_hat_final,params,Ns=M,Ns_theta=N,forecast=True)
    thetas, Y_preds_pf, Y_preds_pp = Ypred_posterior_predictive(phi_hat_final,params,Ns=M,forecast=True)
    Y_preds = Y_preds_pp

    # Compute a large sample of thetas for computing means, std
    Ns_large = 500
    theta_hat_samples, _ = theta_samples(phi_final,params,Ns=Ns_large)
    theta_hat_samples_rshp = []
    for i in range(Ns_large):
        theta_hat_samples_rshp.append([theta_hat_samples[0][i,...],theta_hat_samples[1][i,...]])
    thetas_large = list(map(theta_t,theta_hat_samples_rshp))
    thetas_large = thetas_rshp(thetas_large)
    thetas_region_large = thetas_large[0]
    thetas_noise_large = thetas_large[1]
    means_region = np.mean(thetas_region_large,axis=0)
    stds_region = np.std(thetas_region_large,axis=0)
    means_noise = np.mean(thetas_noise_large,axis=0)
    stds_noise = np.std(thetas_noise_large,axis=0)
    
    # Unnormalize by populations
    for r in range(params['Nreg']):
        # Y_pred_mu[r,:] *= params['populations'][r]
        Y_preds[:,r,:] *= params['populations'][r]
    
    if not(os.path.isdir(save_dir)):
        os.mkdir(save_dir)
    
    if names == {}:
        for county in params['counties']:
            names[county] = county
     
    for i,county in enumerate(counties):
        for data_type in data_types:
            if data_type == 'forecast':
                X_forecast = np.empty([Ndays,M+2],dtype=object)
                # Pushforward predcitions
                X_forecast[:,1:M+1] = (Y_preds[:,i,:].T).astype(str)
                # Last column - true counts
                X_forecast[:,-1] = true_counts[i,:].astype(str)
                # 1st column - dates
                X_forecast[:,0] = dates_forecast
                np.savetxt("{d}{county}_{dtype}.csv".format(d=save_dir,county=names[county],dtype=data_type),X_forecast,delimiter=",",fmt='%s')
            elif data_type == 'infection':
                X_infection = np.empty([Ndays,M+2],dtype=object)
                # Pushforward predcitions
                X_infection[:,1:M+1] = (fvals[:,i,:].T).astype(str)
                # Last column - true counts
                X_infection[:,-1] = true_counts[i,:].astype(str)
                # 1st column - dates
                X_infection[:,0] = dates_forecast
                np.savetxt("{d}{county}_{dtype}.csv".format(d=save_dir,county=names[county],dtype=data_type),X_infection,delimiter=",",fmt='%s')
            elif data_type == 'ppparams':
                ppparams = np.zeros([M,K+k])
                for j in range(M):
                    ppparams[j,:K] = thetas[j][0][i,:]
                    if noise_type == 'coupled':
                        ppparams[j,K:] = thetas[j][1]
                    elif noise_type == 'independent':
                        ppparams[j,K:] = thetas[j][1][2:]
                np.savetxt("{d}{county}_{dtype}.csv".format(d=save_dir,county=names[county],dtype=data_type),ppparams,delimiter=",",fmt='%s')
            elif data_type == 'posterior':
                posterior = np.zeros([N,K+k])
                for j in range(M):
                    posterior[j,:K] = thetas[j][0][i,:]
                    if noise_type == 'coupled':
                        posterior[j,K:] = thetas[j][1]
                    elif noise_type == 'independent':
                        posterior[j,K:] = thetas[j][1][2:]
                np.savetxt("{d}{county}_{dtype}.csv".format(d=save_dir,county=names[county],dtype=data_type),posterior,delimiter=",",fmt='%s')
            elif data_type == 'mean_and_std':
                mean_csv = np.empty([2,8],dtype=object)
                mean_csv[0] = np.array(['t0','N','k','theta','tau','lambda','siga','sigm'])
                mean_csv[1][:4] = means_region[i]
                mean_csv[1][4:] = means_noise
                
                std_csv = np.empty([2,8],dtype=object)
                std_csv[0] = np.array(['t0','N','k','theta','tau','lambda','siga','sigm'])
                std_csv[1][:4] = stds_region[i]
                std_csv[1][4:] = stds_noise
                
                np.savetxt("{d}{county}_{dtype}.csv".format(d=save_dir,county=names[county],dtype='mean'),mean_csv,delimiter=",",fmt='%s')
                np.savetxt("{d}{county}_{dtype}.csv".format(d=save_dir,county=names[county],dtype='std'),std_csv,delimiter=",",fmt='%s')
                
    return "Done!"