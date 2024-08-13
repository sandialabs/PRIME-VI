import sys, os
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
# import pickle5 as pickle
import pickle
import time
from numpy import trace as tr, dot, diag, log, sqrt, power, exp
from numpy.linalg import inv
from scipy.special import digamma, erf
from scipy.stats import gamma,lognorm,norm
from transforms import t, phi_hat_to_phi, theta_t, vec_to_phi
import csv, json
from dateutil import parser

def normal_logpdf(x,loc,scale):
    return norm._logpdf((x-loc)/scale)-np.log(scale)

def lognorm_cdf(x,s,loc=0,scale=1):
    return lognorm.cdf(x,s,loc=loc,scale=scale)

def gamma_pdf(x,s,loc=0,scale=1):
    return gamma._pdf((x - loc)/scale,s)/scale

def fGamma(tau,k,theta):
    return np.where(tau >=0,gamma_pdf(tau,s=k,scale=theta),0)

def F_LN(tau,mu,sigma):
    return np.where(tau > 0,lognorm_cdf(tau,sigma,mu),0)

def get_F_LN_lims(inc_mu,inc_sigma,eps):
    x = np.linspace(0,30,2000)
    fx = F_LN(x,inc_mu,inc_sigma)-F_LN(x-1,inc_mu,inc_sigma)
    fx_filt = np.where(fx > eps,1,0)
    a = np.argwhere(fx_filt==1)
    lower = x[a[0,0]]
    upper = x[a[-1,0]]
    return [lower,upper]

def qform(x,A,y):
    return dot(x,A@y)

# Standard normal random samples
def std_normal_samples(params,Ns):
    Nreg = params['Nreg']
    eps = [randn(Ns,Nreg,4),randn(Ns,4)]
    return eps

# Transform standard normal samples to samples of surrogate posterior
def theta_samples(phi,params,Ns):
    eps = std_normal_samples(params,Ns)
    thetas = t(eps,phi)
    return thetas, eps

def thetas_rshp(thetas):
    if len(thetas) == 2:
        thetas_rshp = []
        n = thetas[0].shape[0]
        for i in range(n):
            thetas_rshp.append([thetas[0][i,...],thetas[1][i,...]])
        return thetas_rshp
    else:
        thetas_region = []
        thetas_noise = []
        for i in range(len(thetas)):
            thetas_region.append(thetas[i][0])
            thetas_noise.append(thetas[i][1])
        return [np.array(thetas_region),np.array(thetas_noise)]

def block_decompose_1D_tensor(n,n_blocks):
    block_size = n//n_blocks
    block_delims = [i*block_size for i in range(0,n_blocks)]
    block_delims.append(n)
    return block_delims

# Decompose tensor into tensor product block grid
def block_decompose_ND_tensor(tensor_shape, n_blocks):
    block_delims = np.array([block_decompose_1D_tensor(tensor_shape[i],n_blocks[i]) for i in len(n_blocks)])
    return block_delims

### Save/load pickle files ###
def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def _runningAvgWgts(nDays):
    '''
    Compute average weights
    '''
    disi = np.ones(nDays) / nDays
   
    ka = nDays // 2 + 1

    disb = []
    for i in range(ka,nDays):
        disb.append( np.ones(i) / i )

    return disi,disb

def runningAvg(f,nDays):
    r"""
    Apply nDays running average to the input f

    Parameters
    ----------
    f: numpy array
        array (with daily data for this project) to by filtered 
    nDays: int
        window width for the running average

    Returns
    -------
    favg: numpy array
        filtered data
    """
    disi,disb = _runningAvgWgts(nDays)
    ka = nDays // 2
    npts = f.shape[0]
    favg = np.empty_like(f)
    # interior
    for i in range(ka,favg.shape[0]-ka):
        favg[i] = np.sum(disi*f[i-ka:i+ka+1])
    # boundaries
    for i in range(ka):
        fwidth  = len(disb[i])
        favg[i] = np.sum(disb[i]*f[0:fwidth])
        favg[npts-1-i] = np.sum(disb[i]*f[npts-1:npts-fwidth-1:-1])
    return favg

def test_runningAvg():
    np.random.seed(2020)
    n = 100
    sig = np.random.randn(n)**3 + 3*np.random.randn(n).cumsum()
    plt.plot(sig,label='data')
    sigf = runningAvg(sig,3)
    plt.plot(sigf,label='3')
    sigf = runningAvg(sig,5)
    plt.plot(sigf,label='5')
    sigf = runningAvg(sig,7)
    plt.plot(sigf,label='7')
    sigf = runningAvg(sig,9)
    plt.plot(sigf,label='9')

    plt.legend()
    plt.show()


def prediction_filename(run_setup):
    r"""
    Generate informative name for hdf5 file with prediction data

    Parameters
    ----------
    run_setup: dictionary
        detailed settings for the epidemiological model

    Returns
    -------
    filename: string
        file name ending with a .h5 extension
    """
    fh5 = run_setup["ppopts"]["fpredout"]
    return fh5+".h5"

def output_epicurves(pred,daysPred,newcases,nskip,quantile_list,fileout):
    with open(fileout, mode='w') as output_file:
        csv_writer = csv.writer(output_file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        outputData=["#Date"]
        for qk in quantile_list:
            outputData = outputData+["quantile"+'%.3f'%(qk)]
        for j in range(1,pred.shape[0]+1,nskip):
            outputData = outputData+["sample"+str((j-1)//nskip+1)]
        outputData = outputData+["ConfirmedCases"]
        cso = csv_writer.writerow(outputData)
        ndaysData = len(newcases)
        ndaysPred = pred.shape[1]
        for i in range(ndaysPred):
            outputData = [daysPred[i].date()]
            for qk in quantile_list:
                outputData = outputData+["%d"%(np.quantile(pred,qk,axis=0)[i])]
            outputData = outputData+["%d"%(pred[j,i]) for j in range(0,pred.shape[0],nskip)]
            if i < ndaysData:
                outputData = outputData+["%d"%(newcases[i])]
            else:
                outputData = outputData+[-999]
            cso = csv_writer.writerow(outputData)
        output_file.close()

def output_infcurves(infc,datesmean,nskip,quantile_list,fileout):
    with open(fileout, mode='w') as output_file:
        csv_writer = csv.writer(output_file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        outputData=["#Date"]
        for qk in quantile_list:
            outputData = outputData+["quantile"+'%.3f'%(qk)]
        for j in range(1,infc.shape[0]+1,nskip):
            outputData = outputData+["sample"+str((j-1)//nskip+1)]
        cso = csv_writer.writerow(outputData)
        for i in range(len(datesmean)):
            outputData = [datesmean[i].date()]
            for qk in quantile_list:
                outputData = outputData+["%d"%(np.quantile(infc,qk,axis=0)[i])]
            outputData = outputData+["%d"%(infc[j,i]) for j in range(0,infc.shape[0],nskip)]
            cso = csv_writer.writerow(outputData)
        output_file.close()

def _linear_error_weight(min_wgt,days):
    '''
    compute linearly increasing weighting
    from  at first data point to 1.0 for most recent data point 
    '''
    ndays = len(days)
    return min_wgt + (1.0 - min_wgt)*np.arange(1,int(ndays)+1) / ndays

def _gaussian_error_weight(min_wgt,tau,days):
    '''
    compute semi-gaussian increasing weighting
    "mean" is at most recent data point. 
    Weight increases from min_wgt to 1
    '''
    day_max = np.max(days)
    return min_wgt + (1.0-min_wgt)*np.exp(-0.5 * ((days-day_max)/tau)**2)

def compute_error_weight(error_info,days):
    r"""
    Compute array with specified weighting for the daily cases data. 
    The weights follow either linear of Gaussian expressions with higher
    weights for recent data and lower weights for older data

    Parameters
    ----------
    error_info: list
        (error_type,min_wgt,[tau]), error type is either 'linear' or 'gaussian',
        min_wgt is the minimum weight and tau is the standard deviation 
        of the exponential term if a Gaussian formulation is chosen. 
    days: int
        lenght of the weights array
    Returns
    -------
    error_weight: numpy array
        array of weights
    """
    error_type = error_info[0]
    assert error_info[1] > 0.0, "error_weight second parameter needs to be positive"
    assert error_info[1] < 1.0, "error_weight second parameter needs to be less than 1.0"

    inv_error_weight = None
    if error_type=="linear":
        inv_error_weight = _linear_error_weight(error_info[1],days)
    elif error_type=="gaussian":
        if len(error_info) < 3:
            sys.exit("Need to specify minimum weight and width for 'gaussian'")
        inv_error_weight = _gaussian_error_weight(error_info[1],error_info[2],days)
    else:
        sys.exit("Only current options for error weighting are 'linear' or 'gaussian'")

    # compute error_weight from reciprocal
    error_weight = 1./inv_error_weight
    return error_weight

def get_opts(setupfile,verb=False,return_run_setup=False):

    run_setup=json.load(open(setupfile))
    if verb:
        print("=====================================================")
        print(run_setup)
        print("=====================================================")

    run_opts = dict()

    #-daily counts
    run_opts["count_data"] = run_setup["regioninfo"]["count_data"]
    run_opts["population_data"] = run_setup["regioninfo"]["population_data"]
    if "running_avg_obs" in run_setup["regioninfo"]:
        run_opts["running_avg_obs"] = run_setup["regioninfo"]["running_avg_obs"]

    run_opts["region_tag"] = run_setup["regioninfo"]["region_tag"]
    run_opts["day0"] = run_setup["regioninfo"]["day0"]

    #------------------------------------------------------------------
    #-incubation model
    assert "num_waves" in run_setup["modelopts"]
    run_opts["num_waves"] = run_setup["modelopts"]["num_waves"]
    run_opts["useconv"] = run_setup["modelopts"]["useconv"]
    run_opts["inc_median"] = run_setup["modelopts"]["incubation_median"]
    run_opts["inc_sigma"] = run_setup["modelopts"]["incubation_sigma"]

    if "incubation_model" in run_setup["modelopts"]:
        run_opts["inc_model"] = run_setup["modelopts"]["incubation_model"]
    else:
        run_opts["inc_model"] = "lognormal"

    if "incubation_type" in run_setup["modelopts"]:
        run_opts["inc_type"] = run_setup["modelopts"]["incubation_type"]
    else:
        run_opts["inc_type"] = "deterministic"
    
    #------------------------------------------------------------------
    #-mcmc model parameters
    run_opts["mcmc_log"] = run_setup["mcmcopts"]["logfile"]
    run_opts["mcmc_nsteps"] = run_setup["mcmcopts"]["nsteps"]
    run_opts["mcmc_nfinal"] = run_setup["mcmcopts"]["nfinal"]
    run_opts["mcmc_gamma"] = run_setup["mcmcopts"]["gamma"]

    run_opts["inicov"] = np.array(run_setup["mcmcopts"]["cvini"])
    run_opts["inistate"] = run_setup["mcmcopts"]["cini"]
    if len(run_opts["inicov"].shape) == 1:
        run_opts['inicov'] = np.diag(run_opts["inicov"])

    run_opts["spllo"] = np.array(run_setup["mcmcopts"]["spllo"])
    run_opts["splhi"] = np.array(run_setup["mcmcopts"]["splhi"])

    #------------------------------------------------------------------
    #-bayes framework
    run_opts["lpf_type"] = run_setup["bayesmod"]["lpf_type"]
    run_opts["error_model_type"] = run_setup["bayesmod"]["error_model_type"]
    run_opts["prior_types"] = run_setup["bayesmod"]["prior_types"]
    run_opts["prior_info"] = run_setup["bayesmod"]["prior_info"]
          
    #------------------------------------------------------------------
    run_opts["days_extra"] = run_setup["ppopts"]["days_extra"]

    if return_run_setup:
        return run_opts, run_setup
    else:
        return run_opts

def get_phi_hist(phi_hat_vec_hist):
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
    return phi_hat_hist, phi_hist, theta_mu_hist
    
def get_counts(run_opts,return_raw_data=False):
    """
    Get counts from raw files
    """
    # extract data from raw data
    days_since_day0 = []
    daily_counts = []
    rawdata_all = []
    for ireg, region in enumerate(run_opts["count_data"]):
        rawdata = np.loadtxt(region,delimiter=",",dtype=str)
        rawdata_all.append(rawdata)
        ndays = rawdata.shape[0]
        days_since_day0.append(np.array([(parser.parse(rawdata[i,0])-parser.parse(run_opts["day0"])).days \
                                    for i in range(ndays)]))

        daily_counts.append(np.array([float(rawdata[i,1]) for i in range(rawdata.shape[0])]))
        # scale daily counts
        daily_counts[-1] = daily_counts[-1]/(run_opts["population_data"][ireg] * 1.e6)
        # run averages
        if "running_avg_obs" in run_opts:
            daily_counts[-1] = runningAvg(daily_counts[-1], run_opts["running_avg_obs"])
            print("Taking {}-day running average of observations for {}".format(run_opts["running_avg_obs"],run_opts["region_tag"][ireg]))

    if return_raw_data:
        return days_since_day0, daily_counts, rawdata_all
    else:
        return days_since_day0, daily_counts
    
def csv_to_count_dict(csv_files=['../data/table_nm_diff.csv','../data/county_stats.csv'],oname='../data/count_dict.pkl'):
    with open(csv_files[0], newline='\n') as csvfile:
        count_csv = csv.reader(csvfile, delimiter=',')
        count_data = []
        for row in count_csv:
            count_data.append(row)
    n_counties = len(count_data[0])-1
    row_start = 2
    n_days = len(count_data)-row_start
    count_dict = {}
    for c in range(0,n_counties+1):
        name = count_data[0][c]
        if name.lower() == 'date':
            count_dict[name] = []
            for row in range(row_start,len(count_data)):
                count_dict[name].append(count_data[row][c])
            count_dict[name] = np.array(count_dict[name])
        else:
            count_dict[name] = {'daily_counts':[]}
            for row in range(row_start,len(count_data)):
                count_dict[name]['daily_counts'].append(count_data[row][c])
            count_dict[name]['daily_counts'] = list(map(int,count_dict[name]['daily_counts']))
            count_dict[name]['daily_counts'] = np.array(count_dict[name]['daily_counts'])
    with open(csv_files[1], newline='\n') as csvfile:
        count_csv = csv.reader(csvfile, delimiter=',')
        for row in count_csv:
            name = row[0]
            for key in count_dict.keys():
                if key.lower().replace(' ','') == name.lower().replace(' ',''):
                    count_dict[key]['population'] = int(row[1])  
    save_obj(count_dict,oname)

def f_period(s):
    l = s.split('.')
    if len(l) > 1:
        return l[0] + ' ' + l[1]
    else:
        return s
    
def get_county_adj(adj_csv='adjacency.csv',data_dir='../data/',counties=['BERNALILLO','VALENCIA']):
    with open(data_dir+adj_csv, newline='\n') as csvfile:
        all_county_adj = np.array(list(csv.reader(csvfile, delimiter=',')))
    all_counties = all_county_adj[0,1:]
    n_counties = len(all_counties)
    adj_mat = np.array(all_county_adj[1:,1:],dtype='int')
    all_counties = list(map(f_period,all_counties))
    idxs = []
    for county in counties:
        for i in range(n_counties):
            if county == all_counties[i]:
                idxs.append(i)
    return adj_mat[np.ix_(idxs,idxs)]

def date_to_idx(date,dates):
    for i in range(len(dates)):
        if date == dates[i]:
            date_idx = i
            break
    return date_idx

def get_county_data(day0='2020-06-10', dayf='2020-06-10', day0_pred='2020-06-10', dayf_pred='2020-06-10', running_avg_obs=7, counties=['BERNALILLO','VALENCIA'],poplulation_normalized=True, data_dir='../data/'):
    count_dict = load_obj(data_dir+'count_dict.pkl')
    for i in range(len(count_dict['Date'])):
        if day0 == count_dict['Date'][i]:
            day0_idx = i
            break
    day0_idx = date_to_idx(day0,count_dict['Date'])
    dayf_idx = date_to_idx(dayf,count_dict['Date'])
    n_days = dayf_idx - day0_idx + 1
    
    day0_pred_idx = date_to_idx(day0_pred,count_dict['Date'])
    dayf_pred_idx = date_to_idx(dayf_pred,count_dict['Date'])
    days_extra = dayf_pred_idx - day0_pred_idx + 1
    data = {}
    if counties == 'all':
        with open(data_dir+'adjacency.csv', newline='\n') as csvfile:
            all_county_adj = np.array(list(csv.reader(csvfile, delimiter=',')))
            all_counties = all_county_adj[0,1:]
            n_counties = len(all_counties)
            counties = list(map(f_period,all_counties))
    for county in counties:
        data[county] = {}
        data[county]['daily_counts'] = count_dict[county]['daily_counts'][day0_idx:dayf_pred_idx+1]
        data[county]['population'] = count_dict[county]['population']
        if poplulation_normalized:
            data[county]['daily_counts'] = (1/count_dict[county]['population'])*data[county]['daily_counts'] 
        data[county]['daily_counts'] = runningAvg(data[county]['daily_counts'],running_avg_obs)
        data[county]['daily_counts_forecast'] = data[county]['daily_counts'][n_days:]
        data[county]['daily_counts'] = data[county]['daily_counts'][:n_days]
        
    adj = get_county_adj(counties=counties,data_dir=data_dir)
    dates = count_dict['Date'][day0_idx:day0_idx+n_days]
    dates_forecast = count_dict['Date'][day0_pred_idx:day0_pred_idx+days_extra]
    return {'county_data':data, 'counties':counties, 'county_adjacency':adj, 'dates':dates, 'dates_forecast':dates_forecast,'days_extra':days_extra}

def load_params(params,theta_0_fname=None,theta_0_idx=-1,data_dir='/home/whbridg/repo/outbreakdetector/src/forWyatt/vi/data/'):
    d = get_county_data(day0=params['day0'],
                        dayf=params['dayf'],
                        day0_pred=params['forecast_day0'],
                        dayf_pred=params['forecast_dayf'],
                        counties=params['counties'],
                        data_dir=data_dir,
                        running_avg_obs=params['running_avg_obs'])
    county_data = d['county_data']
    nd = len(county_data[list(county_data.keys())[0]]['daily_counts'])
    for key in county_data.keys():  
        plt.plot(county_data[key]['daily_counts'])
    plt.legend(list(county_data.keys()))
    plt.title(r'Day$_0$: {0}, Day$_f$: {1},  n. days: {2}'.format(params['day0'],params['dayf'],nd))
    plt.xlabel(r'Days since Day$_0$')
    plt.ylabel('Normalized counts')
    plt.savefig('county_data.pdf',bbox_inches='tight')
    
    params["counties"] = d['counties']
    params["Nreg"] = len(county_data.keys())
    params["t"] = np.arange(0,nd)
    params["t_forecast"] = np.arange(params['t'][-1]+1,params['t'][-1]+d['days_extra']+1)
    params["daily_counts"] = np.array([county_data[county]['daily_counts'] for county in county_data.keys()])
    params["daily_counts_forecast"] = np.array([county_data[county]['daily_counts_forecast'] for county in county_data.keys()])
    params["populations"] = np.array([county_data[county]['population'] for county in county_data.keys()])
    params["F_LN_lims"] = get_F_LN_lims(params["incubation_median"],params["incubation_sigma"],params["F_LN_eps"])
    params["W"] = d['county_adjacency']
    params["D"] = np.identity(params["Nreg"])
    if params["scale_cov"]:
        for i in range(params["Nreg"]):
            params["D"][i,i] = np.sum(params["W"][i,:])
    params["dates"] = d['dates']
    params["dates_forecast"] = d['dates_forecast']
    params["days_extra"] = d['days_extra']

class TimerError(Exception):

    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):

        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:

            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:

            raise TimerError(f"Timer is not running. Use .start() to start it")

        self.elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        return self.elapsed_time