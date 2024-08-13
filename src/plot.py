import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_pp(pred_in,region_i,params,ax,ylims=None):
    opts = params['ppopts']
    # plt_regions = opts['plt_regions']
    datesPred = np.concatenate([params['dates'],params['dates_forecast']])
    # pred = np.copy(Y_preds_pp)
    pred = np.copy(pred_in)
    pred *= params['populations'][region_i]
    
    qntList = opts['quantiles_filled']
    iendData = params['dates'].shape[0]
    midpoints = [0.5*(qntList[i]+qntList[i+1]) for i in range(len(qntList)-1)]
    min_val = min(midpoints)
    max_val = 0.5
    normalize = lambda c : 1.0/(max_val-min_val)*(c - min_val)
    
    # colormap settings
    import matplotlib as mpl
    cmap1 = mpl.cm.PuBu
    cmap2 = mpl.cm.PuRd
    
    maxVal = -1.e100
    # Plot filled quantile regions
    for i in range(len(qntList)-1):
        qnt0 = np.quantile(pred,qntList[i],axis=0)
        qnt1 = np.quantile(pred,qntList[i+1],axis=0)
        midPt = 0.5*(qntList[i]+qntList[i+1])
        alph = 0.8
        if qntList[i] >= 0.5:
            pl2 = ax.fill_between(datesPred[:iendData],qnt0[:iendData],qnt1[:iendData],color=cmap1(normalize(1-midPt)),alpha=alph,zorder=1)
            pl2 = ax.fill_between(datesPred[iendData-1:],qnt0[iendData-1:],qnt1[iendData-1:],color=cmap2(normalize(1-midPt)),alpha=alph,zorder=1)
        else:
            pl2 = ax.fill_between(datesPred[:iendData],qnt0[:iendData],qnt1[:iendData],color=cmap1(normalize(midPt)),alpha=alph,zorder=1)
            pl2 = ax.fill_between(datesPred[iendData-1:],qnt0[iendData-1:],qnt1[iendData-1:],color=cmap2(normalize(midPt)),alpha=alph,zorder=1)
            
    # Plot specific quantiles as lines
    for i in range(len(opts["quantiles_plot"])):
        qnt = opts["quantiles_plot"][i]
        ltp = opts["quantiles_linetype"][i]
        lwd = opts["quantiles_linewidth"][i]
        color = opts["quantiles_color"][i]
        qntPred=np.quantile(pred,qnt,axis=0)
        maxVal = max(maxVal,qntPred.max())
        pl1=ax.plot(datesPred,np.quantile(pred,qnt,axis=0),ltp,lw=lwd,color=color)

    if opts['show_truth']:
        offset = -3
        ax.plot(params['dates'],params['populations'][region_i]*params['daily_counts'][region_i],'ko',mfc='black',mec='k',mew=1.5)
        ax.plot(params['dates_forecast'][:offset],params['populations'][region_i]*params['daily_counts_forecast'][region_i][:offset],'ko',mfc='w',mec='k',mew=1.5)
        maxVal = max(maxVal,(params['populations'][region_i]*params['daily_counts'][region_i]).max())

    if ylims is None:
        y0 = 0.0
        y1 = 1.1*maxVal
        ax.set_ylim([y0,y1])
    else:
        ax.set_ylim([ylims[0],ylims[1]])
    ax.set_xlim([datesPred[0],datesPred[-1]])

    if 'xlabel' in opts.keys():
        ax.set_xlabel(opts['xlabel'],fontsize=opts['xlabel_size'])
    if 'ylabel' in opts.keys():
        ax.set_ylabel(opts['ylabel'],fontsize=opts['ylabel_size'])

    ax.set_xticks(opts['xticks'])
    ax.tick_params(axis='x', labelrotation=45)
    ax.tick_params(axis='x', labelsize=opts['xtick_size'])
    ax.tick_params(axis='y', labelsize=opts['ytick_size'])

    return ax