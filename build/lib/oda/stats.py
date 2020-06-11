

import numpy as np 


def pdf(ts,nbin):
    '''

    Calculate Probability Density Function of time series ts.
    Input: 
        ts: time series
        nbins: how many bins you want to calculate
    Output:
        pdf: probability density function
        x: [min_ts,max_ts]
    
    Author:
    Zelun Wu
    zelunwu@stu.xmu.edu.cn, zelunwu@udel.edu
    Xiamen University, University of Delaware
    '''
    ts_min = np.min(ts) 
    ts_max = np.max(ts) 
    ts_range = ts_max-ts_min
    bin = ts_range/((nbin-1)*1.0)
    x = np.arange(ts_min,ts_max+bin/2,bin)
    hist = np.zeros([len(x)])

    for in_x in range(0,nbin):
        hist[in_x] = len(np.intersect1d(np.where(ts>=x[in_x]-bin/2), np.where(ts<x[in_x]+bin/2)))
    pdf = hist/len(ts)
    return pdf,x


def lag_regress(x_lead,y_delay,max_lag):

    # if not (len(ts_lead) == len(ts_delay)):
    #     print('length of two time series should be the same.')
    lags = np.arange(max_lag+1)
    ks = bs = ps = rs = np.array([])
    len_ts = len(x_lead)
    for lag in range(0,max_lag+1):
        x_lead_i = np.array(x_lead[0:len_ts-lag])
        y_delay_i =  np.array(y_delay[lag:])
        k,b,r,p, st = stats.linregress(x_lead_i,y_delay_i)
        ks = np.append(ks,k)
        bs = np.append(bs,b)
        rs = np.append(rs,r)
        ps = np.append(ps,p)

    return lags, ks, bs, ps, rs