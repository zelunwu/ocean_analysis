

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
