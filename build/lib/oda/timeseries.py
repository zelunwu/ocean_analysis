
import numpy.matlib as npml
import numpy as np

def areamean2ts(data,lat,lat_unit='degree'):
    '''
    areamean2ts(data,lat,lat_unit='degree'):
    Input:
        data: 3d numpy narray
        lat: latitude
        lat_unit: 'degree' or 'reg', default is degree
    Output:
        ts: areaweighted mean time series.

    Author:
    Zelun Wu
    zelunwu@stu.xmu.edu.cn, zelunwu@udel.edu
    Xiamen University, University of Delaware
    '''

    pi = npml.pi
    if lat_unit == 'degree':
        lat_reg = np.array(lat/180 * pi)
    weight_lat = np.reshape(npml.cos(lat_reg),[1,len(lat),1])
    shape_data = np.array(np.shape(data))
    weight = np.repeat(weight_lat,shape_data[2],axis=2)
    weight = np.repeat(weight, shape_data[0], axis=0)
    data_weighted = data * weight
    ts = np.squeeze(np.nanmean(np.nanmean(data_weighted,axis=2),axis=1))
    return ts

