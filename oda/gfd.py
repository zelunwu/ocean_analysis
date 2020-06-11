import numpy as np
from .timeseries import *
from .geos import *

def gradient(p, *varargs, unit='m'):
    '''
    gradient_2d(*varargs, p, unit of p). d\
    :param varargs: coordinates of p, can be N arrays, N depends on the ndims of p
    :param p: narray, can be 2d[lat,lon], 3d[time,lat,lon] or 3d[lev,lat,lon], or 4d[time,lev,lat,lon]

    return:
    ndim(p) = 2: dp_dy, dp_dx
    ndim(p) = 3: dp_dt, dp_dy, dp_dx, or dp_dz, dp_dy, dp_dx
    ndim(p) = 4: dp_dt, dp_dz, dp_dy, dp_dx

    Note: default unit of dx and dy are km

    Author:
    Zelun Wu,
    zelunwu@stu.xmu.edu.cn, zelunwu@udel.edu
    '''
    if len(varargs) != np.ndim(p):
        raise TypeError('Coordinates should be the same with property dimensions.')
    if np.ndim(p) == 2:
        lat = varargs[0]
        lon = varargs[1]

        dp_y, dp_x = np.gradient(p)

        y, x = earth_dist_2d(lat,lon,unit=unit)
        _, dx = np.gradient(x)
        dy, _ = np.gradient(y)

        dp_dx = dp_x/dx
        dp_dy = dp_y/dy
        return  dp_dy, dp_dx
    elif np.ndim(p) == 3:
        time = varargs[0]
        lat = varargs[1]
        lon = varargs[2]

        dp_t, dp_y, dp_x = np.gradient(p)

        y, x = earth_dist_2d(lat,lon,unit=unit)
        _, dx = np.gradient(x); dx = np.repeat(np.reshape(dx,[1,len(lat),len(lon)]),len(time),axis=0)
        dy, _ = np.gradient(y); dy = np.repeat(np.reshape(dy,[1,len(lat),len(lon)]),len(time),axis=0)
        dt = np.gradient(time); dt = np.repeat(np.repeat(np.reshape(time,[len(time),1,1]),len(lat),axis=1),len(lon),axis=2)
        dp_dt = dp_t/dt
        dp_dx = dp_x/dx
        dp_dy = dp_y/dy
        return  dp_dt,dp_dy,dp_dx
    elif np.ndim(p) == 4:
        time = varargs[0]
        lev = varargs[1]
        lat = varargs[2]
        lon = varargs[3]
        # print(time,lev,lat,lon)
        sz = np.shape(p)

        dp_t, dp_z, dp_y, dp_x = np.gradient(p)
        y, x = earth_dist_2d(lat, lon, unit=unit)
        _, dx = np.gradient(x);
        dx = np.repeat(np.repeat(np.reshape(dx, [1, 1, len(lat), len(lon)]), len(time), axis=0),len(lev),axis=1)
        dy, _ = np.gradient(y);
        dy = np.repeat(np.repeat(np.reshape(dy, [1, 1, len(lat), len(lon)]), len(time), axis=0),len(lev),axis=1)
        dz = np.reshape(np.gradient(lev),(1,len(lev),1,1))
        dt = np.reshape(np.gradient(time),(len(time),1,1,1))

        for in_dim in range(np.ndim(p)):
            if in_dim != 1:
                dz = np.repeat(dz, sz[in_dim], axis=in_dim)
            if in_dim != 0:
                dt = np.repeat(dt,sz[in_dim],axis=in_dim)

        dp_dt = dp_t/dt
        dp_dz = dp_z/dz
        dp_dy = dp_y/dy
        dp_dx = dp_x/dx
        return  dp_dt, dp_dz, dp_dy, dp_dx

# def budget(p,time,flux_surf, lev,lat,lon,u,v,depth='all'):
#     '''
#
#     :param p:
#     :param time:
#     :param flux_surf:
#     :param lev:
#     :param lat:
#     :param lon:
#     :param u:
#     :param v:
#     :param depth:
#     :return:
#
#     Author:
#     Zelun Wu,
#     zelunwu@stu.xmu.edu.cn, zelunwu@udel.edu
#
#     '''
#
#     lev = np.abs(lev)
#     if depth='all':
#         in_h = range(len(lev))
#     else:
#         in_h = np.squeeze(np.where(lev<=depth))
#
#     p = p[:,in_h,:,:]; sz = np.shape(p)
#     u = u[:,in_h,:,:]; v = v[:,in_h,:,:]; lev = lev[in_h]
#
#     dp_dt, dp_dz, dp_dy, dp_dx = gradient(p,time,lev,lat,lon,unit='m')
#     adv_x = -u*dp_dx
#     adv_y = -v*dp_dy
#     flux = flux_surf/lev[-1]
#     res = dp_dt - (adv_x+adv_y) - flux
#
#     return dp_dt, adv_x, adv_y, flux, res
