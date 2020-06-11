#!/usr/bin/env python

import numpy as np

def earth_dist_2p(lat, lon, unit='m', r=6373.0):
    '''
    Calculate distance between two points.

    dist = earth_dist2p(lon, lat, unit='km', r = 6373.0)
    :param lon: array, longitude, shoule be [lon1,lon2]
    :param lat: array, latitude, shoule be [lat1,lat2]
    :param unit: str, units, default is 'km', can be 'km', 'm', or 'mile'
    :param r: earth radius, defalt is 6373km.
    :return: dist, distance, default unit is 'km'
    '''

    if len(lon) != len(lat):
        raise TypeError("Length of input variable 'lon' and 'lat' should be the same.")

    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)

    lon1 = lon[0]
    lon2 = lon[1]
    lat1 = lat[0]
    lat2 = lat[1]
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    # new rad before two points
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = r * np.abs(c)
    # Units
    if unit == 'm' or unit == 'meter':
        dist = dist * 1.0e3
    elif unit == 'mi' or unit == 'miles' or unit == 'mile':
        dist = dist / 0.621371

    return dist


def earth_dist_2d(lat=np.arange(-89.5, 90), lon=np.arange(0.5, 360), unit='m', r=6373.0, lon_0=None, lat_0=None):
    '''
    Calculate 2 dimensional distance x and y of longitude lon and latitude lat
    :param lon: array, default is np.arange(0.5,360). Can be meshgrid array.
    :param lat: array, default is np.arange(-89.5,90). Can be meshgrid array.
    :param unit: str, units, default is 'km', can be 'km', 'm', or 'mile'
    :param r: float, earth radius in kilometers.
    :param lon_0: float, oringinal longitude
    :param lat_0: float, oringinal latitude
    :return: x, y
    '''
    if np.ndim(lon) != np.ndim(lat):
        raise TypeError("Dimensions of input variable 'lon' and 'lat' should be the same.")

    if lat_0 == None:
        lat_0 = lat[0]
    if lon_0 == None:
        lon_0 = lon[0]

    x = np.reshape(np.array([(lon_p-lon_0)/360.0*2*np.pi*r for lon_p in lon]),(1,len(lon)))
    weight = np.abs(np.reshape(np.cos(np.deg2rad(lat)),(len(lat),1)))
    x = np.dot(weight,x)
    if unit == 'm':
        x = x*10.0e3
    y = np.repeat(np.reshape(np.array([earth_dist_2p([lat_0,lat_p],[0,0],unit=unit,r=r) for lat_p in lat]),(len(lat),1)),len(lon),axis=1)
    return y, x



def area(lat,lon,unit='m'):

    y,x = earth_dist_2d(lat,lon,unit=unit)
    _, dx = np.gradient(x)
    dy,_  = np.gradient(y)
    dA = dx*dy
    return dA



