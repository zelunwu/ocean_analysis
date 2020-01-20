
from mpl_toolkits.basemap import Basemap
import numpy as np
import cmocean
import matplotlib.pyplot as plt

def contourf(*args,projection='cyl',cmap=cmocean.cm.balance):
    '''
    contourf(lon,lat,data,projection,lonrange,latrange,vmin,vmax,cmap)

    :param lon: longitude
    :param lat: latitude
    :param data: Any
    :param lonrange: longitude range, defalt is [min(lon),max(lon]
    :param latrange: latitude range, defalt is [min(lat),max(lat)]
    :param clim: cmap limit, [vmin.vmax]
    :param projection: basemap projection, str = 'cyl'
    :param cmap: cmap, default is cmocea.cm.balance

    Return:
    map: of basemap

    Author:
    Zelun Wu
    zelunwu@stu.xmu.edu.cn, zelunwu@udel.edu
    Xiamen University, University of Delaware
    '''
    if len(args) <3:
        raise TypeError("Not enough input variables")
    else:
        lon = args[0]
        lat = args[1]
        data = args[2]
        lonrange = [np.nanmin(lon), np.nanmax(lon)]
        latrange = [np.nanmin(lat), np.nanmax(lat)]
        vmin = -np.nanmax(np.abs(data))
        vmax = -vmin
        if len(args) >= 4:
            lonrange = args[3]
            if len(args) >=5:
                latrange = args[4]
                if len(args) >=6:
                    vmin = args[5][0]
                    vmax = args[5][1]
    print('lonrange = ',lonrange,', latrange=',latrange)
    print('clim= ',[vmin,vmax])

    llcrnrlon = np.min(lonrange)
    urcrnrlon = np.max(lonrange)

    llcrnrlat = np.min(latrange)
    urcrnrlat = np.max(latrange)
    lon_0 = np.mean(lonrange)
    lat_0 = np.mean(latrange)

    map = Basemap(projection=projection,
                  lat_0=lat_0, lon_0=lon_0,
                  llcrnrlon=llcrnrlon,
                  llcrnrlat=llcrnrlat,
                  urcrnrlon=urcrnrlon,
                  urcrnrlat=urcrnrlat)

    lon_mesh = lon
    lat_mesh = lat
    if np.ndim(lon) == 1:
        lon_mesh, lat_mesh = np.meshgrid(lon,lat)
    h = map.contourf(lon_mesh, lat_mesh, data, cmap=cmap, vmin=vmin,vmax =vmax)
    c = map.contour(lon_mesh, lat_mesh, data, colors='0.6', vmin=vmin,vmax =vmax)
    plt.clabel(c, inline=True, fontsize=13, colors='k')
    map.drawmapboundary()
    # draw parallels and meridians.
    # label parallels on right and top
    # meridians on bottom and left
    parallels = np.arange(0., 91, 10.)
    # labels = [left,right,top,bottom]
    map.drawparallels(parallels, labels=[True, False, False, True])
    meridians = np.arange(10., 361., 10.)
    map.drawmeridians(meridians, labels=[True, False, False, True])

    map.drawlsmask(land_color='0.8')
    map.drawcoastlines(color='0.6')
    cb = map.colorbar(h)
    # print(vmin,vmax)
    return map, h

def pcolor(*args,projection='cyl',cmap=cmocean.cm.balance):
    '''
    contourf(lon,lat,data,projection,lonrange,latrange,vmin,vmax,cmap)

    :param lon: longitude
    :param lat: latitude
    :param data: Any
    :param lonrange: longitude range, defalt is [min(lon),max(lon]
    :param latrange: latitude range, defalt is [min(lat),max(lat)]
    :param clim: cmap limit, [vmin.vmax]
    :param projection: basemap projection, str = 'cyl'
    :param cmap: cmap, default is cmocea.cm.balance

    Return:
    map of basemap

    Author:
    Zelun Wu
    zelunwu@stu.xmu.edu.cn, zelunwu@udel.edu
    Xiamen University, University of Delaware
    '''
    if len(args) < 3:
        raise TypeError("Not enough input variables")
    else:
        lon = args[0]
        lat = args[1]
        data = args[2]
        lonrange = [np.nanmin(lon), np.nanmax(lon)]
        latrange = [np.nanmin(lat), np.nanmax(lat)]
        vmin = -np.nanmax(np.abs(data))
        vmax = -vmin
        if len(args) >= 4:
            lonrange = args[3]
            if len(args) >= 5:
                latrange = args[4]
                if len(args) >= 6:
                    vmin = args[5][0]
                    vmax = args[5][1]
    print('lonrange = ', lonrange, ', latrange=', latrange)
    print('clim= ', [vmin, vmax])

    llcrnrlon = np.min(lonrange)
    urcrnrlon = np.max(lonrange)

    llcrnrlat = np.min(latrange)
    urcrnrlat = np.max(latrange)
    lon_0 = np.mean(lonrange)
    lat_0 = np.mean(latrange)

    map = Basemap(projection=projection,
                  lat_0=lat_0, lon_0=lon_0,
                  llcrnrlon=llcrnrlon,
                  llcrnrlat=llcrnrlat,
                  urcrnrlon=urcrnrlon,
                  urcrnrlat=urcrnrlat)

    lon_mesh = lon
    lat_mesh = lat
    if np.ndim(lon) == 1:
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    h = map.pcolormesh(lon_mesh, lat_mesh, data, cmap=cmap, vmin=vmin, vmax=vmax)
    c = map.contour(lon_mesh, lat_mesh, data, colors='0.6', vmin=vmin,vmax =vmax)
    plt.clabel(c, inline=True, fontsize=13, colors='k')
    map.drawmapboundary()
    # draw parallels and meridians.
    # label parallels on right and top
    # meridians on bottom and left
    parallels = np.arange(0., 91, 10.)
    # labels = [left,right,top,bottom]
    map.drawparallels(parallels, labels=[True, False, False, True])
    meridians = np.arange(10., 361., 10.)
    map.drawmeridians(meridians, labels=[True, False, False, True])

    map.drawlsmask(land_color='0.9')
    map.drawcoastlines(color='0.6')
    # map.drawmeridians()
    cb = map.colorbar(h)
    # print(vmin,vmax)
    return map


