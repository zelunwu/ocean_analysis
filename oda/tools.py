from pathlib import Path
import numpy as np
import datetime
import pandas as pd
import xarray as xr


def where(timeseris,in_range):
    index = np.where((timeseris>=in_range[0])*(timeseris<=in_range[1]))[0]
    return index

def getfiles(dir_data, characteristic_str='*nc'):
    '''
    files = getfiles(dir_data, characteristic_str='*nc'):
    Input
        ::dir_data: str, default is pwd()
        characteristic_str: str, default is '*nc'
    Output:
        files list.
    '''
    var_path = Path(dir_data)
    files = list(var_path.glob('**/'+characteristic_str))
    files.sort()
    return files

def recursive_read_files(files,chunks=1440):
    '''
    recursive_read_files(files,chunks=1440):
    :param files:
    :param chunks:
    :return: ds
    '''
    ds = xr.open_dataset(files[0],chunks=chunks)
    for file in files[1:]:
        ds_i = xr.open_dataset(file,chunks=chunks)
        ds = xr.concat([ds,ds_i],dim='time')
    return ds


def isleap(year):
    '''
    isleap(year)
    To determine whether one year is leap or not.
    :param year: int
    :return: Ture : leap, False: not leap.

    Author:
    Zelun Wu
    zelunwu@stu.xmu.edu.cn, zelunwu@udel.edu
    '''
    if (year % 4) == 0:
        if (year % 100) == 0:
            if (year % 400) == 0:
                return True  # 整百年能被400整除的是闰年
            else:
               return False
        else:
            return True # 非整百年能被4整除的为闰年
    else:
        return False

def datetype(time):
    '''
    Determine whether input date is a valid datetime format or ordinate formate
    :param date:
    :return:
    '''

    try:
        time_new = datetime.datetime.toordinal(time)
        if isinstance(date_num,int):
            type = 'datetime'
    except ordinal:
        time_new = datetime.datetime.fromordinal(time)
        type = 'int'
    except np_datetime64:
        time_new = pd.to_datetime(time)
        type = 'np_datetime_64'
    else:
        type = 'unknown'

    return type

def toordinal(time):
    '''
    Convert time into ordinal(datenum in matlab)
    :param time:
    :return: time_new

    Author:
    Zelun Wu
    zelunwu@stu.xmu.edu.cn, zelunwu@udel.edu
    '''

    if isinstance(time[0],int):
        time_new = time
    else:
        try:
            time_new = datetime.datetime.toordinal(time)
        except:
            time_new = [pd.to_datetime(t).toordinal() for t in time]
    return time_new