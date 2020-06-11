import scipy.ndimage as ndimage
import xarray as xr
import numpy.matlib as npml
import numpy as np
from .geos import earth_dist_2d
from .geos import area
from datetime import date


def ts_areamean(data,lat,lat_unit='degree'):
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

def ts_volumemean(p,lev,lat,lon):

    sz = np.shape(p)
    mask = np.ones_like(p); mask[np.where(np.isnan(p))] = np.nan

    dA = np.repeat(np.repeat(np.reshape(area(lat,lon),(1,1,sz[2],sz[3])),sz[0],axis=0),sz[1],axis=1)*mask
    dz = np.repeat(np.repeat(np.repeat(np.reshape(np.gradient(lev),(1,len(lev),1,1)),sz[0],axis=0),sz[2],axis=2),sz[3],axis=3)
    dV = dA*dz; V = np.nansum(np.nansum(np.nansum(dV,axis=3),axis=2),axis=1)
    dp = p*dV
    ts_p = np.nansum(np.nansum(np.nansum(dp,axis=3),axis=2),axis=1)/V
    return ts_p

def ts_anomaly(p,lat,type='monthly'):

    anom, _ = anomaly(p,type=type)
    ts_anom = ts_areamean(p,lat)

def anomaly(p, time =None, type='monthly', climatologyPeriod=[None, None], pctile=90, windowHalfWidth=1,\
                              smoothPercentile=False,\
                              smoothPercentileWidth=1, maxPadLength=False, coldSpells=False):

    if type == 'annual' or type =='yearly' or type =='year' or type=='y':
        clim = np.nanmean(p,axis=0)
        anom = p - clim
    elif type == 'constant' or type == 'Constant' or type == 'c':
        clim = np.nanmean(p,axis=0)
        anom = p - clim
    elif type == 'monthly' or type == 'm' or type == 'month':
        if p.ndim == 1:
            sp = np.shape(p)
        else:
            sp = np.squeeze(np.shape(p))
        n_month = sp[0]
        if (n_month % 12) != 0:
            raise TypeError('Length of time should be multiplies of 12')
        n_year = int(n_month / 12)
        sp_new = np.array([n_year,12])
        if len(sp) > 1:
            sp_new = list(np.concatenate([[n_year,12],list(sp[1:])]))
        p_new = np.reshape(p, sp_new)  # reshape p into n_year
        clim = np.nanmean(p_new, axis=0)  # calulate climatology
        anom = p_new - clim
        anom = np.reshape(anom,sp)
    elif type == 'daily' or type =='day' or type == 'd':
        if len(np.shape(p)) > 1:
            print('only 1d daily data accepted.')
            raise
        else:
            clim_dict = climatology_daily(t, temp, climatologyPeriod=climatologyPeriod, pctile=pctile, windowHalfWidth=windowHalfWidth,
                              smoothPercentile=smoothPercentile,
                              smoothPercentileWidth=smoothPercentileWidth, maxPadLength=maxPadLength, coldSpells=coldSpells)
            clim = clim_dict['seas']
            anom = p - clim
    return anom, clim

def monthly_to_annual(p,time=None):
    '''
    Calc monthly mean p into annual mean
    :param p:
    :param time:
    :return:
    '''
    from .tools import toordinal
    sp = np.shape(p)
    n_month = sp[0]
    if np.remainder(n_month,12) == 0:
        n_year = int(n_month/12)
        p = np.reshape(p, np.concatenate(([n_year, 12], list(sp[1:]))))  # reshape p into n_year
        p_new = np.nanmean(p,axis=1).squeeze()
        if (time == None):
            return p_new

        else:
            time_new = np.reshape(p, [n_year,12])
            return p_new, time_new

    else:
        if time == None:
            raise TypeError('input property is not the multiplies of 12, time cannot be None.')
        time = toordinal(time)

        years = np.unique([date.fromordinal(t).year for t in time])
        time_new = [date(year,1,1) for year in years]
        T = len(years)
        sp = np.shape(p)
        sp_new = np.concatenate(([T],list(sp[1:])))
        p_new = np.full(sp_new,np.nan)

        for in_t in range(T):
            year = years[in_t]
            d_start = date(year,1,1).toordinal()
            d_end = date(year,12,31).toordinal()
            in_d = np.squeeze(np.where((time>=d_start)*(time<=d_end)))
            p[in_t,:] = np.nanmean(p[in_d,:])
        return p_new,time_new


def daily_to_monthly(data_day,time):
    from datetime import datetime
    import numpy as np

    years = np.unique(np.array([datetime.fromordinal(t).year for t in time]))
    # months= np.array([datetime.fromordinal(t).month for t in time])
    time_frags = np.append(np.array([datetime(year, month, 1).toordinal() for year in years for month in range(1, 13)]),\
                          datetime(years[-1] + 1, 1, 1).toordinal())
    sp = np.array(np.shape(data_day))
    sp[0] = 0
    data_month = np.full(sp,np.nan)
    time_month = np.full(0,np.nan)
    for in_frag in range(len(time_frags)-1):
        in_time = np.where((time>=time_frags[in_frag])*(time<time_frags[in_frag+1]))[0]
        if np.array(in_time).size > 0.0:
            data_month = np.concatenate((data_month,np.array([np.nanmean(data_day[in_time],axis=0)])),axis=0)
            year = datetime.fromordinal(time_frags[in_frag]).year
            month = datetime.fromordinal(time_frags[in_frag]).month
            time_month = np.concatenate((time_month,[datetime(year,month,15).toordinal()]),axis=0)
    return data_month, time_month


def climatology_daily(t, temp, climatologyPeriod=[None, None], pctile=90, windowHalfWidth=5, smoothPercentile=True,
           smoothPercentileWidth=31,  maxPadLength=False, coldSpells=False):
    '''

    Applies the Hobday et al. (2016) marine heat wave definition to an input time
    series of temp ('temp') along with a time vector ('t'). Outputs properties of
    all detected marine heat waves.

    Inputs:

      t       Time vector, in datetime format (e.g., date(1982,1,1).toordinal())
              [1D numpy array of length T]
      temp    Temperature vector [1D numpy array of length T]

    Outputs:
      clim    Climatology of SST. Each key (following list) is a seasonally-varying
              time series [1D numpy array of length T] of a particular measure:

        'thresh'               Seasonally varying threshold (e.g., 90th percentile)
        'seas'                 Climatological seasonal cycle
        'missing'              A vector of TRUE/FALSE indicating which elements in
                               temp were missing values for the MHWs detection

    Options:

      climatologyPeriod      Period over which climatology is calculated, specified
                             as list of start and end years. Default is to calculate
                             over the full range of years in the supplied time series.
                             Alternate periods suppled as a list e.g. [1983,2012].
      pctile                 Threshold percentile (%) for detection of extreme values
                             (DEFAULT = 90)
      windowHalfWidth        Width of window (one sided) about day-of-year used for
                             the pooling of values and calculation of threshold percentile
                             (DEFAULT = 5 [days])
      smoothPercentile       Boolean switch indicating whether to smooth the threshold
                             percentile timeseries with a moving average (DEFAULT = True)
      smoothPercentileWidth  Width of moving average window for smoothing threshold
                             (DEFAULT = 31 [days])
      minDuration            Minimum duration for acceptance detected MHWs
                             (DEFAULT = 5 [days])
      joinAcrossGaps         Boolean switch indicating whether to join MHWs
                             which occur before/after a short gap (DEFAULT = True)
      maxGap                 Maximum length of gap allowed for the joining of MHWs
                             (DEFAULT = 2 [days])
      maxPadLength           Specifies the maximum length [days] over which to interpolate
                             (pad) missing data (specified as nans) in input temp time series.
                             i.e., any consecutive blocks of NaNs with length greater
                             than maxPadLength will be left as NaN. Set as an integer.
                             (DEFAULT = False, interpolates over all missing values).
      coldSpells             Specifies if the code should detect cold events instead of
                             heat events. (DEFAULT = False)


    Notes:

      1. This function assumes that the input time series consist of continuous daily values
         with few missing values. Time ranges which start and end part-way through the calendar
         year are supported.

      2. This function supports leap years. This is done by ignoring Feb 29s for the initial
         calculation of the climatology and threshold. The value of these for Feb 29 is then
         linearly interpolated from the values for Feb 28 and Mar 1.

      3. The calculation of onset and decline rates assumes that the heat wave started a half-day
         before the start day and ended a half-day after the end-day. (This is consistent with the
         duration definition as implemented, which assumes duration = end day - start day + 1.)

      4. For the purposes of MHW detection, any missing temp values not interpolated over (through
         optional maxPadLLength) will be set equal to the seasonal climatology. This means they will
         trigger the end/start of any adjacent temp values which satisfy the MHW criteria.

      5. If the code is used to detect cold events (coldSpells = True), then it works just as for heat
         waves except that events are detected as deviations below the (100 - pctile)th percentile
         (e.g., the 10th instead of 90th) for at least 5 days. Intensities are reported as negative
         values and represent the temperature anomaly below climatology.

    Written by Eric Oliver, Institue for Marine and Antarctic Studies, University of Tasmania, Feb 2015

    '''


    #
    # Time and dates vectors
    #

    # Generate vectors for year, month, day-of-month, and day-of-year
    T = len(t)
    year = np.zeros((T))
    month = np.zeros((T))
    day = np.zeros((T))
    doy = np.zeros((T))
    for i in range(T):
        year[i] = date.fromordinal(t[i]).year
        month[i] = date.fromordinal(t[i]).month
        day[i] = date.fromordinal(t[i]).day
    # Leap-year baseline for defining day-of-year values
    year_leapYear = 2012  # This year was a leap-year and therefore doy in range of 1 to 366
    t_leapYear = np.arange(date(year_leapYear, 1, 1).toordinal(), date(year_leapYear, 12, 31).toordinal() + 1)
    dates_leapYear = [date.fromordinal(tt.astype(int)) for tt in t_leapYear]
    month_leapYear = np.zeros((len(t_leapYear)))
    day_leapYear = np.zeros((len(t_leapYear)))
    doy_leapYear = np.zeros((len(t_leapYear)))
    for tt in range(len(t_leapYear)):
        month_leapYear[tt] = date.fromordinal(t_leapYear[tt]).month
        day_leapYear[tt] = date.fromordinal(t_leapYear[tt]).day
        doy_leapYear[tt] = t_leapYear[tt] - date(date.fromordinal(t_leapYear[tt]).year, 1, 1).toordinal() + 1
    # Calculate day-of-year values
    for tt in range(T):
        doy[tt] = doy_leapYear[(month_leapYear == month[tt]) * (day_leapYear == day[tt])]

    # Constants (doy values for Feb-28 and Feb-29) for handling leap-years
    feb28 = 59
    feb29 = 60

    # Set climatology period, if unset use full range of available data
    if (climatologyPeriod[0] is None) or (climatologyPeriod[1] is None):
        climatologyPeriod[0] = year[0]
        climatologyPeriod[1] = year[-1]

    #
    # Calculate threshold and seasonal climatology (varying with day-of-year)
    #

    # if alternate temperature time series is supplied for the calculation of the climatology
    # if alternateClimatology:
    #     tClim = alternateClimatology[0]
    #     tempClim = alternateClimatology[1]
    #     TClim = len(tClim)
    #     yearClim = np.zeros((TClim))
    #     monthClim = np.zeros((TClim))
    #     dayClim = np.zeros((TClim))
    #     doyClim = np.zeros((TClim))
    #     for i in range(TClim):
    #         yearClim[i] = date.fromordinal(tClim[i]).year
    #         monthClim[i] = date.fromordinal(tClim[i]).month
    #         dayClim[i] = date.fromordinal(tClim[i]).day
    #         doyClim[i] = doy_leapYear[(month_leapYear == monthClim[i]) * (day_leapYear == dayClim[i])]
    # else:
    tempClim = temp.copy()
    TClim = np.array([T]).copy()[0]
    yearClim = year.copy()
    monthClim = month.copy()
    dayClim = day.copy()
    doyClim = doy.copy()

    # Flip temp time series if detecting cold spells
    if coldSpells:
        temp = -1. * temp
        tempClim = -1. * tempClim

    # Pad missing values for all consecutive missing blocks of length <= maxPadLength
    if maxPadLength:
        temp = pad(temp, maxPadLength=maxPadLength)
        tempClim = pad(tempClim, maxPadLength=maxPadLength)

    # Length of climatological year
    lenClimYear = 366
    # Start and end indices
    clim_start = np.where(yearClim == climatologyPeriod[0])[0][0]
    clim_end = np.where(yearClim == climatologyPeriod[1])[0][-1]
    # Inialize arrays
    thresh_climYear = np.NaN * np.zeros(lenClimYear)
    seas_climYear = np.NaN * np.zeros(lenClimYear)
    clim = {}
    clim['thresh'] = np.NaN * np.zeros(TClim)
    clim['seas'] = np.NaN * np.zeros(TClim)
    # Loop over all day-of-year values, and calculate threshold and seasonal climatology across years
    for d in range(1, lenClimYear + 1):
        # Special case for Feb 29
        if d == feb29:
            continue
        # find all indices for each day of the year +/- windowHalfWidth and from them calculate the threshold
        tt0 = np.where(doyClim[clim_start:clim_end + 1] == d)[0]
        # If this doy value does not exist (i.e. in 360-day calendars) then skip it
        if len(tt0) == 0:
            continue
        tt = np.array([])
        for w in range(-windowHalfWidth, windowHalfWidth + 1):
            tt = np.append(tt, clim_start + tt0 + w)
        tt = tt[tt >= 0]  # Reject indices "before" the first element
        tt = tt[tt < TClim]  # Reject indices "after" the last element
        thresh_climYear[d - 1] = np.percentile(nonans(tempClim[tt.astype(int)]), pctile)
        seas_climYear[d - 1] = np.mean(nonans(tempClim[tt.astype(int)]))
    # Special case for Feb 29
    thresh_climYear[feb29 - 1] = 0.5 * thresh_climYear[feb29 - 2] + 0.5 * thresh_climYear[feb29]
    seas_climYear[feb29 - 1] = 0.5 * seas_climYear[feb29 - 2] + 0.5 * seas_climYear[feb29]

    # Smooth if desired
    if smoothPercentile:
        # If the climatology contains NaNs, then assume it is a <365-day year and deal accordingly
        if np.sum(np.isnan(seas_climYear)) + np.sum(np.isnan(thresh_climYear)):
            valid = ~np.isnan(thresh_climYear)
            thresh_climYear[valid] = runavg(thresh_climYear[valid], smoothPercentileWidth)
            valid = ~np.isnan(seas_climYear)
            seas_climYear[valid] = runavg(seas_climYear[valid], smoothPercentileWidth)
        # >= 365-day year
        else:
            thresh_climYear = runavg(thresh_climYear, smoothPercentileWidth)
            seas_climYear = runavg(seas_climYear, smoothPercentileWidth)

    # Generate threshold for full time series
    clim['thresh'] = thresh_climYear[doy.astype(int) - 1]
    clim['seas'] = seas_climYear[doy.astype(int) - 1]

    # Save vector indicating which points in temp are missing values
    clim['missing'] = np.isnan(temp)
    clim['anom'] = temp - clim['seas']
    # Set all remaining missing temp values equal to the climatology
    temp[np.isnan(temp)] = clim['seas'][np.isnan(temp)]

    #
    # Find MHWs as exceedances above the threshold
    #

    # Time series of "True" when threshold is exceeded, "False" otherwise
    exceed_bool = temp - clim['thresh']
    exceed_bool[exceed_bool <= 0] = False
    exceed_bool[exceed_bool > 0] = True
    # Find contiguous regions of exceed_bool = True
    events, n_events = ndimage.label(exceed_bool)


    # Flip climatology and intensties in case of cold spell detection
    if coldSpells:
        clim['seas'] = -1. * clim['seas']
        clim['thresh'] = -1. * clim['thresh']

    return clim


def runavg(ts, w):
    '''

    Performs a running average of an input time series using uniform window
    of width w. This function assumes that the input time series is periodic.

    Inputs:

      ts            Time series [1D numpy array]
      w             Integer length (must be odd) of running average window

    Outputs:

      ts_smooth     Smoothed time series

    Written by Eric Oliver, Institue for Marine and Antarctic Studies, University of Tasmania, Feb-Mar 2015

    '''
    # Original length of ts
    N = len(ts)
    # make ts three-fold periodic
    ts = np.append(ts, np.append(ts, ts))
    # smooth by convolution with a window of equal weights
    ts_smooth = np.convolve(ts, np.ones(w)/w, mode='same')
    # Only output central section, of length equal to the original length of ts
    ts = ts_smooth[N:2*N]

    return ts


def pad(data, maxPadLength=False):
    '''

    Linearly interpolate over missing data (NaNs) in a time series.

    Inputs:

      data	     Time series [1D numpy array]
      maxPadLength   Specifies the maximum length over which to interpolate,
                     i.e., any consecutive blocks of NaNs with length greater
                     than maxPadLength will be left as NaN. Set as an integer.
                     maxPadLength=False (default) interpolates over all NaNs.

    Written by Eric Oliver, Institue for Marine and Antarctic Studies, University of Tasmania, Jun 2015

    '''
    data_padded = data.copy()
    bad_indexes = np.isnan(data)
    good_indexes = np.logical_not(bad_indexes)
    good_data = data[good_indexes]
    interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
    data_padded[bad_indexes] = interpolated
    if maxPadLength:
        blocks, n_blocks = ndimage.label(np.isnan(data))
        for bl in range(1, n_blocks+1):
            if (blocks==bl).sum() > maxPadLength:
                data_padded[blocks==bl] = np.nan

    return data_padded


def nonans(array):
    '''
    Return input array [1D numpy array] with
    all nan values removed
    '''
    return array[~np.isnan(array)]
