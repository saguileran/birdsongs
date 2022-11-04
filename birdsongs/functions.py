import peakutils, lmfit, time #emcee,
import numpy as np
import pandas as pd
from maad import *
from math import floor, ceil
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema, butter, savgol_filter, find_peaks #hilbert
from sklearn.linear_model import LinearRegression
from random import uniform
from numpy.polynomial import Polynomial
from multiprocessing import Pool
from IPython.display import display
from librosa import yin, pyin

#from scipy.interpolate import UnivariateSpline

def consecutive(data, stepsize=1, min_length=1):
    """
    Split an array in chuncks with consecutive data
    Esample:
        [1,2,3,4,6,7,9,10,11] -> [[1,2,3,4],[6,7],[9,10,11]]
    """
    candidates = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    return [x for x in candidates if len(x) > min_length]


def rk4(f, v, dt):
    """
    Implent of Runge-Kuta 4th order
    INPUT:
        f  = differential equations functions y'=f(y)
        v  = vector y of the differential equations [x,y,i1,i2,i3]
        dt = rk4 time step
    OUTPUT:
        rk4 approximation 
    """
    k1 = f(v)    
    k2 = f(v + dt/2.0*k1)
    k3 = f(v + dt/2.0*k2)
    k4 = f(v + dt*k3)
    return v + dt*(2.0*(k2+k3)+k1+k4)/6.0
    
def Windows(s, t, fs, window_time=0.05, overlap=1):
    """
    Split tow signals, s and t, in chunks of at least lenght window_time
    INPUT:
        s  = signal amplitud 
        t  = time asocitaed to signal
        fs = sampling rate
        window_time = window time length by chunck
        overlap     = percentage of overlpa between chuncks
    OUTPUT:
        s_windowed = signal chuncked vector
        t_windowed = time chuncked vector
    """
    window_chunck = floor(window_time*fs) 
    fraction      = np.size(s)/window_chunck
    window_new    = floor(window_chunck + (fraction%1)*window_chunck/(fraction//1))  # or ceil

    s_windowed = np.lib.stride_tricks.sliding_window_view(s, window_new)[::floor(overlap*window_new)] # overlap every
    t_windowed = np.lib.stride_tricks.sliding_window_view(t, window_new)[::floor(overlap*window_new)]
    return s_windowed, t_windowed

def SpectralContent(segment, fs,  window_time=0.001):
    """"
    Calculate the spectral content for a signal, a ratio between the fundamenta frequency (FF) and the mean switching frequency (msf) frequency
    INPUT:
        segment = signal to calculate its spectral content 
        fs      = smaple rate of audio
    OUTPUT:
        f_msf = mean switching frequency
        f_aff = fundamenta frequency
        max1  = amplitud fundamental frequency = amp
    """
    fourier = np.abs(np.fft.rfft(segment))
    freqs   = np.fft.rfftfreq(len(segment), d=1/fs)
    maximos = peakutils.indexes(fourier, thres=0.2, min_dist=20)
    
    f_msf = np.sum(freqs*fourier)/np.sum(fourier)
    amp   = max(segment)-min(segment)
    max1  = np.max(fourier) #max amplitud fourier
    
    if len(maximos)>0 and max1>0.5:  f_aff = freqs[maximos[0]]
    else:                           f_aff = -1e6 # to penalize the bad outputs #np.argmax(fourier)
    
    return f_msf, f_aff, max1, len(maximos)#amp