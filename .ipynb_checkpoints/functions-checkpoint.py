import os, glob, peakutils, lmfit, time #emcee, 
import numpy as np
import pandas as pd
from math import floor, ceil
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.io import wavfile
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema, butter, savgol_filter, find_peaks #hilbert
from sklearn.linear_model import LinearRegression
from random import uniform
from numpy.polynomial import Polynomial
from multiprocessing import Pool

#from pydub import AudioSegment
#import signal_envelope as se
#from scipy.interpolate import UnivariateSpline

def envelope_cabeza(signal, method='percentile', intervalLength=210, perc=90):
    """
    Compute the envelope of an audio by calculating the maximum or percentil on each segments 
    """
    if method == 'percentile':        pp = perc
    else:                             pp = 100
    
    absSignal        = abs(signal)
    dt2              = int(intervalLength/2)
    outputSignal     = np.zeros(len(absSignal))
    outputSignal[0]  = absSignal[0]
    outputSignal[-1] = absSignal[-1]

    for baseIndex in range(1, len(absSignal)-1):
        if baseIndex < dt2:                     percentil = np.percentile(absSignal[:baseIndex], pp)
        elif baseIndex > len(absSignal) - dt2:  percentil = np.percentile(absSignal[baseIndex:], pp)
        else:                                   percentil = np.percentile(absSignal[baseIndex-dt2:baseIndex+dt2], pp)
        
        outputSignal[baseIndex] = percentil
    #print(np.shape(outputSignal))
    
    #analytic_signal = se(signal)
    #amplitude_envelope = se(signal)#np.abs(analytic_signal)
    #print(np.shape(amplitude_envelope))
    #print(np.shape(outputSignal))
    return outputSignal
    #return amplitude_envelope

def butter_lowpass(fs, lcutoff=3000.0, order=15):
    nyq            = 0.5*fs
    normal_lcutoff = lcutoff/nyq
    return butter(order, normal_lcutoff, btype='low', analog=False) # =bl, al

def butter_lowpass_filter(data, fs, lcutoff=3000.0, order=6):
    bl, al = butter_lowpass(fs, lcutoff, order=order)
    return  signal.filtfilt(bl, al, data)             # yl =
    
def butter_highpass(fs, hcutoff=100.0, order=6):
    nyq            = 0.5*fs
    normal_hcutoff = hcutoff/nyq
    return butter(order, normal_hcutoff, btype='high', analog=False)  # bh, ah =

def butter_highpass_filter(data, fs, hcutoff=100.0, order=5):
    bh, ah = butter_highpass(fs, hcutoff, order=order)
    return signal.filtfilt(bh, ah, data) #yh = 


def consecutive(data, stepsize=1, min_length=1):
    """
    Split an array in chuncks with consecutive data
    Esample:
        [1,2,3,4,6,7,9,10,11] -> [[1,2,3,4],[6,7],[9,10,11]]
    """
    candidates = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    return [x for x in candidates if len(x) > min_length]


def normalizar(arr, minout=-1, maxout=1, pmax=100, pmin=5, method='extremos'): #extremos
    """
    Normalice an array in the interval minout-maxout
    """
    norm_array = np.copy(np.asarray(arr, dtype=np.double))
    if method == 'extremos':
        norm_array -= min(norm_array)
        norm_array = norm_array/max(norm_array)
        norm_array *= maxout-minout
        norm_array += minout
    elif method == 'percentil':
        norm_array -= np.percentile(norm_array, pmin)
        norm_array = norm_array/np.percentile(norm_array, pmax)
        norm_array *= maxout-minout
        norm_array += minout
    return norm_array


def get_spectrogram(data, sampling_rate, window=1024, overlap=1/1.1,
                    sigma=102.4, scale=0.000001):
    """
    Compute the spectrogram of a signal using gaussian window
    INPUT:
        sampling_rate = sampling rate of signal
        window        = window number of points 
        overlap       = percentage of overlpa between windows
        sigma         = window dispersion
    OUTPUT:
        tu  = spectrum times
        fu  = frequencies
        Sxx = spectrogram, 

    Example:
    tt, ff, SS = get_spectrogram(song, 44100)
    plt.pcolormesh(tt, ff, np.log(SS), cmap=plt.get_cmap('Greys'), rasterized=True)
    """
    fu, tu, Sxx = signal.spectrogram(data, sampling_rate, nperseg=window,
                                     noverlap=window*overlap,
                                     window=signal.get_window
                                     (('gaussian', sigma), window),
                                     scaling='spectrum')
    Sxx = np.clip(Sxx, a_min=np.amax(Sxx)*scale, a_max=np.amax(Sxx))
    return fu, tu, Sxx

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
    window_new    = floor(window_chunck + (fraction%1)*window_chunck/(fraction//1))  # se podria usar ceil

    s_windowed = np.lib.stride_tricks.sliding_window_view(s, window_new)[::floor(overlap*window_new)] # overlap every
    t_windowed = np.lib.stride_tricks.sliding_window_view(t, window_new)[::floor(overlap*window_new)]
    
    return s_windowed, t_windowed

def SpectralContent(segment, fs):
    """"
    Calculate the spectral content for a signal, a ratio between the fundamenta frequency (FF) and the mean switching frequency (msf) frequency
    INPUT:
        segment = signal to calculate its spectral content 
        fs      = smaple rate of audio
    OUTPUT:
        f_msf = mean switching frequency
        f_aff = fundamenta frequency
        max1  = amplitud fundamental frequency
    """
    fourier = np.abs(np.fft.rfft(segment))
    freqs   = np.fft.rfftfreq(len(segment), d=1/fs)
    maximos = peakutils.indexes(fourier, thres=0.2, min_dist=5)
    
    f_msf = np.sum(freqs*fourier)/np.sum(fourier)
    amp   = max(segment)-min(segment)
    max1  = np.max(fourier) #max amplitud fourier
    
    if len(maximos)>0 and max1>50:  f_aff = freqs[maximos[0]]
    else:                           f_aff = 0.1 #np.argmax(fourier)
    
    return f_msf, f_aff, max1#amp

def FFandSCI(s, time, fs, t0, window_time=0.01, overlap=1):
    """
    Compute the fundamental frequency (FF) and spectral content index (SCI) using chuncks
    INPUT:
        s    = signal 
        time = time vector of the signal
        t0   = initial time for the signal
        window_time =
    OUTPUT:
        SCI          = sound content index array
        time_ampl    = time array sampled by chuncks
        freq_amp     = frequency array sampled by chuncks
        Ampl_freq    = amplitud frequency array sampled by chuncks
        freq_amp_int = frequency array with same size as signal input
        tim_inter    = time array with same size as signal input
    """
    s, t = Windows(s, time, fs, window_time=0.005, overlap=0.5)
    
    SCI,      time_ampl = np.zeros(np.shape(s)[0]), np.zeros(np.shape(s)[0])
    freq_amp, Ampl_freq = np.zeros(np.shape(s)[0]), np.zeros(np.shape(s)[0])
    
    for i in range(s.shape[0]):
        f_msf, f_aff, amp = SpectralContent(s[i], fs) 
        
        SCI[i]       = f_msf/f_aff
        time_ampl[i] = t[i,0] #window_length*i # left point
        freq_amp[i]  = f_aff  #max1/window_time
        Ampl_freq[i] = amp    #np.amax(amplitud_freq)
    
    time_ampl += t0
    
    tim_inter       = np.linspace(time_ampl[0], time_ampl[-1], time.size)  # time interpolated
    #time_ampl1 = time_ampl.reshape((-1,1)) # functions to interpolate
    
    model = LinearRegression().fit(time_ampl.reshape((-1,1)), freq_amp)
    
    inte_freq_amp   = interp1d(time_ampl, freq_amp) # = model.coef_*tim_inter+model.intercept_
    inte_Amp_freq   = interp1d(time_ampl, Ampl_freq) 
    # interpolate and smooth
    freq_amp_int = savgol_filter(inte_freq_amp(tim_inter), window_length=13, polyorder=3)
    Ampl_freq    = savgol_filter(inte_Amp_freq(tim_inter), window_length=13, polyorder=3)
    #print(np.shape(tim_inter), np.shape(freq_amp_int), np.shape(Ampl_freq))
    
    return SCI, time_ampl, freq_amp, Ampl_freq, freq_amp_int, tim_inter