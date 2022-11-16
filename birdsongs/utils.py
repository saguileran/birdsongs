import peakutils, time, warnings #emcee,
import numpy as np
import pandas as pd
import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from numpy.polynomial import Polynomial
from numpy.linalg import norm as Norm

from matplotlib import cm
from matplotlib.gridspec import GridSpec

from scipy import signal
from scipy.interpolate import interp1d
from scipy.optimize import root
from scipy.signal import argrelextrema, butter, savgol_filter, find_peaks #hilbert

from sklearn.linear_model import LinearRegression
from random import uniform
from multiprocessing import Pool
from IPython.display import display as Display

import lmfit

from librosa import yin, pyin, feature, display, onset, times_like, stft, fft_frequencies
import librosa 
from maad import *

from IPython.display import Audio # reproduce audio 

Pool() # crea pool to parallel programming for optimization
warnings.filterwarnings(action='once') # omite warnings spam

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

def WriteAudio(name, fs, s):
    sound.write(name, fs, np.asarray(s,  dtype=np.float32))
    
def Enve(out, fs, Nt):
    time = np.linspace(0, len(out)/fs, len(out))
    out_env = sound.envelope(out, Nt=Nt) 
    t_env = np.arange(0,len(out_env),1)*len(out)/fs/len(out_env)
    t_env[-1] = time[-1] 
    fun_s = interp1d(t_env, out_env)
    return fun_s(time)

def AudioPlay(obj):
    return Audio(data=obj.s, rate=obj.fs)