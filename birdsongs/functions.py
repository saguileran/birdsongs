import peakutils, time #emcee,
import numpy as np
import pandas as pd
import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt

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

Pool() # crea pool to parallel programming for optimization

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