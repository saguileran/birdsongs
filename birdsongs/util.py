import peakutils, time, warnings, lmfit, pickle, copyreg #emcee,
import numpy as np
import pandas as pd
import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from playsound import playsound

from mpl_point_clicker import clicker
from mpl_pan_zoom import zoom_factory, PanManager, MouseButton

from numpy.polynomial import Polynomial
from numpy.linalg import norm as Norm

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.gridspec import GridSpec

from scipy import signal
from scipy.interpolate import interp1d
from scipy.optimize import root
from scipy.signal import argrelextrema, butter, savgol_filter, find_peaks #hilbert
from scipy.special import comb

from sklearn.linear_model import LinearRegression
from random import uniform
from multiprocessing import Pool
from IPython.display import display as Display
from IPython.display import display, Math


from librosa import yin, pyin, feature, display, onset, times_like, stft, fft_frequencies
import librosa 
from maad import *

from IPython.display import Audio # reproduce audio 

Pool() # crea pool to parallel programming for optimization
#warnings.filterwarnings(action='once') # omite warnings spam

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

def Klicker(fig, ax):
    zoom_factory(ax)
    pm = PanManager(fig, button=MouseButton.MIDDLE)
    klicker = clicker(ax, [r"$t_{ini}$",r"$t_{end}$"], markers=["o","x"], colors=["blue","green"],
                      legend_bbox=(1.01, 0.98))# #legend_loc='upper right',

    # hacky trick to keep the panmanager alive as long as the clicker is around
    # without having to return another object
    klicker._pm = pm

    #ax.legend(title="Interval Points", bbox_to_anchor=(1.1, 1.05))
    return klicker
    
def Positions(klicker):
    tinis = np.array([tini[0] for tini in klicker._positions[r"$t_{ini}$"]])
    tends = np.array([tend[0] for tend in klicker._positions[r"$t_{end}$"]])
    
    if tinis.size==tends.size:
        return np.array([[tinis[i], tends[i]] for i in range(len(tinis))])
    else:
        return np.array([tinis, tends])
    
def Print(string): return display(Math(string))


def smoothstep(x, x_min=0, x_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    # [result += comb(N+n, n)*comb(2*N+1, N-n)*(-x)**n for n in range(0, N + 1)]
    for n in range(0, N + 1):
         result += comb(N+n, n)*comb(2*N+1, N-n)*(-x)**n
    result *= x**(N+1)

    return result