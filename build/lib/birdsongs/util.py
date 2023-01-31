import peakutils, time, warnings, lmfit, pickle, copyreg, sys, os #emcee,
import numpy as np
import pandas as pd
import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

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

#%%
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

#%%
def WriteAudio(name, fs, s):
    sound.write(name, fs, np.asarray(s,  dtype=np.float32))

#%% 
def Enve(out, fs, Nt):
    time = np.linspace(0, len(out)/fs, len(out))
    out_env = sound.envelope(out, Nt=Nt) 
    t_env = np.arange(0,len(out_env),1)*len(out)/fs/len(out_env)
    t_env[-1] = time[-1] 
    fun_s = interp1d(t_env, out_env)
    return fun_s(time)

#%%
def AudioPlay(obj):
    return Audio(data=obj.s, rate=obj.fs)

#%%
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

#%%    
def Positions(klicker):
    tinis = np.array([tini[0] for tini in klicker._positions[r"$t_{ini}$"]])
    tends = np.array([tend[0] for tend in klicker._positions[r"$t_{end}$"]])
    
    if tinis.size==tends.size:
        return np.array([[tinis[i], tends[i]] for i in range(len(tinis))])
    else:
        return np.array([tinis, tends])
 
#%%   
def Print(string): return display(Math(string))

#%%
def smoothstep(x, x_min=0, x_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    # [result += comb(N+n, n)*comb(2*N+1, N-n)*(-x)**n for n in range(0, N + 1)]
    for n in range(0, N + 1):
         result += comb(N+n, n)*comb(2*N+1, N-n)*(-x)**n
    result *= x**(N+1)

    return result

#%%
def grab_audio(path, audio_format='mp3'):
    filelist = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name[-3:].casefold() == audio_format and name[:2] != '._':
                filelist.append(os.path.join(root, name))
    return filelist

#%%
def DownloadXenoCanto(data, XC_ROOTDIR="./examples/", XC_DIR="Audios/", filters=['english name', 'scientific name'],
                        type=None, area=None, cnt=None, loc=None, nr=None, q='">C"', len=None, len_limits=['00:00', '01:00'],
                        max_nb_files=20, verbose=False, min_quality="B"):
    """
    data = [['Rufous-collared Sparrow', 'Zonotrichia capensis'],
            ['White-backed',        'Dendrocopos leucotos']]
    len = '"5-60"'
    len_limits = ['00:00', '01:00']
    XC_ROOTDIR = './files/'
    XC_DIR = 'zonotrichia_dataset' 
    """
    
    df_species = pd.DataFrame(data,columns=filters)
    sp, gen = [], []

    for name in df_species['scientific name']:
        gen.append(name.rpartition(' ')[0])
        sp.append(name.rpartition(' ')[2])

    df_query = pd.DataFrame()
    df_query['param1'] = gen
    df_query['param2'] = sp
    df_query['param3'] = 'q:'+q
    if type is not None: df_query['param4'] ='type:'+type
    if area is not None: df_query['param5'] ='area:'+area
    if cnt is not None:  df_query['param6'] ='cnt:'+cnt
    if loc is not None:  df_query['param7'] ='loc:'+loc
    if nr is not None:   df_query['param8'] ='nr:'+nr
    if len is not None:  df_query['param9'] ='len:'+len

    # Get recordings metadata corresponding to the query
    df_dataset= util.xc_multi_query(df_query, 
                                    format_time = False,
                                    format_date = False,
                                    verbose = verbose)
    if df_dataset.size!=0:
        df_dataset = util.xc_selection(df_dataset,
                                        max_nb_files=max_nb_files,
                                        min_length=len_limits[0],
                                        max_length=len_limits[1],
                                        min_quality=min_quality,
                                        verbose = verbose )
            
        # download audio files
        util.xc_download(df_dataset,
                        rootdir = XC_ROOTDIR,
                        dataset_name= XC_DIR,
                        overwrite=True,
                        save_csv= True,
                        verbose = verbose)

        filelist = grab_audio(XC_ROOTDIR+XC_DIR)
        df = pd.DataFrame()
        for file in filelist:
            df = df.append({'fullfilename': file,
                            'filename': Path(file).parts[-1][:-4],
                            'species': Path(file).parts[-2]},
                            ignore_index=True)

        for i in range(df_species.shape[0]):
            df = df_dataset["en"].str.contains(df_species["english name"][i], case=False)
            spec = df_dataset[df]["gen"][0] +" "+ df_dataset[df]["sp"][0] #df_species["scientific name"][i]
            scientific = df_dataset[df]["en"][0]#df_species["english name"][i]
            df_dataset[df].to_csv(XC_ROOTDIR+XC_DIR+spec+"_"+scientific+"/spreadsheet-XC.csv")

        return df_dataset#["en"][df]
    else:
        raise ValueError("No sounds were found with your specifications. Try again with other parameters.")

#%%
def BifurcationODE(f1, f2):
        beta_bif = np.linspace(-2.5, 1/3, 1000)  # mu2:beta,  mu1:alpha
        xs, ys, alpha, beta, gamma = sym.symbols('x y alpha beta gamma')
        # ---------------- Labia EDO's Bifurcation -----------------------
        f1 = eval(f1)#ys
        f2 = eval(f2)#(-alpha-beta*xs-xs**3+xs**2)*gamma**2 -(xs+1)*gamma*xs*ys
        x01 = sym.solveset(f1, ys)+sym.solveset(f1, xs)  # find root f1
        f2_x01 = f2.subs(ys,x01.args[0])                     # f2(root f1)
        f  = sym.solveset(f2_x01, alpha)                         # root f2 at root f1, alpha=f(x,beta)
        g  = alpha                                               # g(x) = alpha, above
        df = f.args[0].diff(xs)                                   # f'(x)
        dg = g.diff(xs)                                           # g'(x)
        roots_bif = sym.solveset(df-dg, xs)                       # bifurcation roots sets (xmin, xmas)
        mu1_curves = [] 
        for ff in roots_bif.args:                                       # roots as arguments (expr)
            x_root = np.array([float(ff.subs(beta, mu2)) for mu2 in beta_bif], dtype=float)    # root evaluatings beta
            mu1    = np.array([f.subs([(beta,beta_bif[i]),(xs,x_root[i])]).args[0] for i in range(len(beta_bif))], dtype=float)
            mu1_curves.append(mu1)
        f1 = sym.lambdify([xs, ys, alpha, beta, gamma], f1)
        f2 = sym.lambdify([xs, ys, alpha, beta, gamma], f2)

        return beta_bif, mu1_curves, f1, f2

# def Enve(self, out, fs, Nt):
#     time = np.linspace(0, len(out)/fs, len(out))
#     out_env = sound.envelope(out, Nt=Nt) 
#     t_env = np.arange(0,len(out_env),1)*len(out)/fs/len(out_env)
#     t_env[-1] = time[-1] 
#     fun_s = interp1d(t_env, out_env)
#     return fun_s(time)