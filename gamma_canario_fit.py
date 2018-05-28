#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 12:05:18 2018

Objetivo: Crear cantos sintéticos de canarios

@author: juan
"""

# %%
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.signal import argrelextrema
from scipy.signal import butter
import pandas as pd
import peakutils
from matplotlib import gridspec


def envelope_cabeza(signal, method='percentile', intervalLength=210, perc=90):
    """
    Calcula envolvente. En segmentos de intervalLength calcula el maximo
    (si no se especifica metodo) o el percentil especificado.
    """
    if method == 'percentile':
        pp = perc
    else:
        pp = 100
    absSignal = abs(signal)
    dt2 = int(intervalLength/2)
    outputSignal = np.zeros(len(absSignal))
    outputSignal[0] = absSignal[0]
    outputSignal[-1] = absSignal[-1]

    for baseIndex in range(1, len(absSignal)-1):
        if baseIndex < dt2:
            percentil = np.percentile(absSignal[:baseIndex], pp)
        elif baseIndex > len(absSignal) - dt2:
            percentil = np.percentile(absSignal[baseIndex:], pp)
        else:
            percentil = np.percentile(absSignal[baseIndex-dt2:baseIndex+dt2],
                                      pp)
        outputSignal[baseIndex] = percentil
    return outputSignal


def butter_lowpass(fs, lcutoff=3000.0, order=15):
    nyq = 0.5*fs
    normal_lcutoff = lcutoff/nyq
    bl, al = butter(order, normal_lcutoff, btype='low', analog=False)
    return bl, al


def butter_lowpass_filter(data, fs, lcutoff=3000.0, order=6):
    bl, al = butter_lowpass(fs, lcutoff, order=order)
    yl = signal.filtfilt(bl, al, data)
    return yl


def butter_highpass(fs, hcutoff=100.0, order=6):
    nyq = 0.5*fs
    normal_hcutoff = hcutoff/nyq
    bh, ah = butter(order, normal_hcutoff, btype='high', analog=False)
    return bh, ah


def butter_highpass_filter(data, fs, hcutoff=100.0, order=5):
    bh, ah = butter_highpass(fs, hcutoff, order=order)
    yh = signal.filtfilt(bh, ah, data)
    return yh


def consecutive(data, stepsize=1, min_length=1):
    """
    Parte una tira de datos en bloques de datos consecutivos.
    Ej:
        [1,2,3,4,6,7,9,10,11] -> [[1,2,3,4],[6,7],[9,10,11]]
    """
    candidates = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    return [x for x in candidates if len(x) > min_length]


def sigmoid(x, dt=1, b=0):
    a = 5/(dt*44100)
    return 1/(1+np.exp(-(a*x+b)))


def normalizar(arr, minout=-1, maxout=1):
    """
    Normaliza un array en el intervalo minout-maxout
    """
    norm_array = np.copy(np.asarray(arr, dtype=np.double))
    norm_array -= min(norm_array)
    norm_array = norm_array/max(norm_array)
    norm_array *= maxout-minout
    norm_array += minout
    return norm_array


def get_spectrogram(data, sampling_rate, window=1024, overlap=1/1.1,
                    sigma=102.4):
    """
    Computa el espectrograma de la señal usando ventana gaussiana.

    sampling_rate = sampleo de la señal
    Window = numero de puntos en la ventana
    overlap = porcentaje de overlap entre ventanas
    sigma = dispersion de la ventana

    Devuelve:

    tu = tiempos espectro
    fu = frecuencias
    Sxx = espectrograma

    Ejemplo de uso:
    tt, ff, SS = get_spectrogram(song, 44100)
    plt.pcolormesh(tt, ff, np.log(SS), cmap=plt.get_cmap('Greys'),
                   rasterized=True)
    """
    fu, tu, Sxx = signal.spectrogram(data, sampling_rate, nperseg=window,
                                     noverlap=window*overlap,
                                     window=signal.get_window
                                     (('gaussian', sigma), window),
                                     scaling='spectrum')
    Sxx = np.clip(Sxx, a_min=np.amax(Sxx)*0.000001, a_max=np.amax(Sxx))
    return fu, tu, Sxx


def SpectralContent(data, fs, method='song', fmin=300, fmax=10000,
                    dt_transit=0.002):
    segment = data[int(dt_transit*fs):]
    fourier = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(len(segment), d=1/fs)
    min_bin = np.argmin(np.abs(freqs-fmin))
    max_bin = np.argmin(np.abs(freqs-fmax))
    fourier = np.abs(np.fft.rfft(segment))[min_bin:max_bin]
    freqs = np.fft.rfftfreq(len(segment), d=1/fs)[min_bin:max_bin]
    f_msf = np.sum(freqs*fourier)/np.sum(fourier)
    f_aff = 0
    amp = max(segment)-min(segment)
    if method == 'song':
        f_aff = freqs[np.argmax(fourier*(freqs/(freqs+500)**2))]
    elif method == 'syllable':
        orden = 10
        mm = argrelextrema(segment, np.greater, order=orden)[0]
        difs = np.diff(mm)
        while np.std(difs)/np.mean(difs) > 1/3 and orden > 1:
            orden -= 1
            mm = argrelextrema(segment, np.greater, order=orden)[0]
            difs = np.diff(mm)
        f_aff = fs/np.mean(np.diff(mm))
    elif method == 'synth':
        maximos = peakutils.indexes(fourier, thres=0.05, min_dist=5)
        if amp < 500:
            f_aff = 0
        elif len(maximos) > 0:
            f_aff = freqs[maximos[0]]
    return f_msf, f_aff, amp


# %%
# Nombre del ave
birdname = 'AmaVio'

# Carpeta donde estan guardados los wavs
files_path = '/home/juan/Documentos/Musculo/Codigo canarios/Files'

# Carpeta donde se van a guardar los resultados
analisis_path = '{}analysis/'.format(files_path)
if not os.path.exists(analisis_path):
    os.makedirs(analisis_path)

# %% Cargo datos de sonido y vs
print('Cargando datos...')
# Busca todos los archivos del pajaro.
sound_files = glob.glob(os.path.join(files_path,
                                     '*'+birdname+'*_s*wav'))
# Elijo el primero de estos archivos
num_file = 0
fs, song = wavfile.read(sound_files[num_file])

f_name_long = sound_files[num_file]
f_name = f_name_long.rsplit('/', 1)[1].split('_s_', 1)[0]

# Recorto el file
t_i = 38.9
t_f = 45.
song = song[int(t_i*fs):int(t_f*fs)]
time = np.linspace(0, len(song)/fs, len(song))
envelope = normalizar(envelope_cabeza(song, intervalLength=0.01*fs), minout=0)
umbral = 0.05
supra = np.where(envelope > umbral)[0]
silabas = consecutive(supra, min_length=100)
# %%
NN = 1024
overlap = 1/1.1
sigma = NN/10
fu, tu, Sxx = get_spectrogram(song, fs, window=NN, overlap=overlap,
                              sigma=sigma)
fig, ax = plt.subplots(2, sharex=True, figsize=(12, 6))
ax[0].plot(time, normalizar(song))
ax[0].plot(time, envelope)
ax[0].axhline(y=umbral)
ax[1].pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                 rasterized=True)
ax[1].set_ylim(0, 8000)
ax[1].set_xlim(min(time), max(time))
for ss in silabas:
    ax[1].plot([time[ss[0]], time[ss[-1]]], [0, 0], 'k', lw=5)
print('Datos ok')

# %%ff/SCI por silaba (~promedio)
df_song = pd.DataFrame(index=range(len(silabas)), columns=['fundamental',
                       'msf', 'SCI', 'amplitud'])
fig = plt.figure(figsize=(20, 6))
gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[:, 1])
ax0.pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
               rasterized=True)
ax0.set_ylim(0, 8000)
ax0.set_xlim(min(time), max(time))
dn_seg = np.asarray([len(x) for x in silabas])
dt_seg = dn_seg/fs
t_center = np.asarray([time[silabas[n][0]]+dt_seg[n]/2
                       for n in range(len(silabas))])
SCI_av = np.empty_like(t_center)
ff_av = np.empty_like(t_center)
t_sci_av = t_center
for n_s in range(len(silabas)):
    segment = song[silabas[n_s]]
    msf, aff, amp = SpectralContent(segment, fs, dt_transit=0,
                                    method='syllable')
    df_song.iloc[n_s] = [aff, msf, msf/aff, amp]
    SCI_av[n_s] = msf/aff
    ff_av[n_s] = aff
color = [str(item/(max(t_sci_av))*255) for item in t_sci_av]
ax0.scatter(t_sci_av, ff_av, c=color)
ax1.scatter(t_sci_av, SCI_av, c=color)
ax1.set_ylabel('SCI_av')
ax2.scatter(ff_av, SCI_av, c=color)
ax2.axhline(y=1.)
ax2.set_xlim(0, 5000)
ax2.set_ylim(0.5, 2.)

# %% Cargo las grillas de valores
grid_file = 'ff_SCI-2018-05-24.13.53.26'
df_grid = pd.read_csv(grid_file)
df_fit = pd.DataFrame(columns=['fundamental', 'SCI', 'alfa', 'beta', 'gamma',
                               'ff song', 'SCI song'])
gammas = df_grid.groupby('gamma')['gamma'].unique()  # unique values of gamma
# Para cada gamma busco alfa y beta que ajustan ff y sci
DSCI = np.zeros(len(gammas))
n_fila = 0
for ngama in range(len(gammas)):
    gama = gammas.iloc[ngama][0]
    df_aux = df_grid[df_grid['gamma'] == gama]
    for sil in range(len(df_song)):
        ff_obj = df_song['fundamental'][sil]
        SCI_obj = df_song['SCI'][sil]
        paso = 1
        aux2 = df_aux.loc[np.abs(df_aux['fundamental']-ff_obj) < 100*paso]
        while len(aux2) < 5:
            paso += 1
            aux2 = df_aux.loc[np.abs(df_aux['fundamental']-ff_obj) < 100*paso]
        ff = aux2['fundamental'].astype(float)
        SCI = aux2['SCI'].astype(float)
        fit_ix = np.argmin(np.abs(SCI-SCI_obj))
        alfa = aux2['alpha'].astype(float)
        beta = aux2['beta'].astype(float)
        df_fit.loc[n_fila] = [ff.loc[fit_ix], SCI.loc[fit_ix],
                              alfa.loc[fit_ix], beta.loc[fit_ix], gama,
                              ff_obj, SCI_obj]
        DSCI[ngama] += (SCI.loc[fit_ix] - SCI_obj)**2
        n_fila += 1
# %%
colores = ['r', 'g', 'b', 'k', 'm']
fig, ax = plt.subplots(1)
for ngama in range(len(gammas)):
    gama = gammas.iloc[ngama][0]
    df_aux = df_fit[df_fit['gamma'] == gama]
    alfa = df_aux['alfa']
    beta = df_aux['beta']
    ax.plot(alfa, beta, 'o', c=colores[ngama],
            label='gamma = {:.0f}'.format(gama))
ax.set_xlabel('alfa')
ax.set_ylabel('beta')
ax.legend()

# %% Para cada gamma grafico ff vs sci (todo el espacio) y los puntos del canto
fig, ax = plt.subplots(1, len(gammas), figsize=(20, 4), sharex=True,
                       sharey=True)
colores = ['r', 'g', 'b', 'y', 'c']
for ngama in range(len(gammas)):
    gama = gammas.iloc[ngama][0]
    df_aux = df_grid[df_grid['gamma'] == gama]
    alfa = df_aux['alpha'].astype(float)
    ff = df_aux['fundamental'].astype(float)
    SCI = df_aux['SCI'].astype(float)
    ax[ngama].scatter(ff, SCI, s=10,
                      label='gamma = {:.0f}'.format(gama))
    ax[ngama].legend()
    ax[ngama].plot(df_fit[df_fit['gamma'] == gama]['fundamental'],
                   df_fit[df_fit['gamma'] == gama]['SCI'], 'mo',
                   fillstyle='none', label='fit')
    ax[ngama].plot(df_song['fundamental'], df_song['SCI'], 'kx', label='song')
fig.tight_layout()

# %% Me quedo con el "mejor" gamma y veo las curvas de ff-SCI para alpha
gama = gammas.iloc[3][0]
df_aux = df_grid[df_grid['gamma'] == gama]
df_fit_aux = df_fit[df_fit['gamma'] == gama]
ff = df_aux['fundamental'].astype(float)
SCI = df_aux['SCI'].astype(float)
alfa = df_aux['alpha'].astype(float)
plt.scatter(ff, SCI, s=10, c=np.asarray(np.abs(alfa)))
plt.xlabel('fundamental')
plt.ylabel('SCI')
cbar = plt.colorbar()
cbar.set_label('alfa')
plt.plot(df_song['fundamental'], df_song['SCI'], 'kx', label='song')
alfa_bins = np.linspace(min(alfa), max(alfa), 11, endpoint=True)
filas = 5
columnas = 2
fig, ax = plt.subplots(filas, columnas, figsize=(10, 10), sharex=True,
                       sharey=True)
for nstart in range(len(alfa_bins[:-1])):
    start = alfa_bins[nstart]
    end = alfa_bins[nstart+1]
    df_bin = df_aux[(df_aux['alpha'] > start) & (df_aux['alpha'] < end)]
    ax[nstart % filas][nstart//filas].plot(df_bin['fundamental'],
                                           df_bin['SCI'], '.',
                                           label='alfa: {:.2f} / {:.2f}'.format
                                           (start, end))
    ax[nstart % filas][nstart//filas].plot(df_song['fundamental'],
                                           df_song['SCI'], 'kx', label='song')
    ax[nstart % filas][nstart//filas].legend()
fig.tight_layout()
