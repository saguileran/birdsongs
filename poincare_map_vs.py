#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:35:07 2018

Sirve un mapa de Poincare de vs o de su envolvente para detectar intervalos
de actividad u otra cosa?

x(n+1) vs x(n)

@author: juan
"""

from random import uniform
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import pandas as pd
import sys
from scipy.signal import butter
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['xtick.direction'] = 'none'
plt.rcParams['ytick.direction'] = 'out'


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


# %%
birdname = 'ZF-MCV'
source_path = '/home/juan/Documentos/Musculo/Sintesis sueños/'

# Carpeta donde estan guardados los wavs
files_path = '{}files-sint-crit/all paper/'.format(source_path)

# %% Cargo datos de sonido y vs
os.chdir(files_path)
sound_files = glob.glob(os.path.join(os.getcwd(), '*' + birdname + '*_s*wav'))
vs_files = glob.glob(os.path.join(os.getcwd(),
                                  '*'+birdname+'*_vs*'+'denoised*'))
num_file = 0
f_name_long = vs_files[num_file]
f_name = f_name_long.rsplit('/', 1)[1].split('_vs_', 1)[0]
fs, vs_raw = wavfile.read(vs_files[num_file])

# Recorto el file
t_i = 1.
t_f = 3.
vs_raw = vs_raw[int(t_i*fs):int(t_f*fs)]
time = np.linspace(0, len(vs_raw)/fs, len(vs_raw))
f_name_dec = f_name.rsplit('_', 2)
sf_name = '{}.{}.{}'.format(f_name_dec[0], f_name_dec[1], f_name_dec[2])
num_file = np.where([sf_name in x for x in sound_files])[0]
if len(num_file) != 1:
    print('error numero de files')
    sys.exit(0)
fs, song = wavfile.read(sound_files[num_file[0]])
song = song[int(t_i*fs):int(t_f*fs)]
# %%
vs = butter_lowpass_filter(vs_raw, fs, order=5)
vs = butter_highpass_filter(vs, fs, order=6)
vs = butter_lowpass_filter(vs, fs, order=5)
vs -= np.mean(vs)
vs /= max(abs(vs))

# Calculo envolventes
percentil = 90
vs_envelope = envelope_cabeza(vs, method='percentile', intervalLength=110,
                              perc=percentil)
vs_envelope = normalizar(vs_envelope, minout=0, maxout=1)

vs_envelope_lenta = envelope_cabeza(vs, method='percentile',
                                    intervalLength=441, perc=percentil)
vs_envelope_lenta = normalizar(vs_envelope_lenta, minout=0, maxout=1)
# %%
fu, tu, Sxx = get_spectrogram(song, fs)
fig, ax = plt.subplots(3, figsize=(12, 6), sharex=True)
ax[0].pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                 rasterized=True, label='Song')
ax[0].set_ylim(0, 8000)
ax[1].plot(time, vs)
ax[2].plot(time, vs_envelope_lenta, 'r')

# %%
window = 0.05
n_window = int(window*fs)

dt = 1/3000     # max frec?
dn = int(dt*fs)
dn = 1

t_lag = 0.005
n_lag = int(t_lag*fs)
total_plots = int(len(vs)/n_window)
total_plots = 5

n_plot = 0
start = 0
end = n_window
fig, ax = plt.subplots(3, total_plots, figsize=(total_plots*10, total_plots*2))
while n_plot < total_plots and end < len(vs):
    ax[0][n_plot].plot(time[start:end][::dn],
                       vs[start:end][::dn])
    ax[0][n_plot].plot(time[start:end][::dn],
                       vs_envelope_lenta[start:end][::dn], 'r')
    ax[0][n_plot].set_ylim(-1, 1)
    ax[0][n_plot].set_xlim(time[start], time[end])
    ax[0][n_plot].axes.get_xaxis().set_visible(False)
    ax[0][n_plot].axes.get_yaxis().set_visible(False)

    ax[1][n_plot].plot(vs[start:end][::dn][:-n_lag],
                       vs[start:end][::dn][n_lag:], '.')
    ax[1][n_plot].set_xlim(-1, 1)
    ax[1][n_plot].set_ylim(-1, 1)
    ax[1][n_plot].axes.get_xaxis().set_visible(False)
    ax[1][n_plot].axes.get_yaxis().set_visible(False)

    ax[2][n_plot].plot(vs_envelope_lenta[start:end][::dn][:-n_lag],
                       vs_envelope_lenta[start:end][::dn][n_lag:], '.')
    ax[2][n_plot].set_xlim(0, 1)
    ax[2][n_plot].set_ylim(0, 1)
    ax[2][n_plot].axes.get_xaxis().set_visible(False)
    ax[2][n_plot].axes.get_yaxis().set_visible(False)
    n_plot += 1
    start += 1000
    end += 1000
fig.tight_layout()