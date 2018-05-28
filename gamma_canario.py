#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 12:05:18 2018

Objetivo: Crear cantos sintéticos de canarios

@author: juan
"""

# %%
from random import uniform
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


def dxdt_synth(v, dv):
    x, y, i1, i2, i3 = v[0], v[1], v[2], v[3], v[4]
    dv[0] = y
    dv[1] = alfa1*gm**2 + beta1*gm**2*x - gm**2*x**3 - gm*x**2*y \
        + gm**2*x**2 - gm*x*y
    dv[2] = i2
    dv[3] = -s1overLG*s1overCH*i1 - rdis*(s1overLB+s1overLG)*i2 \
        + i3*(s1overLG*s1overCH-rdis*RB*s1overLG*s1overLB) \
        + s1overLG*forcing2 + rdis*s1overLG*s1overLB*forcing1
    dv[4] = -(1/s1overLG)*s1overLB*i2 - RB*s1overLB*i3 + s1overLB*forcing1
    return dv


def rk4(dv, v, n, t, dt):
    v1 = []
    k1 = []
    k2 = []
    k3 = []
    k4 = []
    for x in range(0, n):
        v1.append(x)
        k1.append(x)
        k2.append(x)
        k3.append(x)
        k4.append(x)
    dt2 = dt/2.0
    dt6 = dt/6.0
    for x in range(0, n):
        v1[x] = v[x]
    dv(v1, k1)
    for x in range(0, n):
        v1[x] = v[x] + dt2*k1[x]
    dv(v1, k2)
    for x in range(0, n):
        v1[x] = v[x] + dt2*k2[x]
    dv(v1, k3)
    for x in range(0, n):
        v1[x] = v[x] + dt*k3[x]
    dv(v1, k4)
    for x in range(0, n):
        v1[x] = v[x] + dt*k4[x]
    for x in range(0, n):
        v[x] = v[x] + dt6*(2.0*(k2[x]+k3[x])+k1[x]+k4[x])
    return v


def consecutive(data, stepsize=1):
    """
    Parte una tira de datos en bloques de datos consecutivos.
    Ej:
        [1,2,3,4,6,7,9,10,11] -> [[1,2,3,4],[6,7],[9,10,11]]
    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


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
plt.plot(normalizar(song))
plt.plot(envelope)
plt.axhline(y=umbral)
supra = np.where(envelope > umbral)[0]
silabas = consecutive(supra)
NN = 1024
overlap = 1/1.1
sigma = NN/10
fu, tu, Sxx = get_spectrogram(song, fs, window=NN, overlap=overlap,
                              sigma=sigma)
fig, ax = plt.subplots(2, sharex=True, figsize=(12, 6))
ax[0].pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                 rasterized=True)
ax[0].set_ylim(0, 8000)
ax[0].set_xlim(min(time), max(time))
for ss in silabas:
    ax[0].plot([time[ss[0]], time[ss[-1]]], [0, 0], 'k', lw=5)
print('Datos ok')
# ff/SCI en intervalos de longitud predeterminada y moviendo ventanita
dt_seg = 0.01
dn_seg = int(dt_seg*fs)
dn_step = dn_seg//3
t_center = dt_seg
n_center = int(t_center*fs)
SCI = []
ff = []
t_sci = []
while n_center < len(song)-dn_seg:
    segment = song[n_center-dn_seg:n_center+dn_seg]
    msf, aff = SpectralContent(segment, fs)
    SCI.append(msf/aff)
    ff.append(aff)
    t_sci.append(n_center/fs)
    n_center += dn_step
ff = np.asarray(ff)
t_sci = np.asarray(t_sci)
SCI = np.asarray(SCI)
color = [str(item/(max(t_sci))) for item in t_sci[ff > 500]]
ax[0].scatter(t_sci[ff > 500], ff[ff > 500], c=color)
ax[1].scatter(t_sci[ff > 500], SCI[ff > 500], c=color)
ax[1].set_ylabel('SCI')
plt.figure()
plt.scatter(ff[ff > 500], SCI[ff > 500], c=color)
plt.xlabel('fundamental')
plt.ylabel('SCI')
# %%ff/SCI por silaba (~promedio)
fig, ax = plt.subplots(2, sharex=True, figsize=(12, 6))
ax[0].pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                 rasterized=True)
ax[0].set_ylim(0, 8000)
ax[0].set_xlim(min(time), max(time))
dn_seg = np.asarray([len(x) for x in silabas])
dt_seg = dn_seg/fs
t_center = np.asarray([time[silabas[n][0]]+dt_seg[n]/2
                       for n in range(len(silabas))])
SCI_av = np.zeros_like(t_center)
ff_av = np.zeros_like(t_center)
t_sci_av = t_center
for n_s in range(len(silabas)):
    if len(silabas[n_s]) > 100:
        segment = song[silabas[n_s]]
        msf, aff = SpectralContent(segment, fs, method='syllable')
        SCI_av[n_s] = msf/aff
        ff_av[n_s] = aff
color = [str(item/(max(t_sci_av))*255) for item in t_sci_av]
ax[0].scatter(t_sci_av, ff_av, c=color)
ax[1].scatter(t_sci_av, SCI_av, c=color)
ax[1].set_ylabel('SCI_av')
plt.figure()
plt.scatter(ff_av, SCI_av, c=color)
plt.axvline(x=0)
plt.axhline(y=1.)
plt.xlim(0, 5000)
plt.ylim(0.5, 2.)

# %% ff, SCI, Amplitud
df = pd.DataFrame(0, index=np.arange(len(alphas)*len(betas)*len(gammas)),
                  columns=['alpha', 'beta', 'gamma', 'fundamental', 'msf',
                  'SCI', 'amplitud', 'time'])

for n_param in range(len(df)):
    n_start = int(n_param*n_per_param)
    n_center = n_start+int(n_per_param/2)
    segment = out[n_start:n_start+n_per_param]
    # revisar metodo
    msf, ff, amp = SpectralContent(segment, sampling, method='synth')
    SCI = msf/ff
    df.iloc[n_param] = [alpha_out[n_center], beta_out[n_center],
                        gamma_out[n_center], ff, msf, SCI, amp, time[n_center]]
fig, ax = plt.subplots(5, figsize=(12, 18), sharex=True)
ax[0].plot(time, out)
ax[0].set_xlim(min(time), max(time))

ax[1].pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                 rasterized=True)
ax[1].set_xlim(min(time), max(time))
ax[1].set_ylim(0, 8000)
ax[1].plot(df['time'], df['fundamental'], 'o')

ax[2].plot(df['time'], df['fundamental'], '.')
ax[2].set_ylabel('ff')
ax[2].set_ylim(0, 8000)

ax[3].plot(df['time'], df['SCI'])
ax[3].set_ylabel('SCI')

ax[4].plot(df['time'], df['amplitud'])
ax[4].set_ylabel('Amplitud')

# %%
nn = 0
df_aux = df[df['gamma'] == gammas[nn]]
ff = df_aux['fundamental']
SCI = df_aux['SCI']
amp = df_aux['amplitud']
alfa = df_aux['alpha']
beta = df_aux['beta']

fig, ax = plt.subplots(3, figsize=(8, 15), sharex=True)
fig.suptitle('gamma = {:.0f}'.format(gammas[nn]))

cax = ax[0].scatter(alfa, beta, c=ff, s=100)
fig.colorbar(cax, ax=ax[0], fraction=0.046, pad=0.04)
ax[0].set_xlabel('alpha')
ax[0].set_ylabel('beta')
ax[0].set_title('fundamental')

cax = ax[1].scatter(alfa, beta, c=SCI, s=100)
fig.colorbar(cax, ax=ax[1], fraction=0.046, pad=0.04)
ax[1].set_xlabel('alpha')
ax[1].set_ylabel('beta')
ax[1].set_title('SCI')

cax = ax[2].scatter(alfa, beta, c=amp, s=100)
fig.colorbar(cax, ax=ax[2], fraction=0.046, pad=0.04)
ax[2].set_xlabel('alpha')
ax[2].set_ylabel('beta')
ax[2].set_title('amplitud')
