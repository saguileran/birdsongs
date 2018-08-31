#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 18:03:50 2018

Prueba de metodos para calcular envolvente de vS de forma robusta (y rapida
idealmente)

@author: juan
"""
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.signal import butter
from scipy.signal import argrelextrema


def envelope_cabeza(signal, method='percentile', logscale=False,
                    intervalLength=210, perc=90):
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
    if logscale:
        outputSignal = np.log(outputSignal)
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


def loc_extrema(data, window=882):
    """
    Calcula maximos y minimos locales
    Devuelve indices y valores (max y min)
    """
    indmax = argrelextrema(data, np.greater_equal, order=int(window))
    mmax = np.zeros(len(data))
    mmax[indmax[0]] = data[indmax[0]]
    indmin = argrelextrema(data, np.less_equal, order=int(window))
    mmin = np.zeros(len(data))
    mmin[indmin[0]] = data[indmin[0]]
    return(indmax[0], mmax, indmin[0], mmin)


def nmoment(x, counts, n):
    return np.sum(counts*((x-meanmoment(x, counts))/sigm(x,
                          counts))**n)/np.sum(counts)


def meanmoment(x, counts):
    return np.sum(x*counts)/np.sum(counts)


def sigm(x, counts):
    return np.sum(counts*(x-meanmoment(x, counts))**2/np.sum(counts))


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]


def normalizar(arr, minout=-1, maxout=1, pmax=100, pmin=5, method='extremos'):
    """
    Normaliza un array en el intervalo minout-maxout
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


# %%
f_path = '/media/juan/New Volume/Experimentos vS/Datos gogui/wetransfer-f237e5'

s_files = glob.glob(os.path.join(f_path, '*_s_*'))
v_files = [0]*len(s_files)
f_names = [x.split('_s_')[0] for x in s_files]
v_files_aux = glob.glob(os.path.join(f_path, '*_vs_*'))
i = 0
while i < len(s_files):
    indice = np.where([f_names[i] in x for x in v_files_aux])[0][0]
    v_files[i] = v_files_aux[indice]
    i += 1
s_file = s_files[1]
v_file = v_files[1]
fs, s_raw = wavfile.read('{}'.format(s_file))
s_raw = np.concatenate((s_raw[:int(3*fs)], s_raw[int(18*fs):int(21.5*fs)],
                        s_raw[int(31.5*fs):]))
fs, v_raw = wavfile.read('{}'.format(v_file))
v_raw = np.concatenate((v_raw[:int(3*fs)], v_raw[int(18*fs):int(21.5*fs)],
                        v_raw[int(31.5*fs):]))
time = np.arange(len(v_raw))/fs
# %%
v_filt = butter_lowpass_filter(v_raw, fs, order=5)
v_filt = butter_highpass_filter(v_filt, fs, order=6)
v_filt = butter_lowpass_filter(v_filt, fs, order=5)
v_filt = normalizar(v_filt)
v_slow = butter_lowpass_filter(v_filt, fs, lcutoff=500., order=5)
v_slow = normalizar(np.abs(v_slow))

percentil = 90
v_envelope = envelope_cabeza(v_filt, method='percentile', intervalLength=220,
                             perc=percentil)
v_envelope = normalizar(v_envelope, minout=0, maxout=1)
# %%
duration = 1.0
fs = 44150.0
samples = int(fs*duration)
t = np.arange(samples) / fs
sig = signal.chirp(t, 100.0, t[-1], 300.0)
sig *= signal.chirp(t, 5.0, t[-1], 30.0)
sig *= signal.chirp(t, 5.0, t[-1], 30.0)
# sig *= (1.0 + 0.5*np.sin(2.0*np.pi*60.0*t))

analytic_signal = signal.hilbert(sig)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = np.diff(instantaneous_phase) / (2.0*np.pi) * fs
fig = plt.figure()
ax0 = fig.add_subplot(211)
ax0.plot(t, sig, label='signal')
ax0.plot(t, amplitude_envelope, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1 = fig.add_subplot(212)
ax1.plot(t[1:], instantaneous_frequency)
ax1.set_xlabel("time in seconds")
ax1.set_ylim(0.0, 120.0)
# %%
fig, ax = plt.subplots(2, figsize=(10, 5), sharex=True)
hil_env = np.abs(signal.hilbert(v_filt))

ax[0].plot(time, v_filt)
ax[0].plot(time, v_envelope, 'g')
ax[0].plot(time, -v_envelope, 'g')
ax[1].plot(time, v_filt)
for n in range(1, 6):
    hil_filt = butter_lowpass_filter(hil_env, fs, order=5, lcutoff=100*n)
    ax[1].plot(time, hil_filt, label='Filtro {}Hz'.format(100*n))
ax[1].legend()
