#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:08:33 2018

@author: juan
"""

import numpy as np
import os
import glob
import pandas as pd
from scipy.io import wavfile
from scipy import signal
from scipy.signal import butter
import random
import matplotlib.pyplot as plt
from analysis_functions import get_spectrogram, consecutive


def search_file(filename, search_path):
    """ Given a search path, find file with requested name """
    for root, dir, files in os.walk(search_path):
        candidate = os.path.join(root, filename)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def NextPowerOfTwo(number):
    return int(np.ceil(np.log2(number)))


def n_pad_Pow2(arr):
    nextPower = NextPowerOfTwo(len(arr))
    deficit = int(np.power(2, nextPower) - len(arr))
    return deficit


def checkIfPow2(n):
    return bool(n and not (n & (n-1)))


def butter_highpass(data, fs, hcutoff=100.0, order=6):
    nyq = 0.5*fs
    normal_hcutoff = hcutoff/nyq
    bh, ah = butter(order, normal_hcutoff, btype='high', analog=False)
    return bh, ah


def butter_highpass_filter(data, fs, hcutoff=100.0, order=5):
    bh, ah = butter_highpass(fs, hcutoff, order=order)
    yh = signal.filtfilt(bh, ah, data)
    return yh


def resample(data, fs, new_fs=44150):
    resampled = signal.resample(data, int(len(data)*new_fs/fs))
    return resampled


def butter_lowpass(data, fs, lcutoff=3000.0, order=15):
    nyq = 0.5*fs
    normal_lcutoff = lcutoff/nyq
    bl, al = butter(order, normal_lcutoff, btype='low', analog=False)
    return bl, al


def butter_lowpass_filter(data, fs, lcutoff=3000.0, order=6):
    bl, al = butter_lowpass(data, fs, lcutoff, order=order)
    yl = signal.filtfilt(bl, al, data)
    return yl


def calculate_envelope(data, fs, method='hilbert', f_corte=80, logenv=False,
                       pow2pad=True):
    n_pad = 0
    n_dat = len(data)
    if pow2pad and not checkIfPow2(n_dat):
        n_pad = n_pad_Pow2(data)
    elif len(data) % 2 == 1:
        n_pad = 1
    envelope = np.abs(signal.hilbert(data, n_dat+n_pad))
    envelope = butter_lowpass_filter(envelope, fs, order=5, lcutoff=f_corte)
    if logenv:
        envelope = np.log(envelope)
    if method != 'hilbert':
        print('Hilbert is the only available method (and what you got)')
    if n_pad > 0:
        envelope = envelope[:-n_pad]
    return envelope


def envelope_spectrogram(time, data, fs, tstep=0.01, sigma_factor=5,
                         plot=False, fmin=0, fmax=50, freq_resolution=5):
    """
    Espectrograma de la envolvente

    Parameters
    ----------
    tstep(float):
        Paso temporal del espectrograma

    sigma_factor(float):
        Relacion entre el tamaño de la ventana y la dispersion. Se usa
        ventana gaussiana

    plot(boolean):
        Plotear?

    fmin(float):
        minima frecuencia del grafico

    fmax(float):
        maxima frecuencia del grafico

    freq_resolution(float):
        Resolucion en frecuencia

    Returns
    -------
    fu(array):
        array de frecuencias

    tu(array):
        array de tiempos

    Sxx(array x array):
        intensidad
    """
    time_win = 1/freq_resolution
    window = int(fs*time_win)
    overlap = 1-(tstep/time_win)
    sigma = window/sigma_factor
    envelope = calculate_envelope()
    fu, tu, Sxx = signal.spectrogram(envelope, fs,
                                     nperseg=window,
                                     noverlap=window*overlap,
                                     window=signal.get_window
                                     (('gaussian', sigma), window),
                                     scaling='spectrum')
    Sxx = np.clip(Sxx, a_min=np.amax(Sxx)*0.0001, a_max=np.amax(Sxx))
    if plot:
        fig, ax = plt.subplots(2, figsize=(16, 4), sharex=True)
        ax[0].plot(time, data)
        ax[0].plot(time, envelope)
        ax[1].pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                         rasterized=True)
        ax[1].set_ylim(fmin, fmax)
        fig.tight_layout()
    return fu, tu, Sxx


def get_file_spectrogram(data, fs, window=1024, overlap=1/1.1, sigma=102.4,
                         plot=False, fmin=0, fmax=8000):
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
    fu, tu, Sxx = signal.spectrogram(data, fs, nperseg=window,
                                     noverlap=window*overlap,
                                     window=signal.get_window
                                     (('gaussian', sigma), window),
                                     scaling='spectrum')
    Sxx = np.clip(Sxx, a_min=np.amax(Sxx)*0.000001, a_max=np.amax(Sxx))
    if plot:
        plt.figure()
        plt.pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                       rasterized=True)
        plt.ylim(fmin, fmax)
    return fu, tu, Sxx


def get_intersilabic_freq(data, min_value=0.03):
    supra_umbral = consecutive(np.where(data > min_value)[0])
    peaks = [supra_umbral[i][np.argmax(data[val])] for i, val
             in enumerate(supra_umbral)]
    return peaks


def wavplot(data, fs, plotEnvelope=False, subsampling=1, plot_peaks=False,
            min_value=0.03):
    plt.figure()
    time = np.arange(len(data))/fs
    plt.plot(time, data, alpha=0.5)
    if plotEnvelope:
        envelope = calculate_envelope(data=data, fs=fs)
        plt.plot(time, envelope)
    if plot_peaks:
        peaks = get_intersilabic_freq(data=data, min_value=min_value)
        plt.plot(time[peaks], data[peaks], '.')
        plt.twinx()
        plt.plot(time[peaks][:-1], 1/np.diff(time[peaks]), 'o')
    return 0


def normalizar(arr, minout=-1, maxout=1, pmax=100, pmin=5, method='extremos',
               zeromean=True):
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
    if zeromean:
        norm_array -= np.mean(norm_array)
    return norm_array


# %%
os.chdir('/home/juan/Documentos/Musculo/Codigo canarios/')
exp_folder = '/media/juan/New Volume/Experimentos vS/2018/MaVio/2018-10-10-day'
wavs = glob.glob(os.path.join(exp_folder, 'vs*.wav'))

fs, data = wavfile.read(random.choice(wavs))
wavplot(data, fs, plotEnvelope=False, plot_peaks=True, min_value=20000)
#calculate_envelope(data, fs)
norm_data = normalizar(data)
plt.figure()
plt.hist(norm_data, bins='auto')
