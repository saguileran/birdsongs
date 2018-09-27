#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:12:25 2018

Funciones para analisis

@author: juan
"""
import numpy as np
from scipy import signal
from scipy.signal import butter
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt


def consecutive(data, stepsize=1, min_length=1):
    """
    Parte una tira de datos en bloques de datos consecutivos.
    Ej:
        [1,2,3,4,6,7,9,10,11] -> [[1,2,3,4],[6,7],[9,10,11]]
    """
    candidates = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    return [x for x in candidates if len(x) > min_length]


def envelope_cabeza(signal, method='percentile', intervalLength=210, perc=90,
                    step=1):
    """
    Calcula envolvente. En segmentos de intervalLength calcula el maximo
    (si no se especifica metodo) o el percentil especificado. El argumento step
    determina cada cuantos puntos calcula (subsampling)
    """
    if method == 'percentile':
        pp = perc
    else:
        pp = 100
    absSignal = abs(signal)
    dt2 = int(intervalLength/2)
    outputSignal = np.zeros(len(absSignal))
    outputSignal[:] = np.nan
    outputSignal[0] = absSignal[0]
    outputSignal[-1] = absSignal[-1]

    for baseIndex in range(1, len(absSignal)-1, int(step)):
        if baseIndex < dt2:
            percentil = np.percentile(absSignal[:baseIndex], pp)
        elif baseIndex > len(absSignal) - dt2:
            percentil = np.percentile(absSignal[baseIndex:], pp)
        else:
            percentil = np.percentile(absSignal[baseIndex-dt2:baseIndex+dt2],
                                      pp)
        outputSignal[baseIndex] = percentil
    return outputSignal[~np.isnan(outputSignal)]


def envelope_hilbert(data, samp_rate=44150, f_corte=100.):
    """
    Calcula envolvente por medio de la transformada de hilbert. Primero calcula
    la transformada de hilbert y toma valor absoluto. Luego aplica un filtro
    pasa bajo de frecuencia f_corte.

    data(np.array) = señal para calcular la envolvente
    samp_rate(float) = frecuencia de sampleo de la señal
    f_corte(float) = frecuencia de corte del filtro pasa bajo para la
    transformada de hilbert.

    Devuelve hil_env(np.array)
    """
    hil_env = np.abs(signal.hilbert(data))
    hil_env = butter_lowpass_filter(hil_env, samp_rate, order=5,
                                    lcutoff=f_corte)
    return hil_env


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
                    sigma=102.4, plot=False, fmax=8000):
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
    if plot:
        plt.figure()
        plt.pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                       rasterized=True)
        plt.ylim(0, fmax)
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
