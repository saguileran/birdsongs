# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 00:53:47 2022

@author: sebas
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.signal import argrelextrema
from scipy.signal import butter
import peakutils


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



dir_path = 'C:\\Users\\sebas\\Documents\\GitHub\\BirdSongs-Audios-Copeton\\'#Colombianos\\'
dir_specto = 'C:\\Users\\sebas\\Documents\\GitHub\\BirdSongs-Audios-Copeton\\Spectrograms\\'#Colombianos\\Spectrograms\\'

files = []
[files.append(path.name) for path in os.scandir(dir_path) if path.name[-3:]=="wav"]


#file = "XC53006 - Rufous-collared Sparrow - Zonotrichia capensis.wav"#files[120]
files = files[:]
N = 512
for file in files:
    
    fs, data = wavfile.read(dir_path+file)
    if len(np.shape(data)) == 2 and np.shape(data)[1]==2: data = (data[:,0]+data[:,1])/2
    f, t, Sxx = get_spectrogram(data, fs, window=N, overlap=1/1.1, sigma=N/10)
    
    plt.pcolormesh(t, f, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                   rasterized=True)
    plt.xlabel("time (s)"); plt.ylabel("Frequency (Hz)");
    plt.colorbar()
    plt.savefig(dir_specto+file[:-3]+"png")
    plt.close('all') 