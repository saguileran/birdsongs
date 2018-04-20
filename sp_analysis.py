# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 20:07:04 2016

@author: juan
"""

#%%
from scipy import stats
import os
import glob
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
import pandas as pd
import errno
from scipy import signal
from scipy.signal import argrelextrema
from scipy.signal import butter


def consecutive(data, stepsize=1):
    """
    Separa en segmentos de datos consecutivos. Se usa para separar indices
    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def running_mean(x, N):
    cumsum = np.cumsum(numpy.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def make_path(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


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
    nyq = 0.5 * fs
    normal_lcutoff = lcutoff/nyq
    bl, al = butter(order, normal_lcutoff, btype='low', analog=False)
    return bl, al


def butter_lowpass_filter(data, fs, lcutoff=3000.0, order=6):
    bl, al = butter_lowpass(fs, lcutoff, order=order)
    yl = signal.filtfilt(bl, al, data)
    return yl


def butter_highpass(fs, hcutoff=100.0, order=6):
    nyq = 0.5 * fs
    normal_hcutoff = hcutoff/nyq
    bh, ah = butter(order, normal_hcutoff, btype='high', analog=False)
    return bh, ah


def butter_highpass_filter(data, fs, hcutoff=100.0, order=5):
    bh, ah = butter_highpass(fs, hcutoff, order=order)
    yh = signal.filtfilt(bh, ah, data)
    return yh

#%%
birdname = 'AmaVio'
os.chdir('/home/juan/Documentos/Musculo/Codigo canarios/Files')
destino = '/home/juan/Documentos/Musculo/Codigo canarios/Preliminar'
make_path(destino)

#%%
fig, ax = plt.subplots(2, figsize=(10,4), sharex=True)
files = glob.glob(os.path.join(os.getcwd(), '*.wav'))
for wav in files:
    fs, data = wavfile.read(wav)
    time = np.linspace(0, np.double(len(data))/fs, len(data))
    ax[0].plot(time, data)
    NN = 1024
    overlap = 1/1.1
    sigma = NN/10
    fu, tu, Sxx = signal.spectrogram(data, fs, nperseg=NN, noverlap=NN*overlap,
                                     window=signal.get_window(('gaussian',
                                                               sigma), NN),
                                     scaling='spectrum')
    Sxx = np.clip(Sxx, a_min=np.amax(Sxx)*0.000001, a_max=np.amax(Sxx))
    ax[1].pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                     rasterized=True)
    ax[1].set_ylim(0, 8000)
    display.clear_output(True)
    display.display(gcf())
    plt.pause(0.01)

#%% Corto
fig, ax = plt.subplots(2, figsize=(10, 4), sharex=True)
fs, data = wavfile.read(files[0])
data = data[38*fs:45*fs]
time = np.linspace(0, np.double(len(data))/fs, len(data))
ax[0].plot(time, data)
NN = 1024
overlap = 1/1.1
sigma = NN/10
fu, tu, Sxx = signal.spectrogram(data, fs, nperseg=NN, noverlap=NN*overlap,
                                 window=signal.get_window(('gaussian', sigma),
                                                          NN),
                                 scaling='spectrum')
Sxx = np.clip(Sxx, a_min=np.amax(Sxx)*0.000001, a_max=np.amax(Sxx))
ax[1].pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                 rasterized=True)
ax[1].set_ylim(0, 8000)

#%% Maximos de frecuencia
fs, data = wavfile.read(files[0])
data = data[int(40.1*fs):int(45*fs)]
time = np.linspace(0, np.double(len(data))/fs, len(data))
sound = butter_highpass_filter(data, fs, order=6)
sound = sound/(max(abs(sound)))

NN = 1024
overlap = 1/1.1
sigma = NN/10
fu, tu, Sxx = signal.spectrogram(sound, fs, nperseg=NN, noverlap=NN*overlap,
                                 window=signal.get_window(('gaussian', sigma),
                                                          NN),
                                 scaling='spectrum')
Sxx = np.clip(Sxx, a_min=np.amax(Sxx)*0.000001, a_max=np.amax(Sxx))

sound_envelope = envelope_cabeza(sound, intervalLength=fs*0.01)
dn = 10
min_envelope = argrelextrema(sound_envelope[::dn], np.less_equal,
                             order=int(fs*0.04/(2*dn)))[0]
t_min_prel = min_envelope*dn
env_min_prel = sound_envelope[t_min_prel]
true_min = env_min_prel < np.histogram(env_min_prel)[1][1]
env_min = env_min_prel[true_min]
t_min = t_min_prel[true_min]

prob_intervals = [[t_min[nn], t_min[nn+1]] for nn in range(len(t_min)-1)]
syllables = []
envelope_threshold = 0.1
for candidate in prob_intervals:
    if max(sound_envelope[candidate[0]:candidate[1]]) > envelope_threshold:
        syllables.append(candidate)

nfigs = 7
fig, ax = plt.subplots(nfigs, figsize=(20, nfigs*1.5), sharex=True)
ax[0].plot(time, sound, label='Sound')
ax[1].plot(time, sound_envelope, 'g', label='Envelope')
ax[1].plot(time, -sound_envelope, 'g')

t_sil = np.asarray([int(0.5*(x[0]+x[1])) for x in syllables])
f_intersil = np.zeros(len(t_sil))
f_intersil[1:] = fs/np.diff(t_sil)
ax[2].plot(time[t_sil], f_intersil, 'g.', label='Frec intersil')
ax[2].set_ylim(3, 20)
for tmin in range(len(syllables)):
    ax[1].axvline(x=time[syllables[tmin][0]])
    ax[1].axvline(x=time[t_sil[tmin]], color='red')

denvelope = np.zeros(len(sound_envelope))
denvelope[1:] = np.diff(sound_envelope)
skewness = [stats.skew(denvelope[x[0]:x[1]]) for x in syllables]

ax[3].plot(time[t_sil], skewness, 'r.', label='derivative skewness')
ax[3].axhline(y=0)
max_frec = np.empty(len(tu))
max_value = np.empty(len(tu))
df = fu[1]
f_cut = 500
for tt in range(len(tu)):
    fourier = np.log(Sxx[:, tt])
    max_frec[tt] = (np.argmax(fourier[int(f_cut/df):])+int(f_cut/df))*df
    max_value[tt] = max(fourier[int(f_cut/df):])


ax[4].pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                 rasterized=True, label='Spectrogram')
ax[4].set_ylim(0, 8000)
# ax[3].plot(tu, max_frec, 'r.')
ax[5].plot(tu, max_value, 'b', label='Max power')

N_av = 6
smooth_max_frec = running_mean(max_frec, N_av)
dmax_frec = np.zeros(len(tu))
dmax_frec[N_av:] = np.diff(smooth_max_frec)
ax[6].plot(tu, dmax_frec, label='Max power deriv')
fig.tight_layout()
for axis in ax:
    axis.legend(loc='upper right')
