#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 18:03:50 2018

@author: juan
"""
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.signal import butter


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
s_file = s_files[0]
v_file = v_files[0]
fs, s_raw = wavfile.read('{}'.format(s_file))
s_raw = s_raw[:10*fs]
fs, v_raw = wavfile.read('{}'.format(v_file))
v_raw = v_raw[:10*fs]
time = np.arange(len(v_raw))/fs
# %%
v_filt = butter_lowpass_filter(v_raw, fs, order=5)
v_filt = butter_highpass_filter(v_filt, fs, order=6)
v_filt = butter_lowpass_filter(v_filt, fs, order=5)
v_filt -= np.mean(v_filt)
v_filt /= max(abs(v_filt))
# %%
fig, ax = plt.subplots(3, figsize=(10, 8), sharex=True)

fu, tu, Sxx = get_spectrogram(s_raw, fs)
ax[0].pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                 rasterized=True)
ax[0].set_ylim(0, 8000)
ax[1].plot(time, v_filt, 'b')
ax[1].set_xlim(0, 10)
# %%
percentil = 90
v_envelope = envelope_cabeza(v_filt, method='percentile', intervalLength=110,
                             perc=percentil)
v_envelope = normalizar(v_envelope, minout=0, maxout=1)
# %%
ax[1].plot(time, v_envelope, 'g')
ax[1].plot(time, -v_envelope, 'g')

# %%
template_starts = [4.83]
template_ends = [4.93]
start = 0.
end = 10.
dt = int(0.1*fs)
step = 5
for f_start in template_starts:
    n_start = int(f_start*fs)
    f_end = template_ends[template_starts.index(f_start)]
    n_end = int(f_end*fs)
    fragment = v_envelope[n_start:n_end]
    time_window = len(fragment)
    ax[1].plot(time[n_start:n_end], fragment)
    correlacion = []
    t_corr = []
    left = int(start*fs)
    while left + time_window < int(end*fs):
        right = left + time_window
        t_corr.append((left+time_window/2)/fs)
        correlacion.append(np.corrcoef(fragment, v_envelope[left:right])[0][1])
        left += step
    correlacion = np.asarray(correlacion)
    t_corr = np.asarray(t_corr)
    ax[2].plot(t_corr, correlacion)
    ax[2].plot(t_corr[np.where(correlacion > 0.6)],
               correlacion[np.where(correlacion > 0.6)], 'r.')

# %% Warpeo
template_starts = [4.83, 7.8]
template_ends = [4.93, 7.9]
start = 0.
end = 10.
dt = int(0.1*fs)
step = 10
compression = np.logspace(np.log2(0.7), -np.log2(0.7), 9, base=2)
fig, ax = plt.subplots(2+len(compression), figsize=(10, 20), sharex=True)

fu, tu, Sxx = get_spectrogram(s_raw, fs)
ax[0].pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                 rasterized=True)
ax[0].set_ylim(0, 8000)
log_envelope = np.log(v_envelope)
log = True
if log:
    eenv = np.log(v_envelope)
else:
    eenv = 1*v_envelope

ax[1].set_xlim(0, 10)
correlacion = [0]*len(compression)
t_corr = [0]*len(compression)
indice = 0
for f_start in template_starts:
    for i in range(len(compression)):
        t_corr[i] = []
        correlacion[i] = []
        n_start = int(f_start*fs)
        f_end = template_ends[template_starts.index(f_start)]
        n_end = int(f_end*fs)
        fragment = eenv[n_start:n_end]
        res_fragment = signal.resample(fragment,
                                       int(len(fragment)*compression[i]))
        time_window = len(res_fragment)
        ax[1].plot(time[n_start:n_end], fragment)
        left = int(start*fs)
        while left + time_window < int(end*fs):
            right = left + time_window
            t_corr[i].append((left+time_window/2)/fs)
            correlacion[i].append(np.corrcoef(res_fragment,
                                              eenv[left:right])[0][1])
            left += step
        correlacion[i] = np.asarray(correlacion[i])
        t_corr[i] = np.asarray(t_corr[i])
#        ax[2+i].plot(t_corr[i], correlacion[i])
        ax[2+i].plot(t_corr[i][np.where(correlacion[i] > 0.8)],
                     correlacion[i][np.where(correlacion[i] > 0.8)], '.',
                     label='{:.2f}'.format(len(res_fragment)/len(fragment)))
        ax[2+i].set_ylim(-1, 1)
        ax[2+i].yaxis.set_major_locator(plt.MaxNLocator(4))
        ax[2+i].legend()
fig.tight_layout()
ax[1].plot(time, eenv, zorder=0)
# %% SVD

dtime = 0.4
tstep = dtime/5
ntime = int(dtime*fs)
nstep = int(tstep*fs)
n_filas = (len(v_envelope)-ntime)//nstep
vecs = np.asmatrix([normalizar(v_envelope[n*nstep:n*nstep+ntime], minout=0,
                               maxout=1) for n in range(n_filas)])
U_matrix = np.matmul(vecs, np.transpose(vecs))
eigval, eigvec = np.linalg.eig(U_matrix)
base = np.matmul(eigvec, vecs)
base_vecs = [np.asarray(x)[0] for x in base]
plt.matshow(U_matrix)
plt.colorbar()

n = 5
fig, ax = plt.subplots(n, figsize=(10, 20))
for i in range(n):
    ax[i].plot(np.arange(len(base_vecs[i]))/fs, base_vecs[i])
fig.tight_layout()
