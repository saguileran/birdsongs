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
from scipy.signal import argrelextrema
from matplotlib.patches import Ellipse
from matplotlib.ticker import FormatStrFormatter


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
v_filt -= np.mean(v_filt)
v_filt /= max(abs(v_filt))
percentil = 90
v_envelope = envelope_cabeza(v_filt, method='percentile', intervalLength=220,
                             perc=percentil)
v_envelope = normalizar(v_envelope, minout=0, maxout=1)
# %%
fig, ax = plt.subplots(3, figsize=(10, 8), sharex=True)

fu, tu, Sxx = get_spectrogram(s_raw, fs)
ax[0].pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                 rasterized=True)
ax[0].set_ylim(0, 8000)
ax[1].set_xlim(0, max(time))
ax[1].plot(time, v_envelope, 'g')
ax[1].plot(time, -v_envelope, 'g')

tstart = [0.2, 1., 1.5, 4., 5.1, 6., 7.1, 8.7, 9.5, 15.2, 17., 18.5,
          20., 21.]
cc = ['k', 'r', 'g', 'c', 'k', 'm', 'c', 'k', 'b', 'c', 'k', 'm', 'g', 'y']
nstart = [int(x*fs) for x in tstart]
tend = [x+0.2 for x in tstart]
nend = [int(x*fs) for x in tend]
correlacion = []
end = max(time)
dt = int(0.001*fs)
for n in range(len(nstart)):
    n_start = nstart[n]
    n_end = nend[n]
    fragment = v_envelope[n_start:n_end]
    time_window = len(fragment)
    ax[0].axvspan(xmin=time[n_start], xmax=time[n_end], color=cc[n], alpha=0.3)
    corr_aux = []
    t_corr = []
    left = 0
    while left + time_window < int(end*fs):
        right = left + time_window
        t_corr.append((left+time_window/2)/fs)
        corr_aux.append(np.corrcoef(fragment, v_envelope[left:right])[0][1])
        left += dt
    corr_aux = np.asarray(corr_aux)
    correlacion.append(corr_aux)
    t_corr = np.asarray(t_corr)
    ax[2].plot(t_corr[np.where(corr_aux > 0.5)],
               corr_aux[np.where(corr_aux > 0.5)], color=cc[n],
               marker='.', ls='None')
# %%
distinct = [0, 1, 2, 3, 5, 8, 13]
fig, ax = plt.subplots(len(distinct), len(distinct), figsize=(14, 14))
for ncol in range(len(distinct)):
    nc = distinct[ncol]
    n_start = nstart[nc]
    n_end = nend[nc]
    fragment = v_envelope[n_start:n_end]
    ax[-1][ncol].plot(fragment)
    ax[-1][ncol].axes.get_xaxis().set_visible(False)
    ax[-1][ncol].axes.get_yaxis().set_visible(False)
    ax[-1][ncol].set_title('{}'.format(nc))
    ax[ncol][0].plot(fragment)
    ax[ncol][0].axes.get_xaxis().set_visible(False)
    ax[ncol][0].axes.get_yaxis().set_visible(False)
    ax[ncol][0].set_title('{}'.format(nc))
    for nrow in range(ncol):
        nr = distinct[nrow]
        ax[nrow][ncol].plot(correlacion[nr], correlacion[nc])
        ax[nrow][ncol].axvline(x=0.6, color='r')
        ax[nrow][ncol].axhline(y=0.6, color='r')
        ax[nrow][ncol].set_xlim(-1, 1)
        ax[nrow][ncol].set_ylim(-1, 1)
        ax[nrow][ncol].set_xlim(-1, 1)
        ax[nrow][ncol].set_title('{}-{}'.format(nr, nc))
        ax[nrow][ncol].axes.get_xaxis().set_visible(False)
        ax[nrow][ncol].axes.get_yaxis().set_visible(False)
fig.tight_layout()
# %% Simple measures (ACA ESTA LA PAPA)
tempend = [0.7, 1.4, 2.8, 4.7, 5.7, 6.5, 7.5, 9.15, 10.2, 15.8, 18., 18.9,
           20.4, 21.7]
fig, ax = plt.subplots(len(distinct), 6, figsize=(15, 20), sharey=True,
                       sharex='col')
nend = [int(x*fs) for x in tempend]
time_temp = []
templates = []
ttomax = []
vmax = []
skewness = []
tmean = []
tsk = []
colors = [cc[x] for x in distinct]
for n in range(len(distinct)):
    auxtomax = []
    auxvmax = []
    auxsk = []
    auxtmean = []
    auxtsk = []
    time_temp.append(time[nstart[distinct[n]]:nend[distinct[n]]])
    templates.append(v_envelope[nstart[distinct[n]]:nend[distinct[n]]])
    ax[n][0].plot(time_temp[-1]-time_temp[-1][0], templates[-1])
    ix, vx, ii, vn = loc_extrema(templates[-1], window=fs*0.025)
    ax[n][0].axes.get_xaxis().set_visible(False)
    ax[n][0].set_ylabel('{}'.format(distinct[n]))
    det = signal.detrend(autocorr(templates[-1]))
    indmax, valmax, indmin, valmin = loc_extrema(det, window=fs*0.025)
    ind_breaks_prop = np.arange(np.argmin(templates[-1][:indmax[1]]),
                                len(templates[-1]), indmax[1])
    ind_breaks = [ind_breaks_prop[0]]
    for nn in range(1, len(ind_breaks_prop)):
        prop = ind_breaks_prop[nn]
        real = ii[np.argmin(abs(ii-prop))]
        ind_breaks.append(real)
    for nn in range(len(ind_breaks)-1):
        ax[n][0].axvline(x=time_temp[-1][ind_breaks[nn]]-time_temp[-1][0],
                         color='g')
        austart = ind_breaks[nn]
        ausend = ind_breaks[nn+1]
        ax[n][1].plot(templates[-1][austart:ausend], 'k', lw=2, alpha=0.3)
        auxtomax.append(np.argmax(templates[-1][austart:ausend]))
        auxvmax.append(np.max(templates[-1][austart:ausend]))
        auxsk.append(auxtomax[-1]/(ausend-austart))
        auxtmean.append(meanmoment(time_temp[-1][austart:ausend] -
                                   time_temp[-1][austart],
                                   templates[-1][austart:ausend]))
        auxtsk.append(nmoment(time_temp[-1][austart:ausend] -
                              time_temp[-1][austart],
                              templates[-1][austart:ausend], 3))
    if n == len(distinct)-1:
        ax[n][0].set_xlim(0, np.min([len(x) for x in templates])/fs)
    ax[n][1].axes.get_xaxis().set_visible(False)
    ax[n][1].axes.get_yaxis().set_visible(False)
    ttomax.append(auxtomax)
    vmax.append(auxvmax)
    skewness.append(auxsk)
    tmean.append(auxtmean)
    tsk.append(auxtsk)
    ax[n][2].plot(ttomax[-1], vmax[-1], color=colors[n], marker='.', ls='None')
    ax[n][2].plot(np.mean(ttomax[-1]), np.mean(vmax[-1]), color=colors[n],
                  marker='x')
    ell = Ellipse(xy=(np.mean(ttomax[-1]), np.mean(vmax[-1])),
                  width=np.std(ttomax[-1]),
                  height=np.std(vmax[-1]))
    ell.set_facecolor('none')
    ax[n][2].add_artist(ell)
    ax[n][2].set_xlim(0, 2500)
    ax[n][2].set_xlabel('Time to max')
    ax[n][2].set_ylabel('Max value')
    ax[n][2].xaxis.set_major_locator(plt.MaxNLocator(3))
    ax[n][2].axes.get_yaxis().set_visible(False)
    ax[n][3].plot(skewness[-1], vmax[-1], color=colors[n], marker='.',
                  ls='None')
    ax[n][3].plot(np.mean(skewness[-1]), np.mean(vmax[-1]), color=colors[n],
                  marker='x')
    ell = Ellipse(xy=(np.mean(skewness[-1]), np.mean(vmax[-1])),
                  width=np.std(skewness[-1]),
                  height=np.std(vmax[-1]))
    ell.set_facecolor('none')
    ax[n][3].add_artist(ell)
    ax[n][3].set_xlabel('Time to max/Duration')
    ax[n][3].set_ylabel('Max value')
    ax[n][3].set_xlim(0., 1.)
    ax[n][3].xaxis.set_major_locator(plt.MaxNLocator(3))
    ax[n][3].axes.get_yaxis().set_visible(False)
    ax[n][4].plot(tmean[-1], vmax[-1], color=colors[n], marker='.',
                  ls='None')
    ax[n][4].set_xlabel('Time mean')
    ax[n][4].xaxis.set_major_locator(plt.MaxNLocator(3))
    ax[n][4].axes.get_yaxis().set_visible(False)
    ax[n][5].plot(tsk[-1], vmax[-1], color=colors[n], marker='.',
                  ls='None')
    if n == len(tempend)-1:
        for nn in range(len(tempend)):
            ax[nn][5].set_xlim(min([min(x) for x in tsk]),
                               max([max(x) for x in tsk]))
    ax[n][5].set_xlabel('Time skewness')
    ax[n][5].xaxis.set_major_locator(plt.MaxNLocator(3))
    ax[n][5].axes.get_yaxis().set_visible(False)
    ax[n][5].xaxis.set_major_formatter(FormatStrFormatter('%.e'))
fig.tight_layout()
# %% Figuras juntas
figall, axall = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for n in range(len(colors)):
    axall[0].plot(ttomax[n], vmax[n], color=colors[n], marker='.', ls='None')
    axall[0].plot(np.mean(ttomax[n]), np.mean(vmax[n]), color=colors[n],
                  marker='x')
    ell = Ellipse(xy=(np.mean(ttomax[n]), np.mean(vmax[n])),
                  width=np.std(ttomax[n]),
                  height=np.std(vmax[n]))
    ell.set_facecolor('none')
    ell.set_edgecolor(colors[n])
    axall[0].add_artist(ell)
    axall[0].set_xlabel('Time to max')
    axall[0].set_ylabel('Max value')
    axall[1].plot(skewness[n], vmax[n], color=colors[n], marker='.',
                  ls='None')
    axall[1].plot(np.mean(skewness[n]), np.mean(vmax[n]), color=colors[n],
                  marker='x')
    ell = Ellipse(xy=(np.mean(skewness[n]), np.mean(vmax[n])),
                  width=np.std(skewness[n]),
                  height=np.std(vmax[n]))
    ell.set_facecolor('none')
    ell.set_edgecolor(colors[n])
    axall[1].add_artist(ell)
    axall[1].set_xlabel('Time to max/Duration')
    axall[1].set_ylabel('Max value')

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
# %% De noche ...
n_file = 'cG01_NN_2017-01-18_22_10_48_vs_21_denoised.wav'

fs, vn_raw = wavfile.read('{}/{}'.format(f_path, n_file))
vn_raw = vn_raw[:int(3*fs)]
ntime = np.arange(len(vn_raw))/fs
percentil = 90
vn_envelope = envelope_cabeza(vn_raw, method='percentile', intervalLength=220,
                              perc=percentil)
vn_envelope = normalizar(vn_envelope, minout=0, maxout=1)

fig, ax = plt.subplots(1+len(distinct), 2,
                       gridspec_kw={'width_ratios': [5, 1]},
                       figsize=(10, 8), sharex='col')
ax[0][0].set_xlim(0, max(ntime))
ax[0][0].plot(ntime, vn_envelope, 'g')
ax[0][0].plot(ntime, -vn_envelope, 'g')
correlacion = []
times_corr = []
for n in range(len(distinct)):
    temp = templates[n][:min((int(0.2*fs), len(templates[n])))]
    ax[n+1][1].plot(list(temp)*3, color=cc[distinct[n]])
    corr_aux = []
    end = max(ntime)
    dt = int(0.001*fs)
    time_window = len(temp)
    t_corr = []
    left = 0
    while left + time_window < int(end*fs):
        right = left + time_window
        t_corr.append((left+time_window/2)/fs)
        corr_aux.append(np.corrcoef(temp, vn_envelope[left:right])[0][1])
        left += dt
    corr_aux = np.asarray(corr_aux)
    correlacion.append(corr_aux)
    t_corr = np.asarray(t_corr)
    times_corr.append(t_corr)
    ax[n+1][0].axhline(y=0.6, color=cc[distinct[n]])
    ax[n+1][0].plot(t_corr, corr_aux, color=cc[distinct[n]])
    ax[n+1][0].set_ylim(-1, 1)
