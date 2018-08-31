#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 12:05:18 2018

Objetivo: Crear cantos sintéticos de canarios

@author: juan
"""

# %%
from random import uniform
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import argrelextrema
from scipy.signal import butter
import pandas as pd
import peakutils
import os
from time import strftime, clock
from scipy import ndimage


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


def SpectralContent(filt_data, fs=44150, method='song', fmin=300, fmax=10000,
                    dt_transit=0.002, x_data=None):
    segment = filt_data[int(dt_transit*fs):]
    amp = max(segment)-min(segment)
    freqs = np.fft.rfftfreq(len(segment), d=1/fs)
    min_bin = np.argmin(np.abs(freqs-fmin))
    max_bin = np.argmin(np.abs(freqs-fmax))
    fourier = np.abs(np.fft.rfft(segment))[min_bin:max_bin]
    freqs = np.fft.rfftfreq(len(segment), d=1/fs)[min_bin:max_bin]
    f_msf = np.sum(freqs*fourier)/np.sum(fourier)
    f_aff = 0
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
    elif method == 'synth' and x_data is not None:
        segment = x_data[int(dt_transit*fs):]
        freqs = np.fft.rfftfreq(len(segment), d=1/fs)
        min_bin = np.argmin(np.abs(freqs-fmin))
        max_bin = np.argmin(np.abs(freqs-fmax))
        fourier = np.abs(np.fft.rfft(segment))[min_bin:max_bin]
        freqs = np.fft.rfftfreq(len(segment), d=1/fs)[min_bin:max_bin]
        maximos = peakutils.indexes(fourier, thres=0.05, min_dist=5)
        if amp < 500:
            f_aff = 0
            amp = 0
        elif len(maximos) > 0:
            f_aff = freqs[maximos[0]]
    return f_msf, f_aff, amp


# %% Sintetizo en un rango grande y me armo una base de datos
os.chdir('/home/juan/Documentos/Musculo/Codigo canarios')
outfile = 'ff_SCI-{}'.format(strftime("%Y-%m-%d.%H.%M.%S"))

n_alphas = 100
n_betas = 2*n_alphas
n_gammas = 6
alphas = np.linspace(-0.5, 0, n_alphas, endpoint=False)
betas = np.linspace(-0.5, 0.5, n_betas, endpoint=False)
agrid, bgrid = np.meshgrid(alphas, betas)
ab_grid = np.c_[np.ravel(bgrid), np.ravel(agrid)]
gammas = np.linspace(25000, 50000, n_gammas)
gammas = np.asarray([50000., 42500., 45000., 60000., 70000., 80000., 90000.,
                     100000.])
N_total = len(alphas)*len(betas)*len(gammas)

# Sampleo
sampling = 44150
oversamp = 20
dt = 1./(oversamp*sampling)
dt_per_param = 0.02
n_per_param = int(dt_per_param*sampling)

# -- por parametro -- #
df = pd.DataFrame(index=range(N_total),
                  columns=['alpha', 'beta', 'gamma', 'fundamental', 'msf',
                  'SCI', 'amplitud'])
n_param = 0
dp = 1
for gm in gammas:
    for beta1, alfa1 in ab_grid:
        out_size = int(n_per_param)
        tmax = out_size*oversamp

        v = np.zeros(5)
        v[0] = 0.000000000005
        v[1] = 0.00000000001
        v[2] = 0.000000000001
        v[3] = 0.00000000001
        v[4] = 0.000000000001

        forcing1 = 0.
        forcing2 = 0.
        tiempot = 0.
        tcount = 0
        noise = 0
        tnoise = 0
        r = 0.4
        rdis = 7000
        gamma2 = 1.
        gamma3 = 1.
        atenua = 1
        c = 35000.

        a = np.zeros(tmax)
        db = np.zeros(tmax)

        ancho = 0.2
        largo = 3.5
        tau = int(largo/(c*dt + 0.0))
        t = tau
        taux = 0
        amplitud = 1

        n_out = 0

        s1overCH = (36*2.5*2/25.)*1e09
        s1overLB = 1.*1e-04
        s1overLG = (50*1e-03)/.5
        RB = 1*1e07
        A1 = 0
        out = np.zeros(out_size)
        x_out = np.zeros(out_size)
        while t < tmax and v[1] > -5000000:
            dbold = db[t]
            a[t] = (.50)*(1.01*A1*v[1]) + db[t-tau]
            db[t] = -r*a[t-tau]
            ddb = (db[t]-dbold)/dt  # Derivada
            forcing1 = db[t]
            forcing2 = ddb
            PRESSURE = a[t]
            tiempot += dt
            rk4(dxdt_synth, v, 5, t + 0.0, dt)
            noise = 0
            preout = RB*v[4]
            if taux == oversamp:
                out[n_out] = preout*10
                x_out[n_out] = v[1]
                n_out += 1
                taux = 0
            taux += 1
            if tiempot > 0.0:
                # alfa1 = -0.15-0.00*amplitud
                r = 0.1
                noise = 0.21*(uniform(0, 1) - 0.5)
                beta1 = beta1 + 0.0*noise
                s1overCH = (360/0.8)*1e08
                s1overLB = 1.*1e-04
                s1overLG = (1/82.)
                RB = (.5)*1e07
                rdis = (300./5.)*(10000.)
                A1 = amplitud + 0.5*noise
            t += 1
        msf, ff, amp = SpectralContent(out, sampling, method='synth',
                                       x_data=x_out)
        SCI = 0
        if ff != 0:
            SCI = msf/ff
        df.iloc[n_param] = [alfa1, beta1, gm, ff, msf, SCI, amp]
        n_param += 1
        if (n_param*100/N_total) % dp == 0:
            print('{:.0f}%'.format(100*n_param/N_total))
            df.to_csv(outfile)
# %% ff, SCI, Amplitud
df = pd.read_csv('ff_SCI-all_2')
fig, ax = plt.subplots(6, figsize=(12, 18), sharex=True)
ax[0].plot(df['alpha'], '.')

ax[1].plot(df['beta'], '.')

ax[2].plot(df['gamma'], '.')

ax[3].plot(df['fundamental'], '.')
ax[3].set_ylabel('fundamental')

ax[4].plot(df['SCI'])
ax[4].set_ylabel('SCI')

ax[5].plot(df['amplitud'])
ax[5].set_ylabel('Amplitud')

# %%
gammas = df.groupby('gamma')['gamma'].unique()  # unique values of gamma
gms = gammas
for gg in gms:
    gama = gg[0]
    df_aux = df_fit[df_fit['gamma'] == gama]
    alfa_f = df_aux['alfa']
    beta_f = df_aux['beta']

    df_aux = df[df['gamma'] == gama]
    df_aux = df_aux.fillna(0)
    ff = df_aux['fundamental'].astype(float)
    SCI = df_aux['SCI'].astype(float)
    amp = df_aux['amplitud'].astype(float)
    alfa = df_aux['alpha'].astype(float)
    beta = df_aux['beta'].astype(float)

    fig, ax = plt.subplots(1, 3, figsize=(12, 9))
    fig.suptitle('gamma = {:.0f}'.format(gama))

    cax = ax[0].imshow(ff.values.reshape(len(set(beta)), -1), origin='lower',
                       extent=[min(alfa), max(alfa), min(beta), max(beta)],
                       interpolation='nearest', cmap='hot')
    fig.colorbar(cax, ax=ax[0], fraction=0.085, pad=0.04)
    ax[0].scatter(alfa_f, beta_f)
    ax[0].set_xlabel('alpha')
    ax[0].set_ylabel('beta')
    ax[0].set_title('fundamental')
    cax = ax[1].imshow(SCI.values.reshape(len(set(beta)), -1), origin='lower',
                       extent=[min(alfa), max(alfa), min(beta), max(beta)],
                       interpolation='nearest', cmap='hot')
    fig.colorbar(cax, ax=ax[1], fraction=0.085, pad=0.04)
    ax[1].set_xlabel('alpha')
    ax[1].set_ylabel('beta')
    ax[1].set_title('SCI')

    cax = ax[2].imshow(amp.values.reshape(len(set(beta)), -1), origin='lower',
                       extent=[min(alfa), max(alfa), min(beta), max(beta)],
                       interpolation='nearest', cmap='hot')
    fig.colorbar(cax, ax=ax[2], fraction=0.085, pad=0.04)
    ax[2].set_xlabel('alpha')
    ax[2].set_ylabel('beta')
    ax[2].set_title('amplitud')

    fig.tight_layout()
#    fig.savefig('espacio_parametros_gamma{}_fit_extendido.pdf'.format(gama),
#                format='pdf')
