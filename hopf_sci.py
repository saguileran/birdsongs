#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 12:05:18 2018

Objetivo: Crear cantos sintéticos de canarios

@author: juan
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import argrelextrema


def hopf(v, dv):
    r = v[1]
    dv[0] = omega + b_hopf*r**2
    dv[1] = r*(mu - r**2)
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


def SpectralContent(segment, fs, synth=False):
    dn_seg = len(segment)//2
    fourier = np.abs(np.fft.rfft(segment))[5:dn_seg]
    freqs = np.fft.rfftfreq(len(segment), d=1/fs)[5:dn_seg]
    limite = np.argmin(np.abs(freqs-10000))
    f_msf = np.sum(freqs[:limite]*fourier[:limite])/np.sum(fourier[:limite])
    f_aff = 0
    if not synth:
        f_aff = freqs[np.argmax(fourier*(freqs/(freqs+500)**2))]
    else:
        maximos = argrelextrema(fourier, np.greater, order=5)
        if len(maximos[0]) > 0:
            f_aff = freqs[maximos[0][0]]
    return f_msf, f_aff


# %% Integro hopf
# Sampleo
fs = 44150
oversamp = 20
dt = 1./(oversamp*fs)

n_params = 10
b_hopf = 0.
omegas = 2*np.pi*np.linspace(10, 3000, n_params)
#mus = np.linspace(-.1, 10, n_params)
mus = [.1]
dt_all_param = [20*2*np.pi/x for x in omegas]*len(mus)
n_all_param = [int(x*fs) for x in dt_all_param]
dt_tot = sum(dt_all_param)
n_tot = int(dt_tot*fs)

SCI = np.zeros(len(omegas)*len(mus))
ff = np.zeros(len(omegas)*len(mus))
t_sci = np.zeros(len(omegas)*len(mus))
n_sci = 0

out_size = n_tot
tita_out = np.zeros(out_size)
r_out = np.zeros(out_size)
mu_out = np.zeros(out_size)
omega_out = np.zeros(out_size)

time = np.arange(0, out_size/fs, 1/fs)
tmax = out_size*oversamp

n_out = 0
nparam = 0
omega = omegas[nparam]
mu = mus[nparam]

dt_per_param = dt_all_param[nparam]
n_per_param = int(dt_per_param*fs)
n_this_param = 0
dp = 20
porcentaje = dp
taux = 0
t = 0

v = np.zeros(2)
v[0] = 0.
v[1] = np.sqrt(mu)
if np.isnan(v[1]):
    v[1] = 0
while t < tmax and v[1] > -5000000:
    rk4(hopf, v, 2, t + 0.0, dt)
    if taux == oversamp:
        tita_out[n_out] = v[0]
        r_out[n_out] = v[1]
        omega_out[n_out] = omega
        mu_out[n_out] = mu
        n_out += 1
        n_this_param += 1
        if n_this_param == n_per_param:
            nparam += 1
            v = np.zeros(2)
            omega = omegas[nparam % len(omegas)]
            mu = mus[nparam//len(omegas)]
            dt_per_param = dt_all_param[nparam]
            n_per_param = int(dt_per_param*fs)
            v[0] = 0.
            v[1] = np.sqrt(mu)
            n_this_param = 0
        taux = 0
    taux += 1
    t += 1
    completion = t/tmax*100
    if np.isnan(v[1]):
        v[1] = 0
    if completion > porcentaje:
        print('{:.0f}% completado'.format(porcentaje))
        porcentaje += dp
# %%
ff_hopf = np.ones_like(ff)
msf = np.ones_like(ff)
SCI_hopf = np.ones_like(ff)
for n_sci in range(len(omegas)*len(mus)):
    n_start = int(sum(n_all_param[:n_sci]))
    segment = (r_out*np.cos(tita_out))[n_start:n_start+n_all_param[n_sci]]
    msf[n_sci], ff[n_sci] = SpectralContent(segment, fs, synth=True)
    ff_hopf[n_sci] = omega_out[n_start+int(n_all_param[n_sci]/2)]/(2*np.pi)
    t_sci[n_sci] = (n_start+int(n_all_param[n_sci]/2))/fs
SCI_aff = msf/ff
SCI_hopf = msf/ff_hopf
NN = 1024
overlap = 1/1.1
sigma = NN/10
fu, tu, Sxx = get_spectrogram(r_out*np.cos(tita_out), fs, window=NN,
                              overlap=overlap, sigma=sigma)
fig, ax = plt.subplots(4, figsize=(12, 18))
ax[0].pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                 rasterized=True)
ax[0].set_ylim(0, max(omegas[:-1])/(2*np.pi))
ax[0].set_xlim(min(time), max(time))
ax[0].plot(time[:len(omega_out)], omega_out/(2*np.pi))

ax[1].plot(t_sci, msf, '.')
ax[1].set_xlim(min(time), max(time))
ax[1].set_xlabel('tiempo (s)')
ax[1].set_ylabel('msf')

ax[2].plot(ff, SCI_aff, '.')
ax[2].set_xlabel('Fundamental')
ax[2].set_ylabel('SCI (aff)')

ax[3].plot(ff_hopf, SCI_hopf, '.')
ax[3].set_xlabel('Fundamental')
ax[3].set_ylabel('SCI (hopf)')
