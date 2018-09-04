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
import os
import glob
from scipy.io import wavfile
from scipy.interpolate import interp1d


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


def consecutive(data, stepsize=1, min_length=1):
    """
    Parte una tira de datos en bloques de datos consecutivos.
    Ej:
        [1,2,3,4,6,7,9,10,11] -> [[1,2,3,4],[6,7],[9,10,11]]
    """
    candidates = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    return [x for x in candidates if len(x) > min_length]


# %%
# Nombre del ave
birdname = 'CeRo'
# Carpeta donde estan guardados los wavs
base_path = '/home/juan/Documentos/Musculo/Experimentos vS/CeRo/Playbacks/26082018/'
# files_path = '{}Files'.format(base_path)
files_path = '/home/juan/Documentos/Musculo/Experimentos vS/CeRo/Playbacks/26082018/'
# Carpeta donde se van a guardar los resultados
analisis_path = '{}analysis/'.format(base_path)
if not os.path.exists(analisis_path):
    os.makedirs(analisis_path)

# %% Cargo datos de sonido
# Busca todos los archivos del pajaro.
sound_files = glob.glob(os.path.join(files_path, '*'+birdname+'*BOS.wav'))
num_file = 0
file_id = sound_files[num_file].rsplit('/', 1)[1].split('.', 1)[0]
fs, song = wavfile.read(sound_files[num_file])
norm_song = normalizar(song)
envelope44k = np.empty(0)
envelope44k = np.concatenate((envelope44k,
                              normalizar(envelope_cabeza
                                         (song, intervalLength=0.01*fs),
                                         minout=0, method='extremos')))

time = np.linspace(0, len(song)/fs, len(song))
umbral = 0.05
supra = np.where(envelope44k > umbral)[0]
silabas = consecutive(supra, min_length=100)
fig, ax = plt.subplots(5, figsize=(10, 10), sharex=True)
ax[0].plot(time, norm_song)
ax[0].plot(time, envelope44k)
ax[0].plot(time, -envelope44k)
# %%
times = []
freqs = []
times_sil = []
freqs_sil = np.zeros_like(song)
freq_zc = np.zeros_like(song)
freq_max = 6000
max_sig = argrelextrema(song, np.greater, order=int(fs/freq_max))
max_pos = [x for x in max_sig[0] if song[x] > np.mean(song)]
for x in range(len(max_pos)-1):
    n_time = int((max_pos[x+1]+max_pos[x])/2)
    freq_zc[n_time] = fs/(max_pos[x+1]-max_pos[x])


def on_pick(event):
    global xmouse, ymouse
    global time
    global Sxx
    global df
    global tu
    global tu_closest
    global time_closest
    global f_event
    xmouse, ymouse = event.xdata, event.ydata
    tu_closest = np.argmin(abs(xmouse-tu))
    time_closest = np.argmin(abs(xmouse-time))
    aux = int(ymouse/df)
    ancho = 5
    f_event = df*(np.argmax(np.log(Sxx[:, tu_closest][aux-ancho:aux+ancho])) +
                  aux-ancho)
    ax[1].plot(tu[tu_closest], f_event, 'b.')
    plt.gcf().canvas.draw_idle()
    print('Guardar t = {:.3f} f = {:.0f}?'.format(time[time_closest], f_event))


def press(event):
    global times_sil
    global freqs_sil
    global times
    global freq_file
    global freqs
    global f
    global ax
    global fs
    global song
    if event.key == 'enter':
        ax[1].plot(time[time_closest], f_event, 'r.')
        plt.gcf().canvas.draw_idle()
        times.append(time_closest)
        freqs.append(f_event)
        print('Guardado en memoria')
    if event.key == '2':
        ax[1].plot(time[time_closest], f_event/2, 'r.')
        plt.gcf().canvas.draw_idle()
        times.append(time_closest)
        freqs.append(f_event/2)
        print('Guardado en memoria')
    if event.key == 'n':
        sorted_times = np.asarray(times)[np.argsort(times)]
        sorted_freqs = np.asarray(freqs)[np.argsort(times)]
        f_q = interp1d(sorted_times, sorted_freqs, kind='linear')
        i1 = sorted_times[0]
        i2 = sorted_times[-1]
        t_interp = time[i1+1:i2-1]
        f_interp = f_q([x for x in range(i1+1, i2-1)])
        ax[1].plot(t_interp, f_interp)
        freqs_sil[i1+1:i2-1] = f_interp
        plt.gcf().canvas.draw_idle()
        times = []
        freqs = []
        print('Nueva silaba')
    if event.key == 'escape':
        np.savetxt(freq_file, freqs_sil)
    if event.key == 'f':
        segment = song[time_closest-256:time_closest+256]
        fourier = np.abs(np.fft.rfft(segment))
        freqs = np.fft.rfftfreq(len(segment), d=1/fs)
        min_bin = np.argmin(np.abs(freqs-300))
        max_bin = np.argmin(np.abs(freqs-10000))
        fourier = np.abs(np.fft.rfft(segment))[min_bin:max_bin]
        freqs = np.fft.rfftfreq(len(segment), d=1/fs)[min_bin:max_bin]
        norm_fou = fourier/np.sum(fourier)
        entropy = -np.sum(norm_fou*np.log2(norm_fou))
        plt.figure()
        plt.plot(freqs, fourier, label='{}'.format(entropy))
        plt.title('{:.2f}'.format(time[tu_closest]))
        plt.legend()


NN = 1024
overlap = 1/1.1
sigma = NN/10
fu, tu, Sxx = get_spectrogram(song, fs, window=NN, overlap=overlap,
                              sigma=sigma)
df = np.diff(fu)[0]
fig, ax = plt.subplots(2, figsize=(12, 6), sharex=True)
ax[0].plot(time, song)
ax[1].pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                 rasterized=True)
ax[1].set_ylim(0, 8000)
ax[1].set_xlim(min(time), max(time))

freq_file = '{}fundamental_{}'.format(files_path, file_id)
if not os.path.isfile(freq_file):
    print('click = elegir punto; n = nueva silaba')
    print('enter = guardar; escape = guardar en file, f = fourier')
    fig.tight_layout()
    fig.canvas.callbacks.connect('button_press_event', on_pick)
    fig.canvas.callbacks.connect('key_press_event', press)
else:
    print('Cargando {}'.format(freq_file))
    freqs_sil = np.loadtxt(freq_file)
    ax[1].plot(time, freqs_sil, '.', ms=1)

# %% Integro hopf
fs = 44150
oversamp = 20
dt = 1./(oversamp*fs)

b_hopf = 0.
n_tot = int(len(freqs_sil))

out_size = n_tot
r_out = np.zeros(out_size)
tita_out = np.zeros(out_size)
tmax = out_size*oversamp

n_out = 0
mu = envelope44k[n_out]**2
omega = 2*np.pi*freqs_sil[n_out]

dp = 10
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
        r_out[n_out] = v[1]
        tita_out[n_out] = v[0]
        n_out += 1
        omega = 2*np.pi*freqs_sil[n_out]
        mu = envelope44k[n_out]**2
        v[1] = np.sqrt(mu)
        taux = 0
    taux += 1
    t += 1
    completion = t/tmax*100
    if np.isnan(v[1]):
        v[1] = 0
    if completion > porcentaje:
        print('{:.0f}% completado'.format(porcentaje))
        porcentaje += dp
out = r_out*np.cos(tita_out)
# %%
NN = 1024
overlap = 1/1.1
sigma = NN/10
fig, ax = plt.subplots(4, figsize=(12, 18), sharex=True)
ax[0].plot(time, song)
fu, tu, Sxx = get_spectrogram(song, fs, window=NN, overlap=overlap,
                              sigma=sigma)
ax[1].pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                 rasterized=True)
ax[1].set_ylim(0, 8000)
ax[2].plot(time, out)
fu_s2, tu_s2, Sxx_s2 = get_spectrogram(out, fs, window=NN,
                                       overlap=overlap, sigma=sigma)
ax[3].pcolormesh(tu_s2, fu_s2, np.log(Sxx_s2), cmap=plt.get_cmap('Greys'),
                 rasterized=True)
ax[3].set_ylim(0, 8000)

# %%
out_name = '{}{}-synth_hopf'.format(files_path, file_id)
n = 1
while os.path.isfile(out_name):
    n += 1
    out_name = '{}{}-synth_hopf-{}'.format(files_path, file_id, n)
wavfile.write('{}.wav'.format(out_name), fs,
              np.asarray(normalizar(out), dtype=np.float32))
