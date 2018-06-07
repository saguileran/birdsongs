#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 12:05:18 2018

Objetivo: Crear cantos sintéticos de canarios

@author: juan
"""

# %%
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.signal import argrelextrema
from scipy.signal import butter
import pandas as pd
import peakutils
from random import uniform
from scipy.interpolate import interp1d


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


def consecutive(data, stepsize=1, min_length=1):
    """
    Parte una tira de datos en bloques de datos consecutivos.
    Ej:
        [1,2,3,4,6,7,9,10,11] -> [[1,2,3,4],[6,7],[9,10,11]]
    """
    candidates = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    return [x for x in candidates if len(x) > min_length]


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


def sigmoid(x, dt=1, b=0, minout=0, maxout=1, fs=44150, rev=False):
    a = 5/(dt*fs)
    if not rev:
        return (1/(1+np.exp(-(a*x+b))))*(maxout-minout)+minout
    else:
        return ((1/(1+np.exp(-(a*x+b))))*(maxout-minout)+minout)[::-1]


def smooth_on_off(onoff, fs=44150, on_time=0.005):
    """
    Suaviza la envolvente on off en las transiciones
    """
    dif = np.diff(onoff)
    pos_dif = max(dif)
    neg_dif = min(dif)
    window = int(fs*on_time)
    smooth = np.copy(onoff)
    trans_on = np.where(dif == neg_dif)[0]
    trans_off = np.where(dif == pos_dif)[0]
    for n_on in range(len(trans_on)):
        start = trans_on[n_on] - window
        end = trans_on[n_on] + window
        if start > 0 and end < len(smooth):
            smooth[start:end] = sigmoid(-np.arange(-window, window, 1),
                                        dt=on_time, fs=fs,
                                        maxout=max(smooth[start:end]),
                                        minout=min(smooth[start:end]))
    for n_off in range(len(trans_off)):
        start = trans_off[n_off] - window
        end = trans_off[n_off] + window
        if start and end < len(smooth):
            smooth[start:end] = sigmoid(np.arange(-window, window, 1),
                                        dt=on_time/10, fs=fs,
                                        maxout=max(smooth[start:end]),
                                        minout=min(smooth[start:end]))
    return smooth


def smooth_trajectory(alfa, beta, fs=44150, on_time=0.001, slow_factor=10):
    """
    Suaviza las trayectorias de alfa-beta
    """
    dif_alfa = np.diff(alfa)
    pos_dif = max(dif_alfa)
    neg_dif = min(dif_alfa)
    window = int(fs*on_time*slow_factor)
    sm_alfa = np.copy(alfa)
    sm_beta = np.copy(beta)
    trans_on = np.where(dif_alfa == neg_dif)[0] + 1
    trans_off = np.where(dif_alfa == pos_dif)[0] + 1
    for n_on in range(len(trans_on)):
        astart = trans_on[n_on] - window
        aend = trans_on[n_on] + window
        bstart = trans_on[n_on] - 2*window
        bend = trans_on[n_on]
        if astart > 0 and aend < len(sm_alfa):
            sm_alfa[astart:aend] = sigmoid(np.arange(-window, window, 1),
                                           dt=on_time*slow_factor, fs=fs,
                                           rev=True,
                                           maxout=max(sm_alfa[astart:aend]),
                                           minout=min(sm_alfa[astart:aend]))
            sm_beta[bstart:bend] = sigmoid(np.arange(-window, window, 1),
                                           dt=on_time, fs=fs,
                                           rev=True,
                                           maxout=max(sm_beta[astart:aend]),
                                           minout=sm_beta[bend+1])
    for n_off in range(len(trans_off)):
        astart = trans_off[n_off] - window
        aend = trans_off[n_off] + window
        bstart = trans_off[n_off]
        bend = trans_off[n_off] + 2*window
        if astart > 0 and aend < len(sm_alfa):
            sm_alfa[astart:aend] = sigmoid(np.arange(-window, window, 1),
                                           dt=on_time, fs=fs,
                                           rev=False,
                                           maxout=max(sm_alfa[astart:aend]),
                                           minout=min(sm_alfa[astart:aend]))
            sm_beta[bstart:bend] = sigmoid(np.arange(-window, window, 1),
                                           dt=on_time*slow_factor, fs=fs,
                                           rev=False,
                                           maxout=max
                                           (sm_beta[bstart-1:bend+1]),
                                           minout=sm_beta[bstart-1])
    return sm_alfa, sm_beta


# %%
# Nombre del ave
birdname = 'AmaVio'
# Carpeta donde estan guardados los wavs
base_path = '/home/juan/Documentos/Musculo/Codigo canarios/'
files_path = '{}Files'.format(base_path)
# Carpeta donde se van a guardar los resultados
analisis_path = '{}analysis/'.format(base_path)
if not os.path.exists(analisis_path):
    os.makedirs(analisis_path)

# %% Cargo datos de sonido
# Busca todos los archivos del pajaro.
sound_files = glob.glob(os.path.join(files_path, '*'+birdname+'*_s*wav'))
song = np.empty(0)
envelope = np.empty(0)
# Elijo el primero de estos archivos
inicio = [28.4, 21., 7.5, 38.9, 28.]
fin = [37.5, 31.5, 13.8, 45., 30.]
num_file = 0
fs, s = wavfile.read(sound_files[num_file])
# Recorto el file
t_i = inicio[num_file]
t_f = fin[num_file]
s_cut = s[int(t_i*fs):int(t_f*fs)]
song = np.concatenate((song, s_cut))
envelope = np.concatenate((envelope,
                           normalizar(envelope_cabeza(s_cut,
                                                      intervalLength=0.01*fs),
                                      minout=0, method='extremos')))
time = np.linspace(0, len(song)/fs, len(song))
umbral = 0.05
supra = np.where(envelope > umbral)[0]
silabas = consecutive(supra, min_length=100)

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
    ax.plot(tu[tu_closest], f_event, 'b.')
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
    if event.key == 'enter':
        ax.plot(time[time_closest], f_event, 'r.')
        plt.gcf().canvas.draw_idle()
        times.append(time_closest)
        freqs.append(f_event)
        print('Guardado en memoria')
    if event.key == '2':
        ax.plot(time[time_closest], f_event/2, 'r.')
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
        ax.plot(t_interp, f_interp)
        freqs_sil[i1+1:i2-1] = f_interp
        plt.gcf().canvas.draw_idle()
        times = []
        freqs = []
        print('Nueva silaba')
    if event.key == 'escape':
        np.savetxt(freq_file, freqs_sil)


NN = 1024
overlap = 1/1.1
sigma = NN/10
fu, tu, Sxx = get_spectrogram(song, fs, window=NN, overlap=overlap,
                              sigma=sigma)
df = np.diff(fu)[0]
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
ax.pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
              rasterized=True)
ax.set_ylim(0, 8000)
ax.set_xlim(min(time), max(time))
for ss in silabas:
    ax.plot([time[ss[0]], time[ss[-1]]], [0, 0], 'k', lw=5)

freq_file = '{}/fundamental-1'.format(files_path)
aux = 1
if os.path.isfile(freq_file):
    var = input('Ya existe archivo de frecuencias,\
                sobreescribir [s], nuevo file [n], exit [e]? [s/n/e] ')
    if var == 's':
        print('click = elegir punto; n = nueva silaba')
        print('enter = guardar; escape = guardar en file')
        fig.tight_layout()
        fig.canvas.callbacks.connect('button_press_event', on_pick)
        fig.canvas.callbacks.connect('key_press_event', press)
    elif var == 'n':
        while os.path.isfile(freq_file):
            freq_file = 'fundamental-{}'.format(aux)
            aux += 1
        print('click = elegir punto; n = nueva silaba')
        print('enter = guardar; escape = guardar en file')
        fig.tight_layout()
        fig.canvas.callbacks.connect('button_press_event', on_pick)
        fig.canvas.callbacks.connect('key_press_event', press)
    elif var == 'e':
        freqs_sil = np.loadtxt(freq_file)
        ax.plot(time, freqs_sil, '.', ms=1)

# %% Cargo las grillas de valores
print('Buscando beta...')
# En el archivo esta todo (para todo alpha, beta, gamma)
grid_file = 'ff_SCI-all'
opt_gamma = 25000
opt_alpha = -0.05
bad_alpha = -0.5
df_all = pd.read_csv('{}{}'.format(base_path, grid_file)).fillna(0)

df_grid = df_all[(df_all['gamma'] == opt_gamma) &
                 (df_all['alpha'] == opt_alpha)]
last = np.where(df_grid['fundamental'] == 0)[0][1]
freq_table = df_grid['fundamental'][:last]
freq_table = freq_table.drop(freq_table.index[[-2, -3, -4, -5]])
alpha_table = df_grid['alpha'][:last]
beta_table = df_grid['beta'][:last]
beta_table = beta_table.drop(beta_table.index[[-2, -3, -4, -5]])
sci_table = df_grid['SCI'][:last]
sci_table = sci_table.drop(beta_table.index[[-2, -3, -4, -5]])
plt.figure()
plt.scatter(freq_table, beta_table, c=np.asarray(sci_table))
plt.colorbar()
plt.clim(1, 3)

df_bad = df_all[(df_all['gamma'] == opt_gamma) &
                (df_all['alpha'] == bad_alpha)]
last_bad = np.where(df_bad['fundamental'] == 0)[0][1]
freq_bad = df_bad['fundamental'][:last_bad]
beta_bad = df_bad['beta'][:last_bad]
sci_bad = df_bad['SCI'][:last_bad]
plt.figure()
plt.scatter(freq_bad, beta_bad, c=np.asarray(sci_bad))
plt.colorbar()
plt.clim(1, 3)
# %% Pre procesamiento beta-ff
unique_ff = list(set(freq_table))
ff = np.asarray(unique_ff)
beta = np.zeros_like(unique_ff)
# Promedio en valores repetidos de frecuencia
for n_ff in range(len(unique_ff)):
    frecuencia = ff[n_ff]
    beta[n_ff] = np.mean(beta_table[freq_table == frecuencia])
# Interpolo
f_q = interp1d(ff, beta, kind='linear', fill_value='extrapolate')
plt.figure()
plt.plot(freq_table, beta_table, 'o', label='Tabla')
plt.plot(ff, beta, 'o', label='Promedio')
new_ff = np.linspace(0, max(ff), num=100)
plt.plot(new_ff, f_q(new_ff), '--x', label='Interpolado')
plt.legend()

unique_ff_bad = list(set(freq_bad))
ff_b = np.asarray(unique_ff_bad)
beta_b = np.zeros_like(unique_ff_bad)
# Promedio en valores repetidos de frecuencia
for n_ff in range(len(unique_ff_bad)):
    frecuencia_b = ff_b[n_ff]
    beta_b[n_ff] = np.mean(beta_bad[freq_bad == frecuencia_b])
# Interpolo
f_q_b = interp1d(ff_b, beta_b, kind='linear', fill_value='extrapolate')
plt.figure()
plt.plot(freq_bad, beta_bad, 'o', label='Tabla')
plt.plot(ff_b, beta_b, 'o', label='Promedio')
new_ff = np.linspace(0, max(ff), num=100)
plt.plot(new_ff, f_q_b(new_ff), '--x', label='Interpolado')
plt.legend()
# %%
freq = np.loadtxt(freq_file)
env_amplitude = envelope

beta_file = '{}beta_song.dat'.format(analisis_path)
beta_file_b = '{}beta_song_b.dat'.format(analisis_path)

ap_alfa = np.zeros_like(freq)   # beta
ap_alfa += 0.05
ap_beta = np.zeros_like(freq)   # beta
ap_freq = np.zeros_like(freq)   # frequency

ap_alfa_b = np.zeros_like(freq)   # beta
ap_alfa_b += 0.05
ap_beta_b = np.zeros_like(freq)   # beta

for i in range(len(freq)):
    if freq[i] > 100:
        ap_freq[i] = freq[i]
        ap_beta[i] = f_q(freq[i])
        ap_alfa[i] = -0.05
        ap_beta_b[i] = f_q_b(freq[i])
        ap_alfa_b[i] = -0.5
    else:
        ap_beta[i] = -0.15
        ap_beta_b[i] = -0.15
        ap_freq[i] = 0.

np.savetxt(beta_file, ap_beta, fmt='%.2f')
np.savetxt(beta_file_b, ap_beta_b, fmt='%.2f')
# %%
smooth_alfa, smooth_beta = smooth_trajectory(ap_alfa, ap_beta, on_time=0.001,
                                             slow_factor=10)
plt.figure()
plt.plot(time, ap_beta, 'b--', label=r'$\beta$')
plt.plot(time, smooth_beta, 'b', label=r'$\beta_{smooth}$')
plt.plot(time, ap_alfa, 'r--', label=r'$\alpha$')
plt.plot(time, smooth_alfa, 'r', label=r'$\alpha_{smooth}$')
plt.legend()
plt.xlabel('Tiempo')
# %%
smooth_alfa_b, smooth_beta_b = smooth_trajectory(ap_alfa_b, ap_beta_b,
                                                 on_time=0.001, slow_factor=10)
plt.figure()
plt.plot(time, ap_beta_b, 'b--', label=r'$\beta$')
plt.plot(time, smooth_beta_b, 'b', label=r'$\beta_{smooth}$')
plt.plot(time, ap_alfa_b, 'r--', label=r'$\alpha$')
plt.plot(time, smooth_alfa_b, 'r', label=r'$\alpha_{smooth}$')
plt.legend()
plt.xlabel('Tiempo')

# %% Integro (bueno)
N_total = len(song)
# Sampleo
sampling = 44150
oversamp = 20
dt = 1./(oversamp*sampling)

out_size = int(N_total)
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
amplitud = env_amplitude[0]

n_out = 0

s1overCH = (36*2.5*2/25.)*1e09
s1overLB = 1.*1e-04
s1overLG = (50*1e-03)/.5
RB = 1*1e07
A1 = 0

gm = opt_gamma
alfa1 = smooth_alfa[0]
beta1 = smooth_beta[0]
out = np.zeros(out_size)
dp = 5

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
        n_out += 1
        beta1 = smooth_beta[n_out]
        alfa1 = smooth_alfa[n_out]
        amplitud = env_amplitude[n_out]
        taux = 0
        if (n_out*100/N_total) % dp == 0:
            print('{:.0f}%'.format(100*n_out/N_total))
    taux += 1
    if tiempot > 0.0:
        r = 0.1
        noise = 0.21*(uniform(0, 1) - 0.5)
        beta1 = beta1 + 0.0*noise
        s1overCH = (360/0.8)*1e08
        s1overLB = 1.*1e-04
        s1overLG = (1/82.)
        RB = (.5)*1e07
        rdis = (300./5.)*(10000.)
        A1 = amplitud + 0.0*noise
    t += 1
# %% Integro (malo)
N_total = len(song)
# Sampleo
sampling = 44150
oversamp = 20
dt = 1./(oversamp*sampling)

out_size = int(N_total)
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
amplitud = env_amplitude[0]

n_out = 0

s1overCH = (36*2.5*2/25.)*1e09
s1overLB = 1.*1e-04
s1overLG = (50*1e-03)/.5
RB = 1*1e07
A1 = 0

gm = opt_gamma
alfa1 = smooth_alfa_b[0]
beta1 = smooth_beta_b[0]
out_b = np.zeros(out_size)
dp = 5

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
        out_b[n_out] = preout*10
        n_out += 1
        beta1 = smooth_beta_b[n_out]
        alfa1 = smooth_alfa_b[n_out]
        amplitud = env_amplitude[n_out]
        taux = 0
        if (n_out*100/N_total) % dp == 0:
            print('{:.0f}%'.format(100*n_out/N_total))
    taux += 1
    if tiempot > 0.0:
        r = 0.1
        noise = 0.21*(uniform(0, 1) - 0.5)
        beta1 = beta1 + 0.0*noise
        s1overCH = (360/0.8)*1e08
        s1overLB = 1.*1e-04
        s1overLG = (1/82.)
        RB = (.5)*1e07
        rdis = (300./5.)*(10000.)
        A1 = amplitud + 0.0*noise
    t += 1
# %%
synth_env = normalizar(envelope_cabeza(out, intervalLength=0.01*fs),
                       minout=0, method='extremos')
out_amp = np.zeros_like(out)
not_zero = np.where(synth_env > 0.005)
out_amp[not_zero] = out[not_zero]*envelope[not_zero]/synth_env[not_zero]
s_amp_env = normalizar(envelope_cabeza(out_amp, intervalLength=0.01*fs),
                       minout=0, method='extremos')


# %%
synth_env_b = normalizar(envelope_cabeza(out_b, intervalLength=0.01*fs),
                         minout=0, method='extremos')
out_amp_b = np.zeros_like(out_b)
not_zero = np.where(synth_env_b > 0.005)
out_amp_b[not_zero] = out_b[not_zero]*envelope[not_zero]/synth_env_b[not_zero]
s_amp_env_b = normalizar(envelope_cabeza(out_amp_b, intervalLength=0.01*fs),
                         minout=0, method='extremos')
# %%
NN = 1024
overlap = 1/1.1
sigma = NN/10
fu, tu, Sxx = get_spectrogram(song, fs, window=NN, overlap=overlap,
                              sigma=sigma)
fu_s, tu_s, Sxx_s = get_spectrogram(out, fs, window=NN, overlap=overlap,
                                    sigma=sigma)
fu_s, tu_s, Sxx_s_b = get_spectrogram(out_b, fs, window=NN, overlap=overlap,
                                      sigma=sigma)

fig, ax = plt.subplots(3, 2, figsize=(18, 6), sharex=True, sharey='col')
ax[0][0].plot(time, normalizar(song), label='canto')
ax[0][0].plot(time, envelope)
ax[0][0].legend()
ax[1][0].plot(time, normalizar(out), label='output')
ax[1][0].plot(time, synth_env)
ax[1][0].legend()
ax[2][0].plot(time, normalizar(out_b), label='output (malo)')
ax[2][0].plot(time, s_amp_env_b)
ax[2][0].legend()
ax[0][1].pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                    rasterized=True)
ax[1][1].pcolormesh(tu_s, fu_s, np.log(Sxx_s), cmap=plt.get_cmap('Greys'),
                    rasterized=True)
ax[2][1].pcolormesh(tu_s, fu_s, np.log(Sxx_s_b), cmap=plt.get_cmap('Greys'),
                    rasterized=True)

ax[2][1].set_ylim(0, 8000)
ax[2][1].set_xlim(0, 6.5)
fig.tight_layout()
# %% Guardo wav
wavfile.write('{}/synth5.wav'.format(files_path), fs,
              np.asarray(normalizar(out), dtype=np.float32))
wavfile.write('{}/synth5b.wav'.format(files_path), fs,
              np.asarray(normalizar(out_b), dtype=np.float32))
wavfile.write('{}/song.wav'.format(files_path), fs,
              np.asarray(normalizar(song), dtype=np.float32))
