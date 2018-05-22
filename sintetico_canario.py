#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 12:05:18 2018

Objetivo: Crear cantos sintéticos de canarios

@author: juan
"""

# %%
from random import uniform
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import pandas as pd


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

# %%
# Nombre del ave
birdname = 'AmaVio'

# Carpeta donde estan guardados los wavs
files_path = '/home/juan/Documentos/Musculo/Codigo canarios/Files'

# Carpeta donde se van a guardar los resultados
analisis_path = '{}analysis/'.format(files_path)
if not os.path.exists(analisis_path):
    os.makedirs(analisis_path)

# %% Cargo datos de sonido y vs
print('Cargando datos...')
# Busca todos los archivos del pajaro.
sound_files = glob.glob(os.path.join(files_path,
                                     '*'+birdname+'*_s*wav'))
# Elijo el primero de estos archivos
num_file = 0
fs, song = wavfile.read(sound_files[num_file])

f_name_long = sound_files[num_file]
f_name = f_name_long.rsplit('/', 1)[1].split('_s_', 1)[0]

# Recorto el file
# t_i = 1.
# t_f = 3.
# song = song[int(t_i*fs):int(t_f*fs)]
time = np.linspace(0, len(song)/fs, len(song))

print('Datos ok')
# %% Calculo frecuencia fundamental del canto...
# ff = ...
ff_file = '{}fundamental_{}.dat'.format(analisis_path, f_name)
# np.savetxt(ff_file, ff, fmt='%.2f')
# %% Aproximo beta por tabla
print('Buscando beta...')
# Uso la tabla extendida para baja frecuencia
table_file = '{}OEC.new.ext.dat'.format(files_path)
alpha_table = np.loadtxt(table_file, usecols=(0,))
beta_table = np.loadtxt(table_file, usecols=(1,))
freq_table = np.loadtxt(table_file, usecols=(2,))
sci_table = np.loadtxt(table_file, usecols=(3,))

freq = np.loadtxt(ff_file)
# env_amplitude = np.loadtxt(onoff_file)
env_amplitude = np.loadtxt(ff_file)
beta_file = '{}beta_song_{}.dat'.format(analisis_path, f_name)

ap_alpha = np.zeros(len(freq))  # alpha
ap_beta = np.zeros(len(freq))   # beta
ap_freq = np.zeros(len(freq))   # frequency
ap_sci = np.zeros(len(freq))    # sci

for i in range(len(freq)):
    if freq[i] > 10:
        n_min = np.argmin(abs(freq_table - freq[i]))
        ap_alpha[i] = alpha_table[n_min]
        ap_beta[i] = beta_table[n_min]
        ap_freq[i] = freq_table[n_min]
        ap_sci[i] = sci_table[n_min]
    else:
        ap_alpha[i] = 0.15
        ap_beta[i] = 0.15
        ap_freq[i] = 0.
        ap_sci[i] = 0.
np.savetxt(beta_file, ap_beta, fmt='%.2f')
print('Beta ok')
# %% Integro modelo
print('Integrando modelo fonacion...')
size = len(freq)

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
dt = 1./(882000)
to = size*20
c = 35000.

# ------ GAMMA ------#
gm = 24000.
# ------ ***** ------#

a = np.zeros(to)
db = np.zeros(to)
df = np.zeros(to)

ancho = 0.2
largo = 3.5

tau = int(largo/(c*dt + 0.0))

t = tau
taux = 0

n_beta = 0
beta1 = ap_beta[n_beta]
amplitud = env_amplitude[n_beta]
alfa1 = 0.150

s1overCH = (36*2.5*2/25.)*1e09
s1overLB = 1.*1e-04
s1overLG = (50*1e-03)/.5
RB = 1*1e07
A1 = 0

out = np.zeros(size)
dp = 5
porcentaje = dp
while t < to and v[1] > -5000000:
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
    if taux == 20:
        n_beta += 1
        beta1 = ap_beta[n_beta]
        amplitud = env_amplitude[n_beta]
        out[n_beta] = preout*10
        taux = 0
    taux += 1
    if tiempot > 0.0:
        alfa1 = -0.15-0.00*amplitud
        r = 0.1
        noise = 0.21*(uniform(0, 1) - 0.5)
        beta1 = beta1 + 0.01*noise
        s1overCH = (360/0.8)*1e08
        s1overLB = 1.*1e-04
        s1overLG = (1/82.)
        RB = (.5)*1e07
        rdis = (300./5.)*(10000.)
        A1 = amplitud + 0.5*noise
    t += 1
    completion = t/to*100
    if completion > porcentaje:
        print('{:.0f}% completado'.format(porcentaje))
        porcentaje += dp
# %% Guardo
outfile = '{}synth_{}'.format(analisis_path, f_name)
normalized_out = normalizar(out)
np.savetxt(outfile, out)
wavfile.write('{}.wav'.format(outfile), 44100,
              np.asarray(normalized_out, dtype=np.float32))
print('Modelo fonacion ok')
