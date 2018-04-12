# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 20:07:04 2016

To do:
    - agregar representacion grafica de las silabas a partir de templates
    - matriz de transicion de distinto orden -> hecho en transitions_arb_order.py
    - agregar analisis en ventana de tiempo predefinida (p. ej horas) -> falta testear
@author: juan
"""

#%%
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import errno
from scipy import signal


def make_path(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def get_spectrogram(data, fs=44100, NFFT=1024, overlap=1/1.1, ss=10):
    sigma = NFFT/ss
    fu, tu, Sxx = signal.spectrogram(data, fs, nperseg=NFFT,
                                     noverlap=NFFT*overlap,
                                     window=signal.get_window(('gaussian',
                                                               sigma), NFFT),
                                     scaling='spectrum')
    Sxx = np.clip(Sxx, a_min=np.amax(Sxx)*0.000001, a_max=np.amax(Sxx))
    return fu, tu, Sxx


def normalizar(arr, minout=-1, maxout=1):
    norm_array = np.copy(np.asarray(arr, dtype=np.double))
    norm_array -= min(norm_array)
    norm_array = norm_array/max(norm_array)
    norm_array *= maxout-minout
    norm_array += minout
    return norm_array

#%%
birdname = 'AmaVio'
folders = glob.glob(os.path.join('/media/juan/New Volume/Experimentos vS/2018/canto/{}/wavs/'.format(birdname), '*'))
day = np.asarray([int(x.rsplit('-', 2)[1]) for x in folders])
max_order = 2

df = pd.DataFrame(columns=['File name', 'Song'])
fila = 1
files_path = folders[2]
log_file = glob.glob(os.path.join(files_path, '*song-log*'))[0]
df_log = pd.read_csv(log_file, header=0, sep=',')
all_wavs = list(set(df_log['File name']))
num_files = len(all_wavs)
syl_types = list(set(''.join(list(df_log['song']))))
syl_types.append('silence')
num_syl = len(syl_types)
transition_matrix = np.zeros((num_files, num_syl, num_syl))

df_statistics = pd.DataFrame(columns=['Day', 'Hour', 'Minute', 'Second',
                                      'Phrase transitions',
                                      'Syllable repetition'])

times = [x.split('_s_', 1)[0].rsplit('-', 2) for x in all_wavs]
months = [int(x[1]) for x in times]
days = [int(x[2][:2]) for x in times]
hours = [int(x[2][3:5]) for x in times]
minutes = [int(x[2][6:8]) for x in times]
seconds = [int(x[2][9:]) for x in times]

for n in range(len(all_wavs)):
    ff = all_wavs[n]
    for song in df_log['song'][df_log['File name'] == ff]:
        fin = len(song)
        inicio = 0
        row = syl_types.index('silence')
        column = syl_types.index(song[inicio])
        transition_matrix[all_wavs.index(ff)][row][column] += 1
        while inicio < fin-1:
            row = syl_types.index(song[inicio])
            column = syl_types.index(song[inicio + 1])
            transition_matrix[all_wavs.index(ff)][row][column] += 1
            inicio += 1
        row = syl_types.index(song[-1])
        column = syl_types.index('silence')
        transition_matrix[all_wavs.index(ff)][row][column] += 1
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(transition_matrix[all_wavs.index(ff)], cmap=plt.cm.Blues)
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    tick_list = np.arange(0, num_syl, 1)
    ax.set_xticks(tick_list)
    ax.set_xticklabels(syl_types)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks(tick_list)
    ax.set_yticklabels(syl_types)
    ax.set_ylabel('From')
    ax.set_xlabel('To')
#%% Testear
dtime = 3
groups_index = [np.where(abs(np.asarray(hours)-x) < dtime)[0] for x in hours]
transition_matrix_grouped = np.zeros((len(groups_index), num_syl, num_syl))
for n in len(groups_index):
    start_hour = min(hours[groups_index])
    end_hour = max(hours[groups_index])
    transition_matrix_grouped[n] = sum(transition_matrix[groups_index])
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(transition_matrix_grouped[n], cmap=plt.cm.Blues)
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    tick_list = np.arange(0, num_syl, 1)
    ax.set_xticks(tick_list)
    ax.set_xticklabels(syl_types)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks(tick_list)
    ax.set_yticklabels(syl_types)
    ax.set_title('From: {}, to: {}'.format(start_hour, end_hour))
    ax.set_ylabel('From')
    ax.set_xlabel('To')
