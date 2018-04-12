# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 20:07:04 2016

To do:
    - agregar representacion grafica de las silabas a partir de templates
    - matriz de transicion de distinto orden -> done
    - agregar analisis en ventana de tiempo predefinida (p. ej horas) ->
    falta testear
    - convertir de numero a probabilidad de transicion
@author: juan
"""

# %%
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import errno
from scipy import signal
from sklearn.preprocessing import normalize


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

# %%
birdname = 'AmaVio'
base_path = '/media/juan/New Volume/Experimentos vS/2018/'
folders = glob.glob(os.path.join
                    ('{}canto/{}/wavs/'.format(base_path, birdname), '*'))
day = np.asarray([int(x.rsplit('-', 2)[1]) for x in folders])
ordenes = [1]
order_hier = 'syllable'
order_hier = 'phrase'

fila = 1
files_path = folders[2]
log_file = glob.glob(os.path.join(files_path, '*song-log*'))[0]
df_log = pd.read_csv(log_file, header=0, sep=',')
all_wavs = list(df_log['File name'])
num_files = len(all_wavs)
# num_songs =
syl_types = list(set(''.join(list(df_log['song']))))
syl_types.append('silence')

num_syl = len(syl_types)
# Transition matrix, numbers
n_tr_matrix = np.zeros((num_files, len(ordenes), num_syl, num_syl))
n_tr_matrix_norep = np.zeros((num_files, len(ordenes), num_syl, num_syl))
# Transition matrix, probabilities
p_tr_matrix = np.zeros((num_files, len(ordenes), num_syl, num_syl))
p_tr_matrix_norep = np.zeros((num_files, len(ordenes), num_syl, num_syl))

times = [x.split('_s_', 1)[0].rsplit('-', 2) for x in all_wavs]
months = np.asarray([int(x[1]) for x in times])
days = np.asarray([int(x[2][:2]) for x in times])
hours = np.asarray([int(x[2][3:5]) for x in times])

# De cada file me armo una matriz. Podria armar por bout?
for i_order in range(len(ordenes)):
    n_order = ordenes[i_order]
    for n in range(len(all_wavs)):
#        ff = all_wavs[n]
#        for song in df_log['song'][df_log['File name'] == ff]:
        song = df_log['song'][n]
        song_array = np.asarray([int(x) for x in song])
        transitions_index = np.where([song[n] != song[n+1]
                                      for n in range(len(song)-1)])[0]
        if order_hier == 'phrase':
            song = ''.join([song[x] for x in transitions_index])
        fin = len(song)
        if fin > n_order:
            inicio = n_order - 1
            row = syl_types.index('silence')
            column = syl_types.index(song[inicio])
            n_tr_matrix[n][i_order][row][column] += 1
            while inicio < fin-n_order:
                row = syl_types.index(song[inicio])
                column = syl_types.index(song[inicio + n_order])
                n_tr_matrix[n][i_order][row][column] += 1
                inicio += 1
            row = syl_types.index(song[-n_order])
            column = syl_types.index('silence')
            n_tr_matrix[n][i_order][row][column] += 1
        p_tr_matrix[n][i_order] = \
            normalize(n_tr_matrix[n][i_order],
                      axis=1, norm='l1')
        reps = np.diag(np.diag(n_tr_matrix[n][i_order]))
        n_tr_matrix_norep[n][i_order] = \
            n_tr_matrix[n][i_order] - reps
        p_tr_matrix_norep[n][i_order] = \
            normalize(n_tr_matrix_norep[n][i_order],
                      axis=1, norm='l1')
        fig, ax = plt.subplots(ncols=2, figsize=(16, 8))
        cax = ax[0].matshow(p_tr_matrix[n][i_order],
                            cmap=plt.cm.Blues)
        fig.colorbar(cax, ax=ax[0], fraction=0.046, pad=0.04)
        tick_list = np.arange(0, num_syl, 1)
        ax[0].set_xticks(tick_list)
        ax[0].set_xticklabels(syl_types)
        ax[0].xaxis.set_ticks_position('bottom')
        ax[0].set_yticks(tick_list)
        ax[0].set_yticklabels(syl_types)
        ax[0].set_ylabel('From')
        ax[0].set_xlabel('To')
        ax[0].set_title('{}\nTransiciones de orden {}, tipo: {}'.format(song, n_order, order_hier))
        cax = ax[1].matshow(p_tr_matrix_norep[n][i_order],
                            cmap=plt.cm.Blues)
        fig.colorbar(cax, ax=ax[1], fraction=0.046, pad=0.04)
        ax[1].set_xticks(tick_list)
        ax[1].set_xticklabels(syl_types)
        ax[1].xaxis.set_ticks_position('bottom')
        ax[1].set_yticks(tick_list)
        ax[1].set_yticklabels(syl_types)
        ax[1].set_ylabel('From')
        ax[1].set_xlabel('To')
        ax[1].set_title('Sin repeticiones')
        fig.tight_layout()
# %% Testear
dtime = 3
groups_index = [np.where(np.logical_and(hours-x <= dtime, hours >= x))[0] for x in hours[:-1]]
n_tr_matrix_grouped = np.zeros((len(groups_index), num_syl, num_syl))
for n in range(len(groups_index)):
    i_grupo = groups_index[n]
    start_hour = hours[i_grupo]
    end_hour = max(hours[i_grupo])
    n_tr_matrix_grouped[n] = sum(n_tr_matrix[i_grupo])
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(n_tr_matrix_grouped[n], cmap=plt.cm.Blues)
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
