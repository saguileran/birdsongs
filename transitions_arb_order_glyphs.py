# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 20:07:04 2016

To do:
    - agregar representacion grafica de las silabas a partir de templates
    - matriz de transicion de distinto orden -> done
    - revisar orden phrase-syllable
    - agregar analisis en ventana de tiempo predefinida (p. ej horas) -> test
@author: juan
"""

# %%
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import wavfile
import errno
from scipy import signal
from sklearn.preprocessing import normalize
import itertools as it
from collections import OrderedDict


def make_path(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def get_spectrogram(data, fs=44150, NFFT=1024, overlap=1/1.1, ss=10):
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
base_path = '/home/juan/Documentos/Musculo/Codigo canarios/Files'
template_folder = '{}/Templates'.format(base_path)
# folders = glob.glob(os.path.join
#                    ('{}/canto/{}/wavs/'.format(base_path, birdname), '*'))
# day = np.asarray([int(x.rsplit('-', 2)[1]) for x in folders])
# ------------------#
ordenes = [1]
n_ordenes = len(ordenes)
# order_hier = 'syllable'
order_hier = 'phrase'
# ------------------#
# Ej. song = ABCDE a orden 2
# single: analizo A -> C; B -> D ...
# combined: analizo AB -> C; BC -> D ...
order_type = 'single'
order_type = 'combined'
# ------------------#
files_path = base_path
log_file = glob.glob(os.path.join(files_path, '*song-log*'))[0]
df_log = pd.read_csv(log_file, header=0, sep=',')
all_wavs_wrep = list(df_log['File name'])
all_wavs = list(set(all_wavs_wrep))
num_files = len(all_wavs)

song_times = [x.split('_s_', 1)[0].rsplit('-', 2) for x in all_wavs_wrep]
months = np.asarray([int(x[1]) for x in song_times])
days = np.asarray([int(x[2][:2]) for x in song_times])
hours = np.asarray([int(x[2][3:5]) for x in song_times])
# ------------------#
syl_types = list(set(''.join(list(df_log['Song']))))

# Creo figura de templates
fig, ax = plt.subplots(2, len(syl_types), figsize=(30, 4))
NN = 1024
overlap = 1/1.1
sigma = NN/10

for n in range(len(syl_types)):
    syl_file = '{}/silaba_{}.wav'.format(template_folder, syl_types[n])
    if os.path.isfile(syl_file):
        samp, temp = wavfile.read(syl_file)
        ax[0][n].plot(temp)
        ax[0][n].set_xticklabels([])
        ax[0][n].set_yticklabels([])
        ax[0][n].set_title(syl_types[n])
        fu, tu, Sxx = signal.spectrogram(temp, samp, nperseg=NN,
                                         noverlap=NN*overlap,
                                         window=signal.get_window
                                         (('gaussian', sigma), NN),
                                         scaling='spectrum')
        Sxx = np.clip(Sxx, a_min=np.amax(Sxx)*0.000001, a_max=np.amax(Sxx))
        ax[1][n].pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                            rasterized=True)
        ax[1][n].set_ylim(0, 8000)
        ax[1][n].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[1][n].xaxis.set_major_locator(plt.MaxNLocator(2))
fig.tight_layout()
# Ahora 0 va a ser silencio
# syl_types.append('s')
num_syl = len(syl_types)
glyphs_to = [x for x in syl_types]
orden = ordenes[0]
combinations = list(it.product(glyphs_to, repeat=orden))
glyphs_from = [''.join(list(x)) for x in combinations]
num_glyphs_to = len(glyphs_to)
num_glyphs_from = len(glyphs_from)
# ------------------#
# Transition matrix, numbers
n_tr_matrix = np.zeros((num_files, n_ordenes, num_glyphs_from, num_glyphs_to))
n_tr_matrix_norep = np.zeros((num_files, n_ordenes, num_glyphs_from,
                              num_glyphs_to))
# Transition matrix, probabilities
p_tr_matrix = np.zeros((num_files, n_ordenes, num_glyphs_from, num_glyphs_to))
p_tr_matrix_norep = np.zeros((num_files, n_ordenes, num_glyphs_from,
                              num_glyphs_to))
#%%
# De cada file me armo una matriz
min_length = 3
for i_order in range(n_ordenes):
    n_order = ordenes[i_order]
    for n_file in range(num_files):
        ff = all_wavs[n_file]
        print(ff)
        for song_aux in df_log['Song'][df_log['File name'] == ff]:
            song_array = np.asarray([x for x in song_aux])
            transitions_index = np.where([song_aux[n] != song_aux[n+1]
                                          for n in range(len(song_aux)-1)])[0]
            if order_hier == 'phrase':
                song = ''.join([song_aux[x] for x in transitions_index])
                if len(transitions_index > 0):
                    song += song_aux[transitions_index[-1]+1]
            elif order_hier == 'syllable':
                song = song_aux
            print(song)
            fin = len(song)
            if fin > n_order and fin > min_length:
                inicio = 0
                while inicio < fin-n_order:
                    row = glyphs_from.index(song[inicio:inicio + n_order])
                    column = glyphs_to.index(song[inicio + n_order])
                    n_tr_matrix[all_wavs.index(ff)][i_order][row][column] += 1
                    inicio += 1
            else:
                print('Descartada!\n')
        ocurring = np.where(n_tr_matrix[n_file][i_order] != 0)
        glyphs_from_oc = list(OrderedDict.fromkeys(ocurring[0]))
        num_glyphs_from_oc = len(glyphs_from_oc)
        p_tr_matrix[n_file][i_order] = \
            normalize(n_tr_matrix[n_file][i_order],
                      axis=1, norm='l1')
        fig, ax = plt.subplots(figsize=(8, 8))
        cax = ax.matshow(p_tr_matrix[n_file][i_order][glyphs_from_oc],
                         cmap=plt.cm.Blues)
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        tick_list_to = np.arange(0, num_glyphs_to, 1)
        tick_list_from = np.arange(0, num_glyphs_from_oc, 1)
        ax.set_xticks(tick_list_to)
        ax.set_xticklabels(glyphs_to)
        ax.xaxis.set_ticks_position('bottom')
        ax.set_yticks(tick_list_from)
        ax.set_yticklabels([glyphs_from[x] for x in glyphs_from_oc])
        ax.set_ylabel('From')
        ax.set_xlabel('To')
        ax.set_title('{}\n Orden {}, tipo: {}'.format(ff.split('.wav')[0],
                     n_order, order_hier))
        fig.tight_layout()
        if n_order == 1 and order_hier == 'syllable':
            reps = np.diag(np.diag(n_tr_matrix[n_file][i_order]))
            n_tr_matrix_norep[n_file][i_order] = \
                n_tr_matrix[n_file][i_order] - reps
            p_tr_matrix_norep[n_file][i_order] = \
                normalize(n_tr_matrix_norep[n_file][i_order],
                          axis=1, norm='l1')
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            cax2 = ax2.matshow(p_tr_matrix_norep[n_file][i_order],
                               cmap=plt.cm.Blues)
            fig2.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
            ax2.set_xticks(tick_list_to)
            ax2.set_xticklabels(glyphs_to)
            ax2.xaxis.set_ticks_position('bottom')
            ax2.set_yticks(tick_list_from)
            ax2.set_yticklabels(glyphs_from)
            ax2.set_ylabel('From')
            ax2.set_xlabel('To')
            ax2.set_title('Sin repeticiones')
            fig2.tight_layout()
# %% Testear
dtime = 3
groups_index_aux = [np.where(np.logical_and(hours-x <= dtime, hours >= x))[0]
                    for x in hours[:-1]]
groups_index = [groups_index_aux[0]]
for n in range(1, len(groups_index_aux)):
    if not np.array_equal(groups_index_aux[n], groups_index_aux[n-1]):
        groups_index.append(groups_index_aux[n])
n_tr_matrix_grouped = np.zeros((len(groups_index), n_ordenes, num_glyphs_from,
                                num_glyphs_to))
p_tr_matrix_grouped = np.zeros((len(groups_index), n_ordenes, num_glyphs_from,
                                num_glyphs_to))
for i_order in range(n_ordenes):
    n_order = ordenes[i_order]
    if len(groups_index) > 1:
        for n_group in range(len(groups_index)):
            i_grupo = groups_index[n_group]
            start_hour = min(hours[i_grupo])
            end_hour = max(hours[i_grupo])
            n_tr_matrix_grouped[n_group][n_order] = \
                sum(n_tr_matrix[i_grupo][n_order])
            p_tr_matrix_grouped[n_group][n_order] = \
                normalize(n_tr_matrix_grouped[n_group][n_order], axis=1,
                          norm='l1')
            ocurring = np.where(n_tr_matrix_grouped[n_group][n_order] != 0)
            glyphs_from_oc = list(OrderedDict.fromkeys(ocurring[0]))
            num_glyphs_from_oc = len(glyphs_from_oc)
            fig, ax = plt.subplots(figsize=(10, 10))
            cax = ax.matshow(n_tr_matrix_grouped[n_group][n_order]
                             [glyphs_from_oc], cmap=plt.cm.Blues)
            fig.colorbar(cax, fraction=0.046, pad=0.04)
            tick_list = np.arange(0, num_glyphs_from_oc, 1)
            ax.set_xticks(tick_list)
            ax.set_xticklabels(glyphs_from_oc)
            ax.xaxis.set_ticks_position('bottom')
            ax.set_yticks(tick_list)
            ax.set_yticklabels(syl_types)
            ax.set_title('From: {}, to: {}\nOrden: {}'.format(start_hour,
                         end_hour, n_order))
            ax.set_ylabel('From')
            ax.set_xlabel('To')
    else:
        print('Hay solo 1 file')
