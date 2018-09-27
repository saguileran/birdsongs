#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 15:41:05 2018

Codigo para analizar experimentos de playback

@author: juan
"""
# %% Importo librerias
import numpy as np
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.signal import butter
from scipy.signal import argrelextrema
from analysis_functions import get_spectrogram, normalizar
import sys
import gc


# %%
class Experiment:
    """
    Todo lo que es folder-file-wise. Nada de manejo ni analisis de datos.
    """
    def __init__(self, birdname, year):
        self.birdname = birdname
        self.year = year
        # To do -> softcodear la carpeta base dandosela como parametro
        self.base_folder = glob.glob(os.path.join('/media/juan/New Volume/Experimentos vS/{}/{}'.format(self.year, self.birdname)))[0]
        self.analysis_folder = '/home/juan/Documentos/Musculo/Experimentos vS/{}/'.format(self.birdname)
        self.day_folders = glob.glob(os.path.join(self.base_folder,
                                                   '{}*day'.format(self.year)))
        self.night_folders = glob.glob(os.path.join(self.base_folder,
                                                   '{}*night'.format(self.year)))
        self.playback_folders = glob.glob(os.path.join
                                          (self.base_folder,
                                           'Playbacks/*{}'.format(self.year)))

    def get_date_by_folder(self, folder):
        """
        Devuelve la fecha asociada a la carpeta en formato mmdd

        Parameters
        ----------
        folder: string
            Carpeta

        Returns
        -------
        Date: string
            Mes-día en formato: 'mmdd'
        """
        name_split = folder.split('-')
        day = name_split[2]
        month = name_split[1]
        return ''.join([month, day])

    def get_folder_by_date(self, date, daytime=True):
        """
        Devuelve la carpeta asociada a la fecha

        Parameters
        ----------
        date: string
            Fecha en formato 'mmdd'

        daytime: boolean
            Si 'True' busca la carpeta de grabaciones nocturnas, si 'False' las
            nocturnas

        Returns
        -------
        folder: string
            Carpeta
        """
        date = str(date)
        day = date[:2]
        month = date[2:]
        daytime_candidate = os.path.join(self.base_folder,
                                         '{}-{}-{}-day'.format(self.year,
                                                               month, day))
        nighttime_candidate = os.path.join(self.base_folder,
                                           '{}-{}-{}-night'.format(self.year,
                                                                   month, day))
        return daytime_candidate if daytime else nighttime_candidate

    def get_log_file_path(self, folder, playbackOnly=False):
        if playbackOnly:
            return os.path.join(folder, 'playback-log.txt')
        else:
            return os.path.join(folder, 'adq-log.txt')

    def load_log(self, folder, playabackOnly=False):
        log = pd.read_csv(self.get_log_file_path(folder=folder,
                                                 playbackOnly=playabackOnly),
                          delimiter='\t')
        return log


class dataFile:
    def __init__(self, data, fs):
        self.data = data
        self.fs = fs

    def butter_lowpass(data, fs, lcutoff=3000.0, order=15):
        nyq = 0.5*fs
        normal_lcutoff = lcutoff/nyq
        bl, al = butter(order, normal_lcutoff, btype='low', analog=False)
        return bl, al

    def butter_lowpass_filter(self, data, fs, lcutoff=3000.0, order=6):
        bl, al = self.butter_lowpass(fs, lcutoff, order=order)
        yl = signal.filtfilt(bl, al, data)
        return yl

    def butter_highpass(data, fs, hcutoff=100.0, order=6):
        nyq = 0.5*fs
        normal_hcutoff = hcutoff/nyq
        bh, ah = butter(order, normal_hcutoff, btype='high', analog=False)
        return bh, ah

    def butter_highpass_filter(self, data, fs, hcutoff=100.0, order=5):
        bh, ah = self.butter_highpass(fs, hcutoff, order=order)
        yh = signal.filtfilt(bh, ah, data)
        return yh

    def resample(self, new_fs=44150):
        return signal.resample(self.data, int(len(self.data)*new_fs/self.fs))

    def envelope(self, method='hilbert', f_corte=100):
        hil_env = np.abs(signal.hilbert(self.data))
        hil_env = self.butter_lowpass_filter(hil_env, self.fs, order=5,
                                             lcutoff=f_corte)
        return hil_env

    def get_file_spectrogram(self, window=1024, overlap=1/1.1,
                             sigma=102.4, plot=False, fmax=8000):
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
        fu, tu, Sxx = signal.spectrogram(self.data, self.fs, nperseg=window,
                                         noverlap=window*overlap,
                                         window=signal.get_window
                                         (('gaussian', sigma), window),
                                         scaling='spectrum')
        Sxx = np.clip(Sxx, a_min=np.amax(Sxx)*0.000001, a_max=np.amax(Sxx))
        if plot:
            plt.figure()
            plt.pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                           rasterized=True)
            plt.ylim(0, fmax)
        return fu, tu, Sxx


class Protocol:
    def __init__(self, birdname, year):
        self.Experiment = Experiment(birdname=birdname, year=year)
        self.birdname = birdname
        self.year = year
        self.base_folder = glob.glob(os.path.join('/media/juan/New Volume/Experimentos vS/{}/{}'.format(self.year, self.birdname)))[0]
        self.night_folders = glob.glob(os.path.join(self.base_folder,
                                                    '{}*night'.format
                                                    (self.year)))
        self.playback_folders = glob.glob(os.path.join
                                          (self.base_folder,
                                           'Playbacks/*{}'.format(self.year)))

    def get_playback_folder(self, date=-1):
        """
        Devuelve la carpeta de playbacks asociada a la fecha

        Parameters
        ----------
        date: string
            Fecha en formato 'ddmm'

        Returns
        -------
        folder: string
            Carpeta
        """
        if date == -1:
            raise Exception('Date should be in format ddmm (int or string)')
        date = str(date)
        which_one = [x.endswith(''.join([date, '2018'])) for x in
                     self.playback_folders]
        how_many = sum(which_one)
        if how_many != 1:
            raise Exception('There are {} playback folders for date {}'.
                            format(how_many, date))
        return self.playback_folders[np.where(which_one)[0][0]]

    def get_playback_instances(self, date=-1):
        """
        Devuelve todas las instancias de playback en la fecha
        Parameters
        ----------
        date: string
            Fecha en formato 'ddmm'

        with_delays: boolean
            Si True, agrega una columna al final que da el delay entre el
            playback y el estimulo (para alinear)

        Returns
        -------
        playback_log: pandas dataFrame
            Log de playbacks con información relevante
        """
        if date == -1:
            raise Exception('Date should be in format mmdd (int or string)')
        date = str(date)
        try:
            playback_log = self.Experiment.load_log(self.Experiment.get_folder_by_date(date, daytime=False),
                                                    playabackOnly=True)
            delays = playback_log['delay']
        except FileNotFoundError as e:
            print(e)
            print('Creating a playback log...')
            playback_log, delays = self.create_playback_log(date=date,
                                                            with_delays=True,
                                                            save=True)
            print('Playback log created at ({})'.format
                  (self.Experiment.get_folder_by_date(date, daytime=False)))
        return playback_log, delays

    def create_playback_log(self, date=-1, with_delays=True, save=True):
        """
        Devuelve todas las instancias de playback en la fecha
        Parameters
        ----------
        date: string
            Fecha en formato 'ddmm'

        with_delays: boolean
            Si True, agrega una columna al final que da el delay entre el
            playback y el estimulo (para alinear)

        Returns
        -------
        playback_log: pandas dataFrame
            Log de playbacks con información relevante
        """
        if date == -1:
            raise Exception('Date should be in format mmdd (int or string)')
        date = str(date)
        date_log = self.Experiment.load_log(self.Experiment.get_folder_by_date(date, daytime=False))
        playback_log = date_log[date_log['trigger'] == 'playback']
        delays = np.zeros(len(playback_log))
        if with_delays:
            print('Calculating delays...')
            delays = np.asarray([self.single_playback_align(playback_log,
                                                            date=date,
                                                            index=ii)[0]
                                 for ii in playback_log.index.values])
        playback_log.loc[:, 'delay'] = delays
        if save:
            playback_log.to_csv('{}/playback-log.txt'.format
                                (self.Experiment.get_folder_by_date(date, daytime=False)),
                                sep='\t')
        return playback_log, delays

    def plotProtocol(self, date=-1, test=False):
        if date == -1:
            raise Exception('Date should be in format mmdd (int or string)')
        date = str(date)
        pb_folder = self.get_playback_folder(date)
        vs_folder = self.Experiment.get_folder_by_date(date, daytime=False)
        playback_log, delays = self.get_playback_instances(date=date)
        playback_type = [x.split('_', 1)[0] for x in
                         playback_log['playback_fname']]
        if playback_type[0] == self.birdname:
            playback_type = [x.rsplit('_', 1)[1].split('.', 1)[0] for x in
                             playback_log['playback_fname']]
        all_types = np.sort(list(set(playback_type)))
        if test:
            all_types = all_types[:2]
            playback_type = playback_type[:100]
        print(all_types)
        for pb_type in all_types:
            pb_index = [i for i, val in enumerate(playback_type) if val==pb_type]
            reduced_log = playback_log.loc[pb_index]
            pb_name = reduced_log['playback_fname'].iloc[0]
            print(pb_folder)
            print(pb_name)
            pb_fname = os.path.join(pb_folder, pb_name)
            fs_b, pb = wavfile.read(pb_fname)
            pbFile = dataFile(pb, fs_b)
            fu, tu, Sxx = pbFile.get_file_spectrogram()
            fig, ax = plt.subplots(2, figsize=(14, 4), sharex=True)
            ax[0].pcolormesh(tu, fu, np.log(Sxx),
                             cmap=plt.get_cmap('Greys'),
                             rasterized=True)
            ax[0].set_ylim(0, 8000)
            ax[0].set_ylabel(pb_type)
            for index, row in reduced_log.iterrows():
                vs_name = row['vS_fname']
                vs_fname = os.path.join(vs_folder, vs_name)
                fs, vs_raw = wavfile.read(vs_fname)
                vs_min = row['vS_min']
                vs_max = row['vS_max']
                vs_delay = row['delay']
                vs = normalizar(vs_raw, minout=vs_min, maxout=vs_max,
                                zeromean=False)
                vs_time = np.arange(len(vs))/fs-vs_delay
                vsFile = dataFile(vs, fs)
                vs_envelope = vsFile.envelope()
                ax[1].plot(vs_time, vs_envelope, 'b', alpha=0.1)
                ax[1].set_ylim(0, 0.1)
                ax[1].set_xlim(-2, max(tu)+2)
                del(vsFile)
            fig.tight_layout()
            fig.savefig('{}_{}.pdf'.format(date, pb_type), fmt='pdf')
            del(pbFile)
            gc.collect()
            fig.clf()
        return 1

    def single_playback_align(self, log, date, index, show_me=False):
        """
        Alinea el vS evocado con el estímulo buscando el tiempo de maxima
        correlacion entre el sonido guardado y el del estimulo para un playback

        Parameters
        ----------
        log: pandas dataFrame
            Registro de grabaciones. Se obtiene con la funcion load_log

        index: int
            Indice del playback a alinear

        show_me: boolean
            Si True grafica las señales alineadas
        Returns
        -------

        """
        folder = self.Experiment.get_folder_by_date(date=date, daytime=False)
        sound_file = os.path.join(folder, log.loc[index]['s_fname'])
        vs_file = os.path.join(folder, log.loc[index]['vS_fname'])
        fs, sound = wavfile.read(sound_file)
        sound = normalizar(sound, minout=-1, maxout=1)
        fs, vs = wavfile.read(vs_file)
        pb_folder = self.get_playback_folder(date)
        pb_file = os.path.join(pb_folder, log.loc[index]['playback_fname'])
        fs_b, pb = wavfile.read(pb_file)
        if fs_b != fs:
            pb = signal.resample(pb, int(len(pb)*fs/fs_b))
        pb = normalizar(pb, minout=-1, maxout=1)
        corr = signal.correlate(sound, pb, mode='valid')
        delay = np.argmax(corr)/fs
        if show_me:
            fig, ax = plt.subplots(4, figsize=(12, 6), sharex=True)
            time_sound = np.arange(len(sound))/fs
            ax[0].plot(time_sound, sound)
            fu, tu, Sxx = get_spectrogram(sound, fs)
            ax[1].pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                             rasterized=True)
            ax[1].set_ylim(0, 8000)
            pb_delay = np.concatenate(([0]*int(delay*fs), pb))
            time_pb = np.arange(len(pb_delay))/fs
            ax[2].plot(time_pb, pb_delay)
            fupb, tupb, Sxxpb = get_spectrogram(pb_delay, fs)
            ax[3].pcolormesh(tupb, fupb, np.log(Sxxpb),
                             cmap=plt.get_cmap('Greys'), rasterized=True)
            ax[3].set_ylim(0, 8000)
            fig.tight_layout()
        return delay, sound, vs, pb, fs


# %% Defino directorios de trabajo (files y resultados)
birdname = 'CeRo'
year = 2018
exp = Experiment(birdname=birdname, year=year)
pb = Protocol(birdname=birdname, year=year)
pb.plotProtocol('2808')
