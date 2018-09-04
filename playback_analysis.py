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
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.signal import butter
from scipy.signal import argrelextrema
import analysis_functions


# %%
class Experiment:
    def __init__(self, birdname, year):
        self.birdname = birdname
        self.year = year
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
    
    def get_playback_folder(self, date=-1):
        """
        Devuelve la carpeta de playbacks asociada a la fecha

        Parameters
        ----------
        date: string
            Fecha en formato 'mmdd'

        Returns
        -------
        folder: string
            Carpeta
        """
        if date == -1:
            raise Exception('Date should be in format mmdd (int or string)')
        date = str(date)
        which_one = [x.endswith(''.join([date, '2018'])) for x in self.playback_folders]
        how_many = sum(which_one)
        if how_many != 1:
            raise Exception('There are {} playback folders for date {}'.format(how_many, date))
        return self.playback_folders[np.where(which_one)[0][0]]

    def get_log_file_path(self, folder):
        return os.path.join(folder, 'adq-log.txt')

    def load_log(self, folder):
        return pd.read_csv(self.get_log_file_path(folder), delimiter='\t')

    def get_playback_instances(self, date=-1):
        if date == -1:
            raise Exception('Date should be in format mmdd (int or string)')
        date = str(date)
        date_log = self.load_log(self.get_folder_by_date(date, daytime=False))
        return date_log[date_log['trigger'] == 'playback']

    def playback_align(self, log, index, delay=1.):
        """
        Alinea el vS evocado con el estímulo buscando el tiempo de maxima
        correlacion entre el sonido guardado y el del estimulo. Asumo que todos
        los playbacks son nocturnos.

        Parameters
        ----------
        log: pandas dataFrame
            Registro de grabaciones. Se obtiene con la funcion load_log
        index: int
            Numero de registro a analizar
        delay: float
            Tiempo previo a mostrar.

        Returns
        -------
        
        """
        date = log.iloc[index]['date'].split('_')
        month = date[1]
        day = date[2]
        date = ''.join([day, month])
        folder = self.get_folder_by_date(date=date, daytime=False)
        sound_file = os.path.join(folder, log.iloc[index]['s_fname'])
        vs_file = os.path.join(folder, log.iloc[index]['vS_fname'])
        fs, sound = wavfile.read(sound_file)
        fs, vs = wavfile.read(vs_file)
        pb_folder = self.get_playback_folder(date)
        pb_file = os.path.join(pb_folder, log.iloc[index]['playback_fname'])
        fs, pb = wavfile.read(pb_file)
#        fu, tu, Sxx = analysis_functions.get_spectrogram(sound, fs)
#        plt.pcolormesh(fu, tu, Sxx)
        return sound, vs, pb
# %% Defino directorios de trabajo (files y resultados)
birdname = 'CeRo'
year = 2018
exp = Experiment(birdname=birdname, year=year)

# %% Miro una noche
for folder in day_folders:
    log_files.append()
