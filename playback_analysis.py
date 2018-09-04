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
        day = date[2:]
        month = date[:2]
        daytime_candidate = os.path.join(self.base_folder,
                                         '{}-{}-{}-day'.format(self.year,
                                                               month, day))
        nighttime_candidate = os.path.join(self.base_folder,
                                           '{}-{}-{}-night'.format(self.year,
                                                                   month, day))
        return daytime_candidate if daytime else nighttime_candidate

    def get_log_file_path(self, folder):
        return os.path.join(folder, 'adq-log.txt')

    def load_log(self, folder):
        return pd.read_csv(self.get_log_file_path(folder), delimiter='\t')

#    def get_playback_instances(self, folder):
        

# %% Defino directorios de trabajo (files y resultados)
birdname = 'CeRo'
year = 2018
exp = Experiment(birdname=birdname, year=year)

# %% Miro una noche
for folder in day_folders:
    log_files.append()
