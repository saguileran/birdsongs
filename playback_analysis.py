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
    def get_folder_date(self, folder):
        name_split = folder.split('-')
        day = name_split[2]
        month = name_split[1]
        return ''.join([month, day])
    def get_log_files(day_num):
        return [glob.glob(os.path.join(self.))


# %% Defino directorios de trabajo (files y resultados)
birdname = 'CeRo'
year = 2018
exp = Experiment(birdname=birdname, year=year)

# %% Miro una noche
for folder in day_folders:
    log_files.append()
