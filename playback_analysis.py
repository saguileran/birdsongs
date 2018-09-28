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
from scipy.io import wavfile
from scipy import signal
from scipy.signal import butter
from analysis_functions import get_spectrogram, normalizar
import sys
import gc
import matplotlib
import random
matplotlib.use('agg')
import matplotlib.pyplot as plt


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
                                                    '{}*night'.format
                                                    (self.year)))
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

    def get_random_file(self, daytime=False, playback=False, show_me=False):
        done = False
        if daytime:
            folders = self.day_folders
        else:
            folders = self.night_folders
        random.shuffle(folders)
        it_folder = iter(folders)
        while not done:
            folder = next(it_folder)
            try:
                log = self.load_log(folder)
                if playback:
                    log2 = log[log['trigger'] == 'playback']
                else:
                    log2 = log[log['trigger'] != 'playback']
                rand_select = log2.iloc[random.randint(0, len(log2))]
                vs_file = rand_select['vS_fname']
                vs_max = rand_select['vS_max']
                vs_min = rand_select['vS_min']
                fname = os.path.join(folder, vs_file)
                fs, data = wavfile.read(fname)
                data = normalizar(data, minout=vs_min, maxout=vs_max,
                                  zeromean=False)
                done = True
            except:
                pass
        if done:
            print(vs_file)
            print(folder)
            if show_me:
                plt.figure()
                plt.plot(np.arange(len(data))/fs, data)
            return data, fs
        else:
            print('Try again')
            return 1


class dataFile:
    def __init__(self, data, fs):
        self.data = data
        self.fs = fs
        self.npoints = len(self.data)
        self.time = np.arange(self.npoints)/self.fs
        self.envelope = np.asarray([0])
        self.subsampled = np.asarray([0])
        self.subsampling = 0
        self.subtime = np.asarray([0])
        self.subenv = np.asarray([0])

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
        resampled = signal.resample(self.data, int(len(self.data)*new_fs/self.fs))
        return resampled

    def calculate_envelope(self, method='hilbert', f_corte=100, logenv=False):
        envelope = np.abs(signal.hilbert(self.data))
        envelope = self.butter_lowpass_filter(envelope, self.fs, order=5,
                                              lcutoff=f_corte)
        if logenv:
            envelope = np.log(envelope)
        if method != 'hilbert':
            print('Hilbert is the only available method (and what you got)')
        return envelope

    def autocorr(self, subsampling=100, mode='full', plot=False):
        if self.subsampling != subsampling:
            self.subsample(subsampling=subsampling)
        result = np.correlate(self.subsampled, self.subsampled, mode=mode)
        autocorr = result[result.size//2:]
        if plot:
            plt.figure()
            plt.plot(self.subtime, autocorr)
        return autocorr

    def subsample(self, subsampling=10):
        subsamp_data = self.data[::subsampling]
        self.subsampled = subsamp_data
        subsamp_envelope = self.envelope[::subsampling]
        self.subenv = subsamp_envelope
        sb_time = self.time[::subsampling]
        self.subtime = sb_time
        self.subsampling = subsampling
        return sb_time, subsamp_data, subsamp_envelope

    def plot(self, plotEnvelope=False, subsampling=1):
        plt.figure()
        plt.plot(self.time, self.data, alpha=0.3)
        if plotEnvelope:
            if self.envelope.shape[0] == 1:
                self.envelope = self.calculate_envelope()
            if self.subsampling != subsampling:
                self.subsample(subsampling=subsampling)
            plt.plot(self.time, self.envelope)
            plt.plot(self.subtime, self.subenv)
        return 0

    def envelope_spectrogram(self, tstep=0.01, sigma_factor=5,
                             plot=False, fmin=0, fmax=50, freq_resolution=5):
        time_win = 1/freq_resolution
        window = int(self.fs*time_win)
        overlap = 1-(tstep/time_win)
        sigma = window/sigma_factor
        if len(self.envelope) == 1:
            self.envelope = self.calculate_envelope()
        fu, tu, Sxx = signal.spectrogram(self.envelope, self.fs,
                                         nperseg=window,
                                         noverlap=window*overlap,
                                         window=signal.get_window
                                         (('gaussian', sigma), window),
                                         scaling='spectrum')
        Sxx = np.clip(Sxx, a_min=np.amax(Sxx)*0.0001, a_max=np.amax(Sxx))
        if plot:
            fig, ax = plt.subplots(2, figsize=(16,4), sharex=True)
            ax[0].plot(self.time, self.data)
            ax[0].plot(self.time, self.envelope)
            ax[1].pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                           rasterized=True)
            ax[1].set_ylim(fmin, fmax)
            fig.tight_layout()
        return fu, tu, Sxx

    def get_file_spectrogram(self, window=1024, overlap=1/1.1, sigma=102.4,
                             plot=False, fmin=0, fmax=8000):
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
            plt.ylim(fmin, fmax)
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
            playback_log = self.Experiment.load_log(self.Experiment.get_folder_by_date
                                                    (date, daytime=False),
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
        date_log = self.Experiment.load_log(self.Experiment.get_folder_by_date
                                            (date, daytime=False))
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
                                (self.Experiment.get_folder_by_date
                                 (date, daytime=False)),
                                sep='\t')
        return playback_log, delays

    def plotProtocol(self, dates=-1, test=False, logenv=False):
        if dates == -1:
            raise Exception('"dates" should be an array with elements in \
                            format mmdd (int or string)')
        for date in dates:
            date = str(date)
            print('Date: {}'.format(date))
            pb_folder = self.get_playback_folder(date)
            vs_folder = self.Experiment.get_folder_by_date(date, daytime=False)
            playback_log, delays = self.get_playback_instances(date=date)
            playback_type = [x for x in playback_log['playback_fname']]
            all_types = np.sort(list(set(playback_type)))
            if test:
                all_types = all_types[:2]
                playback_type = playback_type[:100]
            print('Playbacks found at {}:\n{}'.format(pb_folder, all_types))
            for pb_type in all_types:
                pb_index = [i for i, val in enumerate(playback_type) if
                            val == pb_type]
                reduced_log = playback_log.loc[pb_index]
                pb_fname = os.path.join(pb_folder, pb_type)
                fs_b, pb = wavfile.read(pb_fname)
                pbFile = dataFile(pb, fs_b)
                fu, tu, Sxx = pbFile.get_file_spectrogram()
                fig, ax = plt.subplots(2, figsize=(14, 4), sharex=True)
                ax[0].pcolormesh(tu, fu, np.log(Sxx),
                                 cmap=plt.get_cmap('Greys'),
                                 rasterized=True)
                ax[0].set_ylim(0, 8000)
                ax[0].set_title('{}'.format(pb_type))
                alpha = 1./len(reduced_log)
                for index in range(len(reduced_log)):
                    row = reduced_log.iloc[index]
                    vs_name = str(row['vS_fname'])
                    vs_fname = os.path.join(vs_folder, vs_name)
                    fs, vs_raw = wavfile.read(vs_fname)
                    vs_min = row['vS_min']
                    vs_max = row['vS_max']
                    vs_delay = row['delay']
                    vs = normalizar(vs_raw, minout=vs_min, maxout=vs_max,
                                    zeromean=False)
                    vs_time = np.arange(len(vs))/fs-vs_delay
                    vsFile = dataFile(vs, fs)
                    vs_envelope = vsFile.calculate_envelope(logenv=logenv)
                    ax[1].plot(vs_time, vs_envelope, 'k', alpha=alpha)
                    ax[1].set_xlim(-2, max(tu)+2)
                    del(vsFile)
                fig.tight_layout()
                figname = '{}_{}'.format(date, pb_type)
                if logenv:
                    figname += '_logenv'
                if test:
                    figname += '_test'
                fig.savefig('{}.pdf'.format(figname), fmt='pdf')
                del(pbFile)
                gc.collect()
                fig.clf()
            print('Done with {}'.format(date))
        return 1

    def single_playback_align(self, log, date, index, show_me=False):
        """
        Alinea el vS evocado con el estímulo buscando el tiempo de maxima
        correlacion entre el sonido guardado y el del estimulo para un playback
        Si el playback tiene distinto sampling rate, lo resamplea.

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
        delay: float
            Delay entre las señales, en segundos.

        sound: array
            Señal de sonido

        vs: array
            Señal de vs

        pb: array
            Señal (resampleada) de sonido del playback.

        fs: float
            Frecuencia de sampleo del vs
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


def progressbar(array, length=0):
    """
    Simple progress bar utility
    """
    i = 1
    N_pts = 0
    length = len(array)
    iter_ = iter(array)
    N_pb = int(np.ceil(length/100))
    msg = "\n| 0% - Calculating "+str(length)+" values"
    print(msg+" "*(94-len(msg))+"100% |", end="\n", flush=False)
    yield next(iter_)
    while True:
        i += 1
        if i % N_pb == 0:
            N_new = int(np.floor(100*i/length)-N_pts)
            print('\b', "."*N_new, sep="", end=" ", flush=N_pts > 5)
            N_pts += N_new
        if i == length-1:
            print(end="\n", flush=True)
        yield next(iter_)


# %% Defino directorios de trabajo (files y resultados)
if __name__ == "__main__":
    birdname = 'CeRo'
    year = 2018
    exp = Experiment(birdname=birdname, year=year)
    pb = Protocol(birdname=birdname, year=year)
    data, fs = exp.get_random_file(playback=True)
    rnd_file = dataFile(data, fs)
    subsampling = 200
    rnd_file.plot(plotEnvelope=True, subsampling=subsampling)
#    rnd_file.autocorr(subsampling=subsampling, plot=True)
#    rnd_file.envelope_spectrogram(plot=True, fmax=200,window=10240*2, sigma=1024*3)
#    pb.plotProtocol([2608, 2708, '2808'], test=True, logenv=True)
