# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 20:07:04 2016

@author: juan
"""

#%%
import os
import glob
from scipy.io import wavfile
from IPython import display
from shutil import copyfile
import pandas as pd
import errno
from scipy import signal

def make_path(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

#%%
birdname = 'AmaVio'
os.chdir('/media/juan/New Volume/Experimentos vS/2018/canto/{}/wavs'.format(birdname))
folders = glob.glob(os.path.join(os.getcwd(),'*'))
destino = '/home/juan/Documentos/Musculo/Experimentos vS/{}'.format(birdname)
make_path(destino)

#%%
fig, ax = plt.subplots(2, figsize=(10,4), sharex=True)
for w_dir in folders:
    day = w_dir.rsplit('/',1)[1]
    dst = destino + '/{}'.format(day) 
    make_path(dst)
    files = glob.glob(os.path.join(w_dir, '*.wav'))
    for wav in files:
        fs, data = wavfile.read(wav)
        time = np.linspace(0, np.double(len(data))/fs, len(data))
        ax[0].plot(time, data)
        NN = 1024
        overlap = 1/1.1
        sigma = NN/10
        fu, tu, Sxx = signal.spectrogram(data, fs, nperseg=NN, noverlap=NN*overlap,
                                 window=signal.get_window(('gaussian', sigma), NN), scaling='spectrum')
        Sxx = np.clip(Sxx, a_min=np.amax(Sxx)*0.000001, a_max=np.amax(Sxx))
        ax[1].pcolormesh(tu, fu, np.log(Sxx), cmap=plt.get_cmap('Greys'),
                 rasterized=True)
        ax[1].set_ylim(0, 8000)

        display.clear_output(True)
        display.display(gcf())
        plt.pause(0.01)
        b = input('Copiar? [y/n]\n')
        if b == 'y':
            copyfile(wav, dst)
        elif b == 'q':
            sys.exit(0)
        else:
            print('Next...')
            #os.rename(musculo, destino+'/'+os.path.basename(musculo))
        del(data)
        ax[0].cla()
        ax[1].cla()
