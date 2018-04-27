# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 20:07:04 2016

To do:
    - Eleccion de archivo/s para entrenamiento -> done (carpeta separada)
    - Multiples archivos para entrenar -> done
    - Mejorar espectrograma not?
    - Eliminar intervalos espureos -> done ? Aca o post-proc ?
    - Generar output con secuencias de sílabas -> done
    - Guardar tiras de sílabas -> done
    - Separar learning y prediction en py's distintos
    - Preparar para mas de 10 tipos de silabas -> done
    - Agregar signal method
@author: gonza
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from sklearn.cross_validation import cross_val_score
import errno
from sklearn.svm import LinearSVC
import random
import pandas as pd
import glob


def make_path(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def consecutive(data, stepsize=1):
    """
    Separa en segmentos de datos consecutivos. Se usa para separar indices
    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]


# %%
files_folder = '/home/juan/Documentos/Musculo/Codigo canarios/Files'
training_folder = '/home/juan/Documentos/Musculo/Codigo canarios/Files'
templates_folder = '{}/Templates'.format(training_folder)
make_path(templates_folder)
# Read input audio file and define times
train_files = glob.glob(os.path.join(training_folder, '*.wav'))
# Define the amount of points used for the fourier transform in each frame of
# the spectogram
NFFT = 512  # Una potencia de 2 es mas rapido (puedo usar FFT)
# Porcentaje de overlap entre ventanas consecutivas
p_overlap = 0.67
train_song = [0]*len(train_files)
n = 0
for f_name in train_files:
    samp_rate, train_song[n] = wavfile.read('{}'.format(f_name))
    n += 1
signa = np.concatenate((train_song[-1], train_song[-2]))
signa = signa[:44150*10]
times = np.arange(len(signa))/float(samp_rate)
spectrogram, freqs, bins, im = plt.specgram(signa, NFFT=NFFT, Fs=samp_rate,
                                            noverlap=int(NFFT*p_overlap),
                                            cmap='Greys')
min_freq = 500
max_freq = 8000
plt.ylim(0, 8000)
# %%
# ------------Generating the training set---------------
method = 'signal'
# Size of the feature image
dfreq = samp_rate/NFFT
n_skip = 10
t_bins = times[::n_skip]
time_window = 0.4   # Time window in seconds
window_width = int(time_window/np.diff(bins)[0])
if window_width % 2 == 0:
    window_width += 1
half_window = int(window_width/2)
# Corto al rango de frecuencias relevante
signa_relevant = signa[::n_skip]

# %%
# Buffer matrix for computing the frames
training_frame = np.zeros(window_width)

# List of classification tags
frame_class = []

# Create dictionary of syllable classes (for frame_class)
# Va a haber problemas si hay mas de 10 silabas
syl_names = ['S', 'A']
syl_classes = {key: syl_names.index(key) for key in syl_names}
# Create dictionary of syllable colors (for plots)
syl_colors = {'S': 'r', 'A': 'g'}
# Create dictionary of syllable training sets
syl_training_sets = {key: [] for key in syl_classes}
# Create dictionary of syllable classes (for frame_class)
syl_training_begs = {'S': [0.5],
                     'A': [7.45]}
syl_training_ends = {'S': [4.],
                     'A': [8.55]}

fig, ax = plt.subplots(2, len(syl_classes), figsize=(30, 6))
# Defining the positive training set
all_training_set = []
for syl_type in syl_classes:
    begs = syl_training_begs[syl_type]
    ends = syl_training_ends[syl_type]
    # Save syllable templates
    rand_index = random.randrange(0, len(begs))
    training_set = []
    syl_id = syl_classes[syl_type]
    for i in range(len(begs)):
        if i == rand_index:
            cut_wav = signa[int(begs[i]*samp_rate):int(ends[i]*samp_rate)]
            wavfile.write('{}/silaba_{}.wav'.format(templates_folder,
                                                    syl_type),
                          samp_rate, cut_wav)
            ncol = syl_classes[syl_type]
            ax[0][ncol].plot(cut_wav)
            ax[0][ncol].set_xticklabels([])
            ax[0][ncol].set_yticklabels([])
            ax[0][ncol].set_title(syl_type)
            ax[1][ncol].specgram(cut_wav, NFFT=NFFT, Fs=samp_rate,
                                 noverlap=int(NFFT*p_overlap), cmap='Greys')
            ax[1][ncol].set_yticklabels([])
            ax[1][ncol].xaxis.set_major_locator(plt.MaxNLocator(2))
            ax[1][ncol].set_ylim(min_freq, max_freq)
        # conversion from times to 'bins' vector coordinates
        # 'bins' = center times of spec
        time_beg_coord = (np.abs(t_bins-begs[i])).argmin()
        time_end_coord = (np.abs(t_bins-ends[i])).argmin()
        for j in range(time_beg_coord, time_end_coord):
            # Taking the frame from the spectrogram
            training_frame = signa_relevant[j-half_window:j+1+half_window]
            training_set.append(training_frame)
            frame_class.append(syl_id)
    all_training_set += training_set
fig.tight_layout()
# %%
# Shuffle the training list (we have to shuffle things together)

combined = list(zip(all_training_set, frame_class))
random.shuffle(combined)
all_training_set[:], frame_class[:] = zip(*combined)

# Prearing arrays for placing the data
X = np.zeros((len(all_training_set), len(training_frame)))
Y = np.asarray(frame_class)

# Arranging the data format into the X,Y input needed for sklearn
for i in range(len(all_training_set)):
    X[i, :] = all_training_set[i]
# %%
# ---------Training ------------------
# Prediction (only within the training set)
cross = 3
print("LinearSVC + RAW data:")
print(np.mean(cross_val_score(LinearSVC(), X, Y, cv=cross)))

# Classificator training
clf = LinearSVC()
clf.fit(X, Y)
print('Fit complete')
# %%
# ---------Prediction -----------------

prediction = []
confidence = []
predict_time = []

# Preparing buffer for computing the prediciton
# Aca se convierte el espectro en un array y se lo evalua usando el fit (clf)
predict_vector = np.zeros(window_width)

# Goes over all the data (leaving the windows at the beginning and the end)
for j in range(half_window, len(bins)-half_window):
    # Takes the frame
    predict_vector = signa_relevant[j-half_window:j+1+half_window]
    # Append the time and the class prediction
    predict_time.append(bins[j])
    prediction.append(clf.predict([predict_vector])[0])
    confidence.append(clf.decision_function([predict_vector])[0])
# %%
fig, ax = plt.subplots(3, figsize=(12, 12), sharex=True)
spectrogram, freqs, bins, im = ax[0].specgram(signa, NFFT=NFFT, Fs=samp_rate,
                                              noverlap=int(NFFT*p_overlap),
                                              cmap='Greys')

for syl in syl_classes:
    inicios = syl_training_begs[syl]
    fines = syl_training_ends[syl]
    for nn in range(len(inicios)):
        ax[0].axvspan(inicios[nn], fines[nn], color=syl_colors[syl], alpha=0.5,
                      label='{} Training set'.format(syl) if nn == 0 else '')
ax[0].legend()
ax[0].set_ylim(0, 8000)
syl_prediction = {key: [] for key in syl_classes}
for syl in syl_classes:
    syl_prediction[syl] = np.asarray(prediction.copy()).astype(np.double)
    syl_prediction[syl][syl_prediction[syl] != syl_classes[syl]] = np.nan
spectrogram, freqs, bins, im = ax[1].specgram(signa, NFFT=NFFT, Fs=samp_rate,
                                              noverlap=int(NFFT/2),
                                              cmap='Greys')
ax[1].set_ylim(0, 8000)
axt = ax[1].twinx()
for syl in syl_classes:
    axt.fill_between(predict_time,
                     0*syl_prediction[syl]-0.1, 0*syl_prediction[syl]+0.1,
                     color=syl_colors[syl], label=syl)
axt.legend()
axt.set_ylim(-4.1, 0.1)

ax[2].plot(times, signa)
axt = ax[2].twinx()
for syl in syl_classes:
    syl_class = syl_classes[syl]
    axt.plot(predict_time, [x[syl_class] for x in confidence], label=syl,
             c=syl_colors[syl])
axt.legend()
axt.set_ylabel('Confidence')
axt.set_xlabel('Time [sec]')

# %% Prediction new files
df_file = '{}/song-log'.format(files_folder)
df_song = pd.DataFrame(columns=['File name', 'Song', 'Start', 'End'])
df_song = pd.DataFrame(columns=['File name', 'Song'])
song_files = glob.glob(os.path.join(files_folder, '*.wav'))
fig, ax = plt.subplots(len(song_files), figsize=(30, 6), sharex=False)
n = 0
n_fila = 0
for file in song_files:
    fs, song = wavfile.read(file)
    time = np.arange(len(song))/float(fs)
    s_spec, s_freq, s_bins, s_i = ax[n].specgram(song, NFFT=NFFT, Fs=fs,
                                                 noverlap=int(NFFT*p_overlap),
                                                 cmap='Greys')
    s_relevant = s_spec[window_bot:window_top]
    ax[n].set_ylim(0, 8000)
    axt = ax[n].twinx()
    # ---------Prediction -----------------
    prediction = []
    confidence = []
    predict_time = []
    # Preparing buffer for computing the prediciton
    predict_vector = np.zeros((window_height * window_width))
    # Goes over all the data (leaving the windows at the beginning and the end)
    for j in range(half_window, len(s_bins)-half_window):
        # Takes the frame
        pred_frame = np.asarray([10*np.log10
                                 (np.asarray
                                  (line[(j-half_window):(j+1+half_window)]))
                                 for line in s_relevant])
        # Arrange the frame to form a vector
        for i in range(len(pred_frame)):
            l_frame = len(pred_frame[i])
            predict_vector[(i*l_frame):((i+1)*l_frame)] = pred_frame[i]
        # Append the time and the class prediction
        predict_time.append(s_bins[j])
        prediction.append(clf.predict([predict_vector])[0])
        confidence.append(clf.decision_function([predict_vector])[0])
    transitions = np.where(np.diff(prediction) != 0)
    transition_times = s_bins[transitions[0] + half_window]
    durations = np.diff(transition_times)
    relevant = np.where(durations > 0.1)
    aux_pred = np.concatenate(([prediction[0]],
                               np.asarray(prediction)
                               [transitions[0][relevant]+1],
                               [prediction[-1]]))
    aux_times = np.concatenate(([np.min(s_bins)],
                               transition_times[relevant],
                               [np.max(s_bins)]))
    rel_pred = aux_pred[np.where(np.diff(aux_pred) != 0)]
    if aux_pred[-1] != aux_pred[-2]:
        rel_pred = np.append(rel_pred, aux_pred[-1])
    pred_letters = [list
                    (syl_classes.keys())[list(syl_classes.values()).index(x)]
                    for x in rel_pred]
    file_string = ''.join([str(x) for x in pred_letters])
    song_strings = file_string.split('S')
    for nn in song_strings:
        if nn != '':
            df_song.loc[n_fila] = [os.path.basename(file), 'S{}S'.format(nn)]
            n_fila += 1
    syl_prediction = {key: [] for key in syl_classes}
    for syl in syl_classes:
        syl_prediction[syl] = np.asarray(prediction.copy()).astype(np.double)
        syl_prediction[syl][syl_prediction[syl] != syl_classes[syl]] = np.nan
        axt.fill_between(predict_time,
                         0*syl_prediction[syl]-0.1, 0*syl_prediction[syl]+0.1,
                         color=syl_colors[syl], label=syl_classes[syl])
    for tr in aux_times:
        axt.axvline(x=tr)
    axt.legend()
    axt.set_ylim(-4.1, 0.1)
    n += 1
df_song.to_csv(df_file, index=False)
