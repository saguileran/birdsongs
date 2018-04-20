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


# %%
df_song = pd.DataFrame(columns=['File name', 'Song'])
files_folder = '/home/juan/Documentos/P0'
templates_folder = '{}/Templates'.format(files_folder)
make_path(templates_folder)
# Read input audio file and define times
f_name = 'estimulado_VioAma_171106_190725.wav'
samp_rate, signal = wavfile.read('{}/{}'.format(files_folder, f_name))
times = np.arange(len(signal))/float(samp_rate)

# Define the amount of points used for the fourier transform in each frame of
# the spectogram
NFFT = 512  # Una potencia de 2 es mas rapido (puedo usar FFT)
# Porcentaje de overlap entre ventanas consecutivas
p_overlap = 0.67
spectrogram, freqs, bins, im = plt.specgram(signal, NFFT=NFFT, Fs=samp_rate,
                                            noverlap=int(NFFT*p_overlap),
                                            cmap='Greys')
# %%
# ------------Generating the training set---------------

# Size of the feature image
dfreq = samp_rate/NFFT
min_freq = 500
max_freq = 8000
time_window = 0.4   # Time window in seconds
window_width = int(time_window/np.diff(bins)[0])
if window_width % 2 == 0:
    window_width += 1
half_window = int(window_width/2)
window_bot = int(min_freq/dfreq)
window_top = int(max_freq/dfreq)
window_height = window_top - window_bot
# Corto al rango de frecuencias relevante
spec_relevant = spectrogram[window_bot:window_top]

# %%
# Buffer matrix for computing the frames
training_frame = np.zeros((window_height, window_width))

# List of classification tags
frame_class = []

# Create dictionary of syllable classes (for frame_class)
# Syllable name: syllable id
syl_classes = {'P0': 0, 'notP0': 1}
# Create dictionary of syllable colors (for plots)
# Syllable name: syllable color
syl_colors = {'P0': 'r', 'notP0': 'b'}
# Create dictionary of syllable training sets
syl_training_sets = {key: [] for key in syl_classes}
# Create dictionary of syllable classes (for frame_class)
syl_training_begs = {'P0': [24.76], 'notP0': [14.]}
syl_training_ends = {'P0': [26.17], 'notP0': [24.7]}

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
            cut_wav = signal[int(begs[i]*samp_rate):int(ends[i]*samp_rate)]
            wavfile.write('{}/silaba_{}.wav'.format(templates_folder,
                                                    syl_type),
                          samp_rate, cut_wav)
            ncol = syl_classes[syl_type]
            eje0 = ax[0][ncol]
            eje1 = ax[1][ncol]
            eje0.plot(cut_wav)
            eje0.set_xticklabels([])
            eje0.set_yticklabels([])
            eje1.specgram(cut_wav, NFFT=NFFT, Fs=samp_rate,
                                 noverlap=int(NFFT*p_overlap), cmap='Greys')
            eje1.set_yticklabels([])
            eje1.xaxis.set_major_locator(plt.MaxNLocator(2))
            eje1.set_ylim(min_freq, max_freq)
        # conversion from times to 'bins' vector coordinates
        # 'bins' = center times of spec
        time_beg_coord = (np.abs(bins-begs[i])).argmin()
        time_end_coord = (np.abs(bins-ends[i])).argmin()
        for j in range(time_beg_coord, time_end_coord):
            # Taking the frame from the spectrogram
            training_frame = np.asarray([10*np.log10
                                         (np.asarray
                                          (line[(j-half_window):
                                                (j+1+half_window)]))
                                        for line in spec_relevant])
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
X = np.zeros((len(all_training_set),
              (training_frame.shape[0]*training_frame.shape[1])))
Y = np.asarray(frame_class)

# Arranging the data format into the X,Y input needed for sklearn
for i in range(len(all_training_set)):
    for j in range(len(all_training_set[i])):
        X[i, (j*len(all_training_set[i][j])):
            ((j+1)*len(all_training_set[i][j]))] = all_training_set[i][j]

# %%
# ---------Training ------------------
# Prediction (only within the training set)
cross = 3
print("LinearSVC + RAW data:")
print(np.mean(cross_val_score(LinearSVC(), X, Y, cv=cross)))

# Classificator training
clf = LinearSVC()
clf.fit(X, Y)
# %%
# ---------Prediction -----------------

prediction = []
confidence = []
predict_time = []

# Preparing buffer for computing the prediciton
# Aca se convierte el espectro en un array y se lo evalua usando el fit (clf)
predict_vector = np.zeros((window_height * window_width))

# Goes over all the data (leaving the windows at the beginning and the end)
for j in range(half_window, len(bins)-half_window):
    # Takes the frame
    prediction_frame = np.asarray([10*np.log10
                                   (np.asarray
                                    (line[(j-half_window):(j+1+half_window)]))
                                   for line in spec_relevant])
    # Arrange the frame to form a vector
    for i in range(len(prediction_frame)):
        l_frame = len(prediction_frame[i])
        predict_vector[(i*l_frame):((i+1)*l_frame)] = prediction_frame[i]
    # Append the time and the class prediction
    predict_time.append(bins[j])
    prediction.append(clf.predict([predict_vector])[0])
    confidence.append(clf.decision_function([predict_vector])[0])
# %%
fig, ax = plt.subplots(3, figsize=(12, 12), sharex=True)
spectrogram, freqs, bins, im = ax[0].specgram(signal, NFFT=NFFT, Fs=samp_rate,
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
spectrogram, freqs, bins, im = ax[1].specgram(signal, NFFT=NFFT, Fs=samp_rate,
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

ax[2].plot(times, signal)
axt = ax[2].twinx()
for syl in syl_classes:
    syl_class = syl_classes[syl]
    if len(syl_classes) == 2:
        axt.plot(predict_time, confidence)
    else:
        axt.plot(predict_time, [x[syl_class] for x in confidence], label=syl,
                 c=syl_colors[syl])
axt.legend()
axt.set_ylabel('Confidence')
axt.set_xlabel('Time [sec]')

# %% Prediction new files
song_files = glob.glob(os.path.join(files_folder, '*.wav'))
fig, ax = plt.subplots(len(song_files), figsize=(30, 6), sharex=False)
n = 0
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
    syl_prediction = {key: [] for key in syl_classes}
    for syl in syl_classes:
        syl_prediction[syl] = np.asarray(prediction.copy()).astype(np.double)
        syl_prediction[syl][syl_prediction[syl] != syl_classes[syl]] = np.nan
        axt.fill_between(predict_time,
                         0*syl_prediction[syl]-0.1, 0*syl_prediction[syl]+0.1,
                         color=syl_colors[syl], label=syl)
    axt.legend()
    axt.set_ylim(-4.1, 0.1)
    n += 1
