import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from sklearn.cross_validation import cross_val_score
from sklearn.svm import LinearSVC
import random


def consecutive(data, stepsize=1):
    """
    Separa en segmentos de datos consecutivos. Se usa para separar indices
    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


# %%
os.chdir('/home/juan/Documentos/Musculo/Codigo canarios/Files')
# Read input audio file and define times
samp_rate, signal = wavfile.read('AmaVio_2017-03-03_07.13.48_s_75.wav')
times = np.arange(len(signal))/float(samp_rate)

# Define the amount of points used for the fourier transform in each frame of
# the spectogram
# NFFT = int(0.01*samp_rate) -> asi estaba definido por la cant de segundos
NFFT = 512  # Una potencia de 2 es mas rapido (puedo usar FFT)
spectrogram, freqs, bins, im = plt.specgram(signal, NFFT=NFFT, Fs=samp_rate,
                                            noverlap=int(NFFT/1.5),
                                            cmap='Greys')
# %%
# ------------Generating the training set---------------

# Size of the feature image
dfreq = freqs[1]-freqs[0]
min_freq = 500
max_freq = 8000
time_window = 0.25   # Time window in seconds
window_width = int(time_window/np.diff(bins)[0])   # THIS MUST BE ODD
if window_width%2 == 0:
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

# Defining the positive training set
A_training_set = []

# list of the begining and end of "real times" of the desired syllabe
A_beg = [7.45, 15.42, 22.59]#, 33.94]
A_end = [8.55, 16.25, 23.4]#, 34.75]

for i in range(len(A_beg)):
    # conversion from times to 'bins' vector coordinates
    # 'bins' = center times of spec
    time_beg_coord = (np.abs(bins-A_beg[i])).argmin()
    time_end_coord = (np.abs(bins-A_end[i])).argmin()
    for j in range(time_beg_coord, time_end_coord):
        # Taking the frame from the spectrogram
        training_frame = np.asarray([10*np.log10
                                     (np.asarray
                                      (line[(j-half_window):(j+1+half_window)])
                                      ) for line in spec_relevant])
        A_training_set.append(training_frame)
        frame_class.append(2)


# Defining the positive training set
B_training_set = []

# list of the begining and end of "real times" of the desired syllabe
B_beg = [0.917714, 9.017465, 17.163162, 24.291455]#, 35.434351]
B_end = [1.692058, 10.993748, 18.074891, 25.212168]#, 36.486442]

for i in range(len(B_beg)):
    # conversion from times to 'bins' vector coordinates
    # 'bins' = center times of spec
    time_beg_coord = (np.abs(bins-B_beg[i])).argmin()
    time_end_coord = (np.abs(bins-B_end[i])).argmin()
    for j in range(time_beg_coord, time_end_coord):
        # Taking the frame from the spectrogram
        training_frame = np.asarray([10*np.log10
                                     (np.asarray
                                      (line[(j-half_window):(j+1+half_window)])
                                      ) for line in spec_relevant])
        B_training_set.append(training_frame)
        frame_class.append(1)

# Defining the positive training set
C_training_set = []

# list of the begining and end of "real times" of the desired syllabe
C_beg = [14.51, 14.77, 21.75, 22.03, 28.06, 28.52, 28.82]
C_end = [14.65, 14.94, 21.88, 22.20, 28.22, 28.72, 29.01]

for i in range(len(C_beg)):
    # conversion from times to 'bins' vector coordinates
    # 'bins' = center times of spec
    time_beg_coord = (np.abs(bins-C_beg[i])).argmin()
    time_end_coord = (np.abs(bins-C_end[i])).argmin()
    for j in range(time_beg_coord, time_end_coord):
        # Taking the frame from the spectrogram
        training_frame = np.asarray([10*np.log10
                                     (np.asarray
                                      (line[(j-half_window):(j+1+half_window)])
                                      ) for line in spec_relevant])
        C_training_set.append(training_frame)
        frame_class.append(3)

# Defining the positive training set
D_training_set = []

# list of the begining and end of "real times" of the desired syllabe
D_beg = [7.2, 15.15]
D_end = [7.45, 15.37]

for i in range(len(D_beg)):
    # conversion from times to 'bins' vector coordinates
    # 'bins' = center times of spec
    time_beg_coord = (np.abs(bins-D_beg[i])).argmin()
    time_end_coord = (np.abs(bins-D_end[i])).argmin()
    for j in range(time_beg_coord, time_end_coord):
        # Taking the frame from the spectrogram
        training_frame = np.asarray([10*np.log10
                                     (np.asarray
                                      (line[(j-half_window):(j+1+half_window)])
                                      ) for line in spec_relevant])
        D_training_set.append(training_frame)
        frame_class.append(4)

# Defining the negative training set
S_training_set = []

# list of begining and end "real times" of the desaire syllabe
#S_beg = [15.03, 16.25, 18.074891]
#S_end = [15.42, 17.163162, 19.60]
S_beg = [12.]
S_end = [14.]
for i in range(len(S_beg)):
    # conversion from times to 'bins' vector coordinates
    time_beg_coord = (np.abs(bins - S_beg[i])).argmin()
    time_end_coord = (np.abs(bins - S_end[i])).argmin()
    for j in range(time_beg_coord, time_end_coord):
        # Taking the frame from the spectrogram
        training_frame = np.asarray([10*np.log10
                                     (np.asarray
                                      (line[(j-half_window):(j+1+half_window)])
                                      ) for line in spec_relevant])
        S_training_set.append(training_frame)
        frame_class.append(0)

# Shuffle the training list (we have to shuffle things together)

all_training_set = A_training_set+B_training_set+C_training_set+D_training_set+S_training_set
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
predict_vector = np.zeros((training_frame.shape[0] * training_frame.shape[1]))

# Goes over all the data (leaving the windows at the beginning and the end)
for j in range(half_window, len(bins)-half_window):
    # Takes the frame
    training_frame = np.asarray([10*np.log10(np.asarray(line[(j-half_window):(j+1+half_window)]))
                                for line in spec_relevant])
    # Arrange the frame to form a vector
    for i in range(len(training_frame)):
        predict_vector[(i*len(training_frame[i])):((i+1)*len(training_frame[i]))] = training_frame[i]
    # Append the time and the class prediction
    predict_time.append(bins[j])
    prediction.append(clf.predict([predict_vector])[0])
    confidence.append(clf.decision_function([predict_vector])[0])
# %%
fig, ax = plt.subplots(3, figsize=(12, 6), sharex=True)
spectrogram, freqs, bins, im = ax[0].specgram(signal, NFFT=NFFT, Fs=samp_rate,
                                              noverlap=int(NFFT/2),
                                              cmap='Greys')

for nn in range(len(A_beg)):
    ax[0].axvspan(A_beg[nn], A_end[nn], color='k', alpha=0.5,
                  label='A Training set' if nn == 0 else '')
for nn in range(len(B_beg)):
    ax[0].axvspan(B_beg[nn], B_end[nn], color='g', alpha=0.5,
                  label='B Training set' if nn == 0 else '')
for nn in range(len(C_beg)):
    ax[0].axvspan(C_beg[nn], C_end[nn], color='c', alpha=0.5,
                  label='C Training set' if nn == 0 else '')
for nn in range(len(D_beg)):
    ax[0].axvspan(D_beg[nn], D_end[nn], color='y', alpha=0.5,
                  label='D Training set' if nn == 0 else '')
for nn in range(len(S_beg)):
    ax[0].axvspan(S_beg[nn], S_end[nn], color='r', alpha=0.5,
                  label='S Training set' if nn == 0 else '')

prediction = np.asarray(prediction).astype(np.double)
A_prediction = prediction.copy()
A_prediction[A_prediction != 2] = np.nan
B_prediction = prediction.copy()
B_prediction[B_prediction != 1] = np.nan
C_prediction = prediction.copy()
C_prediction[C_prediction != 3] = np.nan
D_prediction = prediction.copy()
D_prediction[D_prediction != 4] = np.nan
S_prediction = prediction.copy()
S_prediction[S_prediction != 0] = np.nan

ax[0].legend()
ax[0].set_ylim(0, 8000)
spectrogram, freqs, bins, im = ax[1].specgram(signal, NFFT=NFFT, Fs=samp_rate,
                                              noverlap=int(NFFT/2),
                                              cmap='Greys')
ax[1].set_ylim(0, 8000)
axt = ax[1].twinx()
axt.fill_between(predict_time, 0*A_prediction-0.1, 0*A_prediction+0.1,
                 color='k', label='A')
axt.fill_between(predict_time, 0*B_prediction-0.1, 0*B_prediction+0.1,
                 color='g', label='B')
axt.fill_between(predict_time, 0*C_prediction-0.1, 0*C_prediction+0.1,
                 color='c', label='C')
axt.fill_between(predict_time, 0*D_prediction-0.1, 0*D_prediction+0.1,
                 color='y', label='D')
axt.fill_between(predict_time, 0*S_prediction-0.1, 0*S_prediction+0.1,
                 color='r', label='S')
axt.legend()
axt.set_ylim(-2, 2)
ax[2].plot(times, signal)
axt = ax[2].twinx()
axt.plot(predict_time, [x[4] for x in confidence], label='D', c='y')
axt.plot(predict_time, [x[3] for x in confidence], label='C', c='c')
axt.plot(predict_time, [x[2] for x in confidence], label='A', c='k')
axt.plot(predict_time, [x[1] for x in confidence], label='B', c='g')
axt.plot(predict_time, [x[0] for x in confidence], label='S', c='r')
axt.legend()
axt.set_ylabel('Confidence')
axt.set_xlabel('Time [sec]')

## %% Test in new file
#samp_rate, signal_test = wavfile.read('AmaVio_2018-03-06_07.45.18_s_1.wav')
#times_t = np.arange(len(signal_test))/float(samp_rate)
#t_spectrogram, t_freqs, t_bins, t_im = plt.specgram(signal_test, NFFT=NFFT,
#                                                    Fs=samp_rate,
#                                                    noverlap=int(NFFT/1.5),
#                                                    cmap='Greys')
## %%
## ---------Prediction -----------------
#t_spec_relevant = t_spectrogram[window_bot:window_top]
#
#t_prediction = []
#t_confidence = []
#t_predict_time = []
#
## Preparing buffer for computing the prediciton
## Aca se convierte el espectro en un array y se lo evalua usando el fit (clf)
#predict_vector = np.zeros((training_frame.shape[0] * training_frame.shape[1]))
#
## Goes over all the data (leaving the windows at the beginning and the end)
#for j in range(half_window, len(t_bins)-half_window):
#    # Takes the frame
#    training_frame = np.asarray([10*np.log10(np.asarray(line[(j-half_window):(j+1+half_window)]))
#                                for line in t_spec_relevant])
#    # Arrange the frame to form a vector
#    for i in range(len(training_frame)):
#        predict_vector[(i*len(training_frame[i])):((i+1)*len(training_frame[i]))] = training_frame[i]
#    # Append the time and the class prediction
#    t_predict_time.append(t_bins[j])
#    t_prediction.append(clf.predict([predict_vector])[0])
#    t_confidence.append(clf.decision_function([predict_vector])[0])
#
#t_prediction = np.asarray(t_prediction).astype(np.double)
#A_prediction = t_prediction.copy()
#A_prediction[A_prediction != 2] = np.nan
#B_prediction = t_prediction.copy()
#B_prediction[B_prediction != 1] = np.nan
#C_prediction = t_prediction.copy()
#C_prediction[C_prediction != 3] = np.nan
#S_prediction = t_prediction.copy()
#S_prediction[S_prediction != 0] = np.nan
#
#fig, ax = plt.subplots(1, figsize=(12, 6), sharex=True)
#spectrogram, freqs, bins, im = ax.specgram(signal_test, NFFT=NFFT, Fs=samp_rate,
#                                              noverlap=int(NFFT/2), cmap='Greys')
#ax.legend()
#ax.set_ylim(0, 8000)
#axt = ax.twinx()
#axt.fill_between(t_predict_time, 0*A_prediction-0.1, 0*A_prediction+0.1, color='k', label='A')
#axt.fill_between(t_predict_time, 0*B_prediction-0.1, 0*B_prediction+0.1, color='g', label='B')
#axt.fill_between(t_predict_time, 0*C_prediction-0.1, 0*C_prediction+0.1, color='c', label='C')
#axt.fill_between(t_predict_time, 0*S_prediction-0.1, 0*S_prediction+0.1, color='r', label='S')
#axt.legend()
#axt.set_ylim(-2, 2)
