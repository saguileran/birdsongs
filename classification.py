import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import scipy
import scipy.io.wavfile
from sklearn.cross_validation import cross_val_score
from sklearn.svm import LinearSVC

os.chdir('/home/juan/Documentos/Musculo/Codigo canarios/Files')
# Read input audio file and define times
samp_rate, signal = scipy.io.wavfile.read("AmaVio_2017-03-03_07.13.48_s_75.wav")
times = np.arange(len(signal)) / float(samp_rate)

# Define the amount of points used for the fourier transform in each frame of the spectogram
# NFFT = int(0.01*samp_rate) -> asi estaba definido por la cant de segundos
NFFT = 512 # Una potencia de 2 es mas rapido (puedo usar FFT)

spectrogram, freqs, bins, im = plt.specgram(signal, NFFT=NFFT, Fs=samp_rate, noverlap=int(NFFT / 2))


# ------------Generating the trainig set---------------

# Size of the feature image
window_width = 25# THIS MUST BE ODD
half_window = int(window_width / 2)
window_bot = 5
window_top = 75
window_hight = window_top - window_bot

# Buffer matrix for computing the frames
training_frame = np.zeros((window_hight,window_width))

# List of classification tags (in this case 1 for the syllabe and 0 for the rest)
frame_class = []

## Defining the positive training set

positive_training_set = []

# list of the begining and end of "real times" of the desired syllabe
syl_beg = [0.917714, 9.017465, 17.163162, 24.291455, 35.434351]
syl_end = [1.692058, 10.993748, 18.074891, 25.212168, 36.486442]

for i in range(len(syl_beg)):
    # conversion from times to 'bins' vector coordinates 
    time_beg_coord = (np.abs(bins - syl_beg[i])).argmin()
    time_end_coord = (np.abs(bins - syl_end[i])).argmin()
    for j in range(time_beg_coord, time_end_coord):
        print(j)
        # Taking the frame from the spectrogram
        training_frame = np.asarray([10 * np.log10(np.asarray(line[(j - half_window):(j + 1 + half_window)])) for line in spectrogram[window_bot:window_top]])
        positive_training_set.append(training_frame)
        frame_class.append(1)

## Defining the negative training set

negative_training_set = []

# list of begining and end "real times" of the desaire syllabe
no_syl_beg = [14.434077, 18.074891, 32.050814, 52.149885]
no_syl_end = [17.163162, 24.291455, 35.434351, 53.110866]

for i in range(len(no_syl_beg)):
    # conversion from times to 'bins' vector coordinates 
    time_beg_coord = (np.abs(bins - no_syl_beg[i])).argmin()
    time_end_coord = (np.abs(bins - no_syl_end[i])).argmin()
    for j in range(time_beg_coord, time_end_coord):
        print(j)
        # Taking the frame from the spectrogram
        training_frame = np.asarray([10 * np.log10(np.asarray(line[(j - half_window):(j + 1 + half_window)])) for line in spectrogram[window_bot:window_top]])
        negative_training_set.append(training_frame)
        frame_class.append(0)

# Shuffle the training list (we hace to shuffle things together)
import random

all_training_set = positive_training_set + negative_training_set
combined = list(zip(all_training_set, frame_class))
random.shuffle(combined)
all_training_set[:], frame_class[:] = zip(*combined)

# Prearing arrays for placing the data
X = np.zeros((len(all_training_set), (training_frame.shape[0] * training_frame.shape[1])))
Y = np.asarray(frame_class)
# Y[0:len(positive_training_set)] = np.ones(len(positive_training_set))

# Arranging the data format into the X,Y input needed for sklearn
for i in range(len(all_training_set)):
    for j in range(len(all_training_set[i])):
        X[i, (j * len(all_training_set[i][j])):((j+1) * len(all_training_set[i][j]))] = all_training_set[i][j]


# ---------Training ------------------


# Predition (only within the training set)
cross = 5
print("LinearSVC + RAW data:")
print (np.mean(cross_val_score(LinearSVC(), X, Y, cv=cross )))

# Classificator training
clf = LinearSVC()
clf.fit(X,Y)

# ---------Prediction -----------------

prediction = []
predict_time = []

# Preparing buffer for computing the prediciton
predict_vector = np.zeros((training_frame.shape[0] * training_frame.shape[1]))

# Goes over all the data (leaving the windows at the beginning and the end)
for j in range(half_window, len(bins)-half_window):
    # Takes the frame
    training_frame = np.asarray([10 * np.log10(np.asarray(line[(j - half_window):(j + 1 + half_window)])) for line in spectrogram[window_bot:window_top]])
    # Arrange the frame to form a vector
    for i in range(len(training_frame)):
        predict_vector[(i * len(training_frame[i])):((i+1) * len(training_frame[i]))] = training_frame[i]
    # Append the time and the class prediction
    predict_time.append(bins[j])
    prediction.append(clf.predict([predict_vector]))
#%%
fig, ax = plt.subplots(2, figsize=(12,6), sharex=True)
spectrogram, freqs, bins, im = ax[0].specgram(signal, NFFT=NFFT, Fs=samp_rate, noverlap=int(NFFT / 2))
for nn in range(len(syl_beg)):
    ax[0].axvspan(syl_beg[nn], syl_end[nn], color='k', alpha=0.3)
ax[1].plot(times, signal)
axt = ax[1].twinx() # Starts a new graph sharing the same sublplot (and x-axis)
axt.plot(predict_time, prediction,'r')
axt.set_ylabel('Deteccion de silaba')
axt.set_xlabel('Time [sec]')
