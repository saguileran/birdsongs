from .functions import *

class Syllable(object):
    """
    Store and define a syllable and its properties
    INPUT:
        s  = signal
        fs = sampling rate
        t0 = initial time of the syllable
    """
    def __init__(self, s, fs, t0, Nt=200, llambda=1.5, NN=512, overlap=0.5, flim=(1.5e3,2e4), center=False, n_mels=32):
        self.t0 = t0
        self.Nt = Nt
        self.NN = NN
        self.fs = fs
        self.n_mels = n_mels
        
        self.flim       = flim
        self.llambda    = llambda
        self.no_overlap = int(overlap*self.NN )
        
        self.s        = sound.normalize(s, max_amp=1.0)
        self.time     = np.linspace(0, len(self.s)/self.fs, len(self.s))
        self.envelope = self.Enve(self.s)
        self.Vs       = [] # np.zeros((self.s.size, 5)); # velocity
        
        Sxx_power, tn, fn, ext = sound.spectrogram (self.s, self.fs, nperseg=self.NN, noverlap=self.no_overlap, mode='psd')  
        Sxx_dB = util.power2dB(Sxx_power) + 96
        Sxx_dB_noNoise, noise_profile, _ = sound.remove_background(Sxx_dB, gauss_std=self.NN//10, gauss_win=self.NN//5, 
                                                                   llambda=self.llambda) # remove back ground noise 

        self.fu     = fn
        self.tu     = np.linspace(0, self.time[-1], Sxx_power.shape[1]) 
        self.Sxx    = sound.smooth(Sxx_dB, std=0.5)
        self.Sxx_dB = util.power2dB(self.Sxx) + 96
        
        # -------------------------------------------------------------------
        # ------------- ACOUSTIC FEATURES -----------------------------------
        # -------------------------------------------------------------------
        #numbers: energy, Ht, Hf, EAS, ECU, ECV, EPS, EPS_KURT, EPS_SKEW
        #vectors: rms, zcr, centroid, rolloff, rolloff_min, onset_env
        #matrix: contrast, mfccs, FFT_coef, s_mel
        self.center = center
        # ----------------------- scalar 
        EAS, ECU, ECV, EPS, EPS_KURT, EPS_SKEW = features.spectral_entropy(Sxx_power, fn, flim=self.flim) 
        ACI_xx, ACI_per_bin, ACI_sum           = features.acoustic_complexity_index(self.Sxx)
        
        energy =  Norm(self.s,ord=2)/self.s.size
        Ht     = features.temporal_entropy (self.s,    compatibility='seewave', mode='fast', Nt=self.NN)
        Hf, _  = features.frequency_entropy(Sxx_power, compatibility='seewave')
        
        self.BI     = features.bioacoustics_index(self.Sxx, self.fu, flim=self.flim, R_compatible='seewave')
        # ----------------------- vector ------------------------------
        self.freq    = fft_frequencies(sr=self.fs, n_fft=self.NN) 
        self.FF_coef = np.abs(stft(y=self.s, n_fft=self.NN, hop_length=self.NN//2, win_length=None, 
                     center=self.center, dtype=None, pad_mode='constant'))
        
        self.f_msf   = np.array([Norm(self.FF_coef[:,i]*self.freq, 1)/Norm(self.FF_coef[:,i], 1) for i in range(self.FF_coef.shape[1])])        
        self.FF_time = np.linspace(0,self.time[-1], self.FF_coef.shape[1]) #times_like(self.FF_coef, sr=self.fs, hop_length=self.NN//2, n_fft=self.NN)
        
        self.centroid = feature.spectral_centroid(y=self.s, sr=self.fs, S=self.Sxx_dB, n_fft=self.NN, 
                                                  hop_length=self.NN//2, freq=None, 
                                             win_length=None, center=self.center, pad_mode='constant')[0]
        self.rms      = feature.rms(y=self.s, S=self.Sxx, frame_length=self.NN, hop_length=self.NN//2, 
                                    center=self.center, pad_mode='constant')[0]
        
        self.mfccs    = feature.mfcc(y=self.s, sr=self.fs, S=self.Sxx, n_mfcc=126, dct_type=2, norm='ortho', lifter=0)
        
#         self.NOP    = features.number_of_peaks(self.Sxx, self.fu, mode='dB', min_peak_val=0.05, min_freq_dist=1e3, 
#                                         slopes=(1, 1), prominence=0, display=False) # Number Of Peaks
        
        # # pitches[..., f, t] contains instantaneous frequency at bin f, time t
        # # magnitudes[..., f, t] contains the corresponding magnitudes.
        # # Both pitches and magnitudes take value 0 at bins of non-maximal magnitude.
        # pitches, magnitudes = librosa.piptrack(y=self.s, sr=self.fs, S=self.Sxx, n_fft=self.NN, hop_length=self.NN//2,
        #                        fmin=self.flim[0], fmax=self.flim[1], threshold=0.01, win_length=None, 
        #                        center=self.center, pad_mode='constant', ref=None)
        # features.zero_crossing_rate(self.s, self.fs)
#         self.zcr         = feature.zero_crossing_rate(self.s, frame_length=self.NN, hop_length=self.NN//2, 
#                                                       center=self.center) 
#         self.rolloff     = feature.spectral_rolloff(y=self.s, sr=self.fs, S=self.Sxx_dB, n_fft=self.NN, 
#                                                     hop_length=self.NN//2, win_length=None, center=self.center,
#                                                     pad_mode='constant', freq=None, roll_percent=0.6)[0]
#         self.rolloff_min = feature.spectral_rolloff(y=self.s, sr=self.fs, S=self.Sxx_dB, n_fft=self.NN, 
#                                                  hop_length=self.NN//2, win_length=None, 
#                                  center=self.center, pad_mode='constant', freq=None, roll_percent=0.2)[0]
#         self.onset_env   = onset.onset_strength(y=self.s, sr=self.fs, S=self.Sxx, lag=1, max_size=1, fmax=self.flim[1],
#                                                  ref=None, detrend=False, center=self.center, feature=None, aggregate=None)

#         # ----------------------- matrix  ----------------------------
        # self.contrast = feature.spectral_contrast(y=self.s, sr=self.fs, S=self.Sxx, n_fft=self.NN, hop_length=self.NN//2, 
        #                             win_length=None, center=self.center, pad_mode='constant', freq=None, 
        #                             fmin=self.flim[0], n_bands=4,quantile=0.02, linear=False)
        self.s_mel    = feature.melspectrogram(y=self.s, sr=self.fs, S=self.Sxx, n_fft=self.NN, 
                                       hop_length=self.NN//2, win_length=None, center=self.center, 
                                       pad_mode='constant', power=2.0, n_mels=self.n_mels,
                                       fmin=self.flim[0], fmax=self.flim[1])
        # self.s_sal    = librosa.salience(S=self.FF_coef, freqs=self.freq, harmonics=[1, 2, 3, 4], weights=[1,1,1,1], 
        #                                  aggregate=None, filter_peaks=True, fill_value=0, kind='linear', axis=-2)
        # self.C        = librosa.cqt(y=self.s, sr=self.fs, hop_length=self.NN//2, fmin=self.flim[0], n_bins=32, 
        #                             bins_per_octave=12, tuning=0.0, filter_scale=1, norm=1, sparsity=0.01, 
        #                             window='hann', scale=True, pad_mode='constant', dtype=None)
        # self.D        = librosa.iirt(y=self.s, sr=self.fs, win_length=2*self.NN, hop_length=self.NN//2, center=self.center, 
        #                  tuning=0.0, pad_mode='constant', flayout='sos')
        
#         self.pitches    = pitches
#         self.magnitudes = magnitudes
        
        self.features  = [energy, Ht, Hf]
        self.entropies = [EAS, ECU, ECV, EPS, EPS_KURT, EPS_SKEW]
        
        # self.times_on = times_on
        
        self.T        = self.tu[-1]-self.tu[0]
        
        # # ------------- "better method" --------------
        # self.FF     = pyin(self.s, fmin=self.flim[0], fmax=self.flim[1], sr=self.fs, frame_length=2*self.NN, 
        #                win_length=None, hop_length=self.NN//2, n_thresholds=100, beta_parameters=(2, 18), 
        #                boltzmann_parameter=2, resolution=0.1, max_transition_rate=35.92, switch_prob=0.01, 
        #                no_trough_prob=0.01, fill_na=0, center=self.center, pad_mode='constant')
        self.FF     = yin(self.s, fmin=self.flim[0], fmax=self.flim[1], sr=self.fs, frame_length=2*self.NN, 
                          win_length=None, hop_length=self.NN//2, trough_threshold=1, center=self.center, pad_mode='constant')
        self.timeFF = np.linspace(0,self.time[-1],self.FF.size)
        self.FF_fun = interp1d(self.timeFF, self.FF)
        self.SCI    = self.f_msf / self.FF_fun(self.FF_time)
    
    def AlphaBeta(self): 
        a = np.array([self.p["a0"].value, self.p["a1"].value, self.p["a2"].value]);   
        b = np.array([self.p["b0"].value, self.p["b1"].value, self.p["b2"].value])
        
        t_1   = np.linspace(0,self.T,len(self.s))   
        t_par = np.array([np.ones(t_1.size), t_1, t_1**2])
        
        self.alpha = np.dot(a, t_par);  # lines (or parabolas)
        
        # define by same shape as fudamenta frequency
        if "syllable" in self.id: poly = Polynomial.fit(self.timeFF, self.FF, deg=10)
        elif "chunck" in self.id: poly = Polynomial.fit(self.timeFF, self.FF, deg=1)
            
        x, y = poly.linspace(np.size(self.s))
        self.beta  = b[0] + b[1]*(1e-4*y) + b[2]*(1e-4*y)**2    
            
    def MotorGestures(self, ovfs=20, prct_noise=0):  # ovfs:oversamp
        #sampling and necessary constants
        dt   = 1./(ovfs*self.fs)
        tmax = int(self.s.size)*ovfs
        # vectors initialization
        pi, pb = np.zeros(tmax), np.zeros(tmax)
        out    = np.zeros(int(self.s.size))
        # initial derivative vector (ODEs), it is not too relevant
        v = 1e-4*np.array([1e2, 1e1, 1, 1, 1, 1]);  #1e-12*np.array([5, 10, 1, 1, 10, 1]);  
        # ------------- BIRD PARAMETERS -----------
        # - Trachea:
        #           r: reflection coeficient    [adimensionelss]
        #           L: trachea length           [m]
        #           c: speed of sound in media  [m/s]
        # - Beak, Glottis and OEC:
        #           CH: OEC Compliance          [m^3/Pa]
        #           MB: Beak Inertance          [Pa s^2/m^3 = kg/m^4]
        #           MG: Glottis Inertance       [Pa s^2/m^3 = kg/m^4]
        #           RB: Beak Resistance         [Pa s/m^3 = kg/m^4 s]
        #           Rh: OEC Resistence          [Pa s/m^3 = kg/m^4 s]
        BirdData = pd.read_csv(self.paths.auxdata+'ZonotrichiaData.csv')
        c, L, r, Ch, MG, MB, RB, Rh = BirdData['value'] # #c, L, r, c, L1, L2, r2, rd 
        t = tau = int(L/c/dt) 
        # before the function calling, gm, alpha, beta, and A must be define
        def ODEs(v, dv=np.zeros(6)):
            [x, y, pout, i1, i2, i3] = v  # (x, y, pout, i1, i2, i3)'
            dv[0] = y
            dv[1] = (-self.alpha[t//ovfs]-self.beta[t//ovfs]*x-x**3+x**2)*self.p["gm"].value**2 - (x**2*y+x*y)*self.p["gm"].value
            # ------------------------- trachea ------------------------
            pbold = pb[t]                                 # pressure back before
            # Pin(t) = Ay(t)+pback(t-L/C) = envelope_Signal*v[1]+pb[t-tau]
            pi[t] = (.5*self.envelope[t//ovfs])*dv[1] + pb[t-tau] 
            pb[t] = -r*pi[t-tau]                          # pressure back after: -rPin(t-tau) 
            pout  = (1-r)*pi[t-tau]                       # pout
            # ---------------------------------------------------------------
            dv[2] = (pb[t]-pbold)/dt                      # dpout
            dv[3] = i2
            dv[4] = -(1/Ch/MG)*i1 - Rh*(1/MB+1/MG)*i2 +(1/MG/Ch+Rh*RB/MG/MB)*i3 \
                    +(1/MG)*dv[2] + (Rh*RB/MG/MB)*pout
            dv[5] = -(MG/MB)*i2 - (Rh/MB)*i3 + (1/MB)*pout
            return dv        
        
        while t < tmax: # and v[1] > -5e6:  # labia velocity not too fast
            v = rk4(ODEs, v, dt);  self.Vs.append(v)  # RK4 - step
            out[t//ovfs] = RB*v[-1]               # output signal (synthetic)
            t += 1;
        #self.Vs = np.array(self.Vs)
        
        # define solution (synthetic syllable) as a Syllable object 
        synth = Syllable(out, self.fs, self.t0,  Nt=self.Nt, llambda=self.llambda, NN=self.NN)
        synth.no_syllable = self.no_syllable
        synth.no_file     = self.no_file
        synth.p           = self.p
        synth.paths       = self.paths
        synth.id          = self.id+"-synth"
        synth.Vs          = self.Vs
        
        return synth
        
    def Enve(self, out):
        out_env = sound.envelope(out, Nt=self.Nt) 
        t_env = np.arange(0,len(out_env),1)*len(out)/self.fs/len(out_env)
        t_env[-1] = self.time[-1] 
        fun_s = interp1d(t_env, out_env)
        return fun_s(self.time)
        
    def WriteAudio(self):
        sound.write('{}/File{}-{}-{}.wav'.format(self.paths.examples,self.no_file, self.id,self.no_syllable), 
                                                 self.fs, np.asarray(self.s,  dtype=np.float32))
        
    def Solve(self, p, orde=2):
        self.ord = orde; self.p = p; # order of score norms and define parameteres to optimize
        self.AlphaBeta()             # define alpha and beta parameters
        synth = self.MotorGestures() # solve the problem and define the synthetic syllable
        
        # deltaNOP    = np.abs(synth.NOP-self.NOP).astype(float)
        deltaSxx    = np.abs(synth.Sxx_dB-self.Sxx_dB)
        deltaMel    = np.abs(synth.FF_coef-self.FF_coef)
        deltaMfccs  = np.abs(synth.mfccs-self.mfccs)
        
        synth.deltaFmsf     = np.abs(synth.f_msf-self.f_msf)
        synth.deltaSCI      = np.abs(synth.SCI-self.SCI)
        synth.deltaEnv      = np.abs(synth.envelope-self.envelope)
        synth.deltaFF       = 1e-3*np.abs(synth.FF-self.FF)#/np.max(deltaFF)
        synth.deltaRMS      = np.abs(synth.rms-self.rms)
        synth.deltaCentroid = 1e-3*np.abs(synth.centroid-self.centroid)
        synth.deltaF_msf    = 1e-3*np.abs(synth.f_msf-self.f_msf)
        synth.deltaSxx      = deltaSxx/np.max(deltaSxx)
        synth.deltaMel      = deltaMel/np.max(deltaMel)
        synth.deltaMfccs    = deltaMfccs/np.max(deltaMfccs)
            
        synth.scoreSCI      = Norm(synth.deltaSCI,      ord=self.ord)/synth.deltaSCI.size
        synth.scoreFF       = Norm(synth.deltaFF,       ord=self.ord)/synth.deltaFF.size
        synth.scoreEnv      = Norm(synth.deltaEnv,      ord=self.ord)/synth.deltaEnv.size
        synth.scoreRMS      = Norm(synth.deltaRMS,      ord=self.ord)/synth.deltaRMS.size
        synth.scoreCentroid = Norm(synth.deltaCentroid, ord=self.ord)/synth.deltaCentroid.size
        synth.scoreF_msf    = Norm(synth.deltaF_msf,    ord=self.ord)/synth.deltaF_msf.size
        synth.scoreSxx      = Norm(synth.deltaSxx,      ord=np.inf)/synth.deltaSxx.size
        synth.scoreMel      = Norm(synth.deltaMel,      ord=np.inf)/synth.deltaSxx.size
        synth.scoreMfccs    = Norm(synth.deltaMfccs,    ord=np.inf)/synth.deltaMfccs.size
        
        # synth.scoreNoHarm        = deltaNOP*10**(deltaNOP-2)
        synth.deltaSCI_mean      = synth.deltaSCI.mean()
        synth.deltaFF_mean       = synth.deltaFF.mean()
        synth.scoreRMS_mean      = synth.scoreRMS.mean()
        synth.scoreCentroid_mean = synth.scoreCentroid.mean()
        synth.deltaEnv_mean      = synth.deltaEnv.mean()
        synth.scoreF_msf_mean    = synth.deltaF_msf.mean()
        
        # -------         acoustic dissimilarity --------------------
        synth.correlation = np.zeros_like(self.FF_time)
        synth.Df          = np.zeros_like(self.FF_time)
        synth.SKL         = np.zeros_like(self.FF_time)
        for i in range(synth.mfccs.shape[1]):
            x, y = self.mfccs[:,i], synth.mfccs[:,i]
            r = Norm(x*y,ord=1)/(Norm(x,ord=2)*Norm(y,ord=2))
            #print(Norm(x*y,ord=1), Norm(x,ord=2), Norm(y,ord=2), r)
            
            synth.correlation[i] = np.sqrt(1-r)
            synth.Df[i]          = 0.5*Norm(x*np.log2(np.abs(x/y))+y*np.log2(np.abs(y/x)), ord=1)
            synth.SKL[i]         = 0.5*Norm(np.abs(x-y), ord=1)
        
            #synth.Df[np.argwhere(np.isnan(synth.Df))]=-10
        
        #synth.correlation /= synth.correlation.max()
        synth.SKL         /= synth.SKL.max()
        synth.Df          /= synth.Df.max()
        
        synth.scoreCorrelation = Norm(synth.correlation, ord=self.ord)#/synth.correlation.size
        synth.scoreSKL         = Norm(synth.SKL, ord=self.ord)#/synth.SKL.size
        synth.scoreDF          = Norm(synth.Df, ord=self.ord)#/synth.Df.size
        
        return synth