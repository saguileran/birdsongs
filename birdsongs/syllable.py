from .functions import *

class Syllable(object):
    """
    Store and define a syllable and its properties
    INPUT:
        s  = signal
        fs = sampling rate
        t0 = initial time of the syllable
    """
    def __init__(self, s, fs, t0, Nt=200, llambda=1.5, NN=512, overlap=0.5, flim=(1.5e3,2e4), center=False):
        self.t0 = t0
        self.Nt = Nt
        self.NN = NN
        self.fs = fs
        self.llambda = llambda
        self.flim = flim
        self.no_overlap = int(overlap*self.NN )
        
        self.s        = sound.normalize(s, max_amp=1.0)
        self.time     = np.linspace(0, len(self.s)/self.fs, len(self.s))
        self.envelope = self.Enve(self.s)
        
        Sxx_power, tn, fn, ext = sound.spectrogram (self.s, self.fs, nperseg=self.NN, noverlap=self.no_overlap, mode='psd')  
        Sxx_dB = util.power2dB(Sxx_power) + 96
        Sxx_dB_noNoise, noise_profile, _ = sound.remove_background(Sxx_dB, gauss_std=self.NN//10, gauss_win=self.NN//5, llambda=self.llambda)

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
        # ----------------------- scalar features
        energy =  Norm(self.s,ord=2)/self.s.size
        Ht     = features.temporal_entropy (self.s, compatibility='seewave', mode='fast', Nt=self.NN)
        Hf, _  = features.frequency_entropy(Sxx_power, compatibility='seewave')
        #EAS, ECU, ECV, EPS, EPS_KURT, EPS_SKEW = features.spectral_entropy(Sxx_power, fn, flim=self.flim) 
        
        
        # ----------------------- vector features
        self.freq    = fft_frequencies(sr=self.fs, n_fft=self.NN) 
        self.FF_coef = np.abs(stft(y=self.s, n_fft=self.NN, hop_length=self.NN//2, win_length=None, 
                     center=self.center, dtype=None, pad_mode='constant'))
        #self.FF_coef /= np.max(self.FF_coef)
        self.f_msf = np.array([Norm(self.FF_coef[:,i]*self.freq, 1)/Norm(self.FF_coef[:,i], 1) for i in range(self.FF_coef.shape[1])])
        
        self.FF_time = np.linspace(0,self.time[-1], self.FF_coef.shape[1]) #times_like(self.FF_coef, sr=self.fs, hop_length=self.NN//2, n_fft=self.NN)
        
        [centroid]      = feature.spectral_centroid(y=self.s, sr=self.fs, S=self.Sxx_dB, n_fft=self.NN, 
                                                  hop_length=self.NN//2, freq=None, 
                                             win_length=None, center=self.center, pad_mode='constant')
        [rms]         = feature.rms(y=self.s, S=self.Sxx, frame_length=self.NN, hop_length=self.NN//2, 
                          center=self.center, pad_mode='constant')
#         zcr           = feature.zero_crossing_rate(self.s, frame_length=self.NN, hop_length=self.NN//2, center=self.center) #features.zero_crossing_rate(self.s, self.fs)
        
#         [rolloff]     = feature.spectral_rolloff(y=self.s, sr=self.fs, S=self.Sxx, n_fft=self.NN, hop_length=self.NN//2, win_length=None, 
#                                  center=self.center, pad_mode='constant', freq=None, roll_percent=0.6)
#         [rolloff_min] = feature.spectral_rolloff(y=self.s, sr=self.fs, S=self.Sxx, n_fft=self.NN, hop_length=self.NN//2, win_length=None, 
#                                  center=self.center, pad_mode='constant', freq=None, roll_percent=0.2)
#         onset_env     = onset.onset_strength(y=self.s, sr=self.fs, S=self.Sxx, lag=1, max_size=1, fmax=12000,
#                              ref=None, detrend=False, center=self.center, feature=None, aggregate=None)

#         ----------------------- matrix features
#         FFT_coef = np.abs(stft(y=self.s,n_fft=self.NN, hop_length=self.NN//2, win_length=None,
#                             center=self.center, dtype=None, pad_mode='constant'))
#         times_on      = times_like(FFT_coef)
#         contrast = feature.spectral_contrast(y=self.s, sr=self.fs, S=self.Sxx, n_fft=self.NN, 
#                                              hop_length=self.NN//2, win_length=None, center=self.center, 
#                                              pad_mode='constant', freq=None, fmin=self.flim[0], n_bands=4,
#                                              quantile=0.02, linear=False)
        mfccs = feature.mfcc(y=self.s, sr=self.fs, S=self.Sxx, n_mfcc=20,
                             dct_type=2, norm='ortho', lifter=0)
        # s_mel = feature.melspectrogram(y=self.s, sr=self.fs, S=self.Sxx, n_fft=self.NN, 
        #                                hop_length=self.NN//2, win_length=None, center=self.center, 
        #                                pad_mode='constant', power=2.0, n_mels=126
        #                                fmin=self.flim[0], fmax=self.flim[1])
        
        self.energy = energy
        self.Ht = Ht
        self.Hf = Hf
        #self.entropyes = [EAS, ECU, ECV, EPS, EPS_KURT, EPS_SKEW]
        
        self.centroid = centroid
        self.rms = rms
        self.mfccs = mfccs 
#         self.zcr = zcr
#         self.rolloff = rolloff
#         self.rolloff_min = rolloff_min
#         self.onset_env = onset_env
        
#         self.FFT_coef = FFT_coef
#         self.times_on = times_on
#         self.contrast = contrast
#         self.mfccs    = mfccs
#         self.s_mel    = s_mel
        
        
        self.T        = self.tu[-1]-self.tu[0]
        
        #f0 = pyin(self.s, fmin=1000, fmax=15000, sr=self.fs, frame_length=128, win_length=None, hop_length=None, n_thresholds=100, beta_parameters=(2, 18), boltzmann_parameter=2, resolution=0.1, max_transition_rate=35.92, switch_prob=0.01, no_trough_prob=0.01, fill_na=nan, center=True, pad_mode='constant')
        self.FF     = yin(self.s, fmin=self.flim[0], fmax=self.flim[1], sr=self.fs, frame_length=2*self.NN, win_length=None, hop_length=self.NN//2, trough_threshold=1, center=self.center, pad_mode='constant')
        self.timeFF = np.linspace(0,self.time[-1],self.FF.size)
        self.NoHarm = 1 # NoHarm
        self.FF_fun = interp1d(self.timeFF, self.FF)
        self.SCI    = self.f_msf / self.FF_fun(self.FF_time)
    
    def AlphaBeta(self): 
        a = np.array([self.p["a0"].value, self.p["a1"].value, self.p["a2"].value]);   
        b = np.array([self.p["b0"].value, self.p["b1"].value, self.p["b2"].value])
        
        t_1   = np.linspace(0,self.T,len(self.s))   
        t_par = np.array([np.ones(t_1.size), t_1, t_1**2])
        # approx alpha and beta as polynomials
        poly = Polynomial.fit(self.timeFF, self.FF, deg=10)
        x, y = poly.linspace(np.size(self.s))
        
        self.beta  = b[0] + b[1]*(1e-4*y) + b[2]*(1e-4*y)**2
        self.alpha = np.dot(a, t_par);  # self.beta = np.dot(b, t_par)
    
    def MotorGestures(self, oversamp=20, prct_noise=0):
        N_total = len(self.s)

        #sampling and necessary constants
        sampling  = self.fs
        oversamp  = oversamp
        out_size  = int(N_total)
        dt        = 1./(oversamp*sampling)
        tmax      = out_size*oversamp

        # vectors initialization
        out = np.zeros(out_size)
        a   = np.zeros(tmax)
        db  = np.zeros(tmax)

        # counters 
        n_out, tcount, taux, tiempot = 0, 0, 0, 0
        forcing1, forcing2, A1 = 0., 0., 0
        
        v = 1e-12*np.array([5, 10, 1, 10, 1]);  self.Vs = [v]
        
        BirdData = pd.read_csv(self.paths.auxdata+'CopetonData.csv')
        c, ancho, largo, s1overCH, s1overLB, s1overLG, RB, r, rdis = BirdData['value']
        
        t = tau = int(largo/(c*dt)) #( + 0.0)
        def dxdt_synth(v):
            [x, y, i1, i2, i3], dv = v, np.zeros(5) #x, y, i1, i2, i3 = v[0], v[1], v[2], v[3], v[4]
            dv[0] = y
            dv[1] = - alpha*gm**2 - beta*gm**2*x - gm**2*x**3 - gm*x**2*y + gm**2*x**2 - gm*x*y
            dv[2] = i2
            dv[3] = -s1overLG*s1overCH*i1 - rdis*(s1overLB+s1overLG)*i2 \
                + i3*(s1overLG*s1overCH-rdis*RB*s1overLG*s1overLB) \
                + s1overLG*forcing2 + rdis*s1overLG*s1overLB*forcing1
            dv[4] = -(s1overLB/s1overLG)*i2 - RB*s1overLB*i3 + s1overLB*forcing1
            return dv

        gm, amplitud = self.p["gamma"].value, self.envelope[0]
        alpha, beta  = self.alpha[0], self.beta[0]
        
        while t < tmax and v[1] > -5e6:
            # -------- trachea ---------------
            dbold  = db[t]                              # forcing 1, before
            a[t]   = (.50)*(1.01*A1*v[1]) + db[t-tau]   # a = Pin, pressure:v(t)y(t) + Pin(t-tau) # envelope*
            db[t]  = -r*a[t-tau]                        # forcing 1, after: -rPin(t-tau)
            ddb    = (db[t]-dbold)/dt                   # Derivada, dPout/dt=Delta forcing1/dt

            #  -rPin(t-tau),  dPout/dt,   v(t)y(t) + Pin(t-tau)
            forcing1, forcing2, PRESSURE = db[t], ddb, a[t] 
            
            tiempot += dt
            v = rk4(dxdt_synth, v, dt);   

            noise    = 0.21*(uniform(0, 1)-0.5)
            A1       = amplitud + prct_noise*noise

            if taux == oversamp and n_out<self.fs-1:
                out[n_out]   = RB*v[4]*10  
                n_out       += 1;  self.Vs.append(v);
                
                alpha, beta = self.alpha[n_out], self.beta[n_out] 
                amplitud    = self.envelope[n_out]
                taux       = 0
            t += 1;   taux += 1;
        
        self.Vs = np.array(self.Vs)
        # pre processing synthetic data
        out               = sound.normalize(out, max_amp=1)
        synth_env         = self.Enve(out)
        out_amp           = np.zeros_like(out)
        not_zero          = np.where(synth_env > 0.005)
        out_amp[not_zero] = out[not_zero] * self.envelope[not_zero] / synth_env[not_zero]
        
        synth = Syllable(out_amp, self.fs, self.t0,  Nt=self.Nt, llambda=self.llambda, NN=self.NN)
        
        synth.no_syllable = self.no_syllable
        synth.no_file     = self.no_file
        synth.p           = self.p
        synth.paths       = self.paths
        synth.id          = "synth"
        synth.Vs          = self.Vs
        
        
        return synth
        
    def Enve(self, out):
        synth_env = sound.envelope(out, Nt=self.Nt) 
        t_env = np.arange(0,len(synth_env),1)*len(out)/self.fs/len(synth_env)
        t_env[-1] = self.time[-1] 
        fun_s = interp1d(t_env, synth_env)
        return fun_s(self.time)
        
        
    def WriteAudio(self):
        sound.write('{}/File{}-{}-{}.wav'.format(self.paths.examples,self.no_file, self.id,self.no_syllable), self.fs, np.asarray(self.s,  dtype=np.float32))
        
    # -------------- --------------
    def Solve(self, p, orde=2):
        #Display(p)
        self.ord = orde; self.p = p
        self.AlphaBeta()
        synth = self.MotorGestures() # solve the problem and define the synthetic syllable
        
        #deltaNoHarm = np.abs(synth.NoHarm_out-self.NoHarm).astype(float)
        deltaSxx    = np.abs(synth.Sxx_dB-self.Sxx_dB)
        deltaMel    = np.abs(synth.FF_coef-self.FF_coef)
        
        synth.deltaFmsf     = np.abs(synth.f_msf-self.f_msf)
        synth.deltaSCI      = np.abs(synth.SCI-self.SCI)
        synth.deltaEnv      = np.abs(synth.envelope-self.envelope)
        synth.deltaFF       = 1e-3*np.abs(synth.FF-self.FF) #/np.max(deltaFF)
        synth.deltaSxx      = deltaSxx/np.max(deltaSxx)
        synth.deltaMel      = deltaMel/np.max(deltaMel)
        synth.deltaRMS      = np.abs(synth.rms-self.rms)
        synth.deltaCentroid = 1e-3*np.abs(synth.centroid-self.centroid)
        synth.deltaF_msf    = 1e-3*np.abs(synth.f_msf-self.f_msf)
        
        #self.DeltaNoHarm = deltaNoHarm*10**(deltaNoHarm-2)
            
        synth.scoreSCI      = Norm(synth.deltaSCI, ord=self.ord)/synth.deltaSCI.size
        synth.scoreFF       = Norm(synth.deltaFF,  ord=self.ord)/synth.deltaFF.size
        synth.scoreEnv      = Norm(synth.deltaEnv, ord=self.ord)/synth.deltaEnv.size
        synth.scoreRMS      = Norm(synth.deltaRMS, ord=self.ord)/synth.deltaRMS.size
        synth.scoreCentroid = Norm(synth.deltaCentroid, ord=self.ord)/synth.deltaCentroid.size
        synth.scoreF_msf    = Norm(synth.deltaF_msf, ord=self.ord)/synth.deltaF_msf.size
        
        synth.scoreSxx    = Norm(synth.deltaSxx, ord=np.inf)/synth.deltaSxx.size
        synth.scoreMel    = Norm(synth.deltaMel, ord=np.inf)/synth.deltaSxx.size
        
        synth.deltaSCI_mean      = synth.deltaSCI.mean()
        synth.deltaFF_mean       = synth.deltaFF.mean()
        synth.scoreRMS_mean      = synth.scoreRMS.mean()
        synth.scoreCentroid_mean = synth.scoreCentroid.mean()
        synth.deltaEnv_mean      = synth.deltaEnv.mean()
        synth.scoreF_msf_mean    = synth.deltaF_msf.mean()
        
        return synth
        
    def residualSCI(self, p):
        syllable_synth = self.Solve(p)
        return syllable_synth.scoreSCI
    
    def residualFF(self, p):
        
        syllable_synth = self.Solve(p)
        return syllable_synth.scoreFF
    
    def residualFFandSCI(self, p):
        syllable_synth = self.Solve(p)
        return syllable_synth.scoreSCI+syllable_synth.scoreFF
    
    # ----------- OPTIMIZATION FUNCTIONS --------------
    def OptimalGamma(self, method_kwargs):
        kwargs = {k: method_kwargs[k] for k in set(list(method_kwargs.keys())) - set(["method"])}
    
        start = time.time()
        self.p["gamma"].set(vary=True)
        mi    = lmfit.minimize(self.residualSCI, self.p, nan_policy='omit', method=method_kwargs["method"], **kwargs) 
        self.p["gamma"].set(value=mi.params["gamma"].value, vary=False)
        end   = time.time()
        print("γ* =  {0:.0f}, t={1:.4f} min".format(self.p["gamma"].value, (end-start)/60))
        return mi.params["gamma"].value
    
    def OptimalBs(self, method_kwargs):
        kwargs = {k: method_kwargs[k] for k in set(list(method_kwargs.keys())) - set(["method"])}
        # ---------------- b0--------------------
        start0 = time.time()
        self.p["b0"].set(vary=True)
        mi0    = lmfit.minimize(self.residualFF, self.p, nan_policy='omit', method=method_kwargs["method"], **kwargs) 
        self.p["b0"].set(vary=False, value=mi0.params["b0"].value)
        end0   = time.time()
        print("b_0*={0:.4f}, t={1:.4f} min".format(self.p["b0"].value, (end0-start0)/60))
        # ---------------- b1--------------------
        start1 = time.time()
        self.p["b1"].set(vary=True)
        mi1    = lmfit.minimize(self.residualFF, self.p, nan_policy='omit', method=method_kwargs["method"], **kwargs) 
        self.p["b1"].set(vary=False, value=mi1.params["b1"].value)
        end1   = time.time()
        print("b_1*={0:.4f}, t={1:.4f} min".format(self.p["b1"].value, (end1-start1)/60))
        #return self.p["b0"].value, self.p["b1"].value #end0-start0, end1-start1
    
    def OptimalParams(self, method_kwargs):
        self.Solve(self.p)  # solve first syllable
        
        kwargs["Ns"] = 51;   self.OptimalGamma(method_kwargs)
        kwargs["Ns"] = 21;   self.OptimalBs(method_kwargs)
        self.WriteAudio()
    
    # Solve the minimization problem at once
    def CompleteSolution(self, opt_gamma, kwargs):
        start = time.time()
        # add params:   (NAME   VALUE    VARY    MIN  MAX  EXPR BRUTE_STEP)
        self.p.add_many(('a0',   0.11,         True ,   0, 0.25,  None, 0.01), 
                        ('a1',   0.05,         True,   -2,    2,  None, 0.1),  
                        ('b0',   -0.1,         True,   -1,  0.5,  None, 0.03),  
                        ('b1',      1,         True,  0.2,    2,  None, 0.04), 
                        ('gamma', opt_gamma,   False,  1e4,  1e5, None, 1000),
                        ('b2',     0.,         False, None, None, None, None), 
                        ('a2',     0.,         False, None, None, None, None))
        mi    = lmfit.minimize(self.residualFFandSCI, self.p, nan_policy='omit', **kwargs) 
        self.p["a0"].set(   vary=False, value=mi.params["a0"].value)
        self.p["a1"].set(   vary=False, value=mi.params["a1"].value)
        self.p["b0"].set(   vary=False, value=mi.params["b0"].value)
        self.p["b1"].set(   vary=False, value=mi.params["b1"].value)
        self.p["gamma"].set(vary=False, value=mi.params["gamma"].value)
        
        self.Solve(self.p)
        end = time.time()
        
        print("Time of execution = {0:.4f}".format(end-start))