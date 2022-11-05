from .functions import *
from .paths import *

class Syllable(object):
    """
    Store and define a syllable and its properties
    INPUT:
        s  = signal
        fs = sampling rate
        t0 = initial time of the syllable
    """
    def __init__(self, s, fs, t0, Nt=200, llambda=1.5, NN=512, overlap=0.5, flim=(1.5e3,2e4)):
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
        self.center = False
        # ----------------------- scalar features
        energy =  Norm(self.s,ord=2)/self.s.size
        Ht     = features.temporal_entropy (self.s, compatibility='seewave', mode='fast', Nt=self.NN)
        Hf, _  = features.frequency_entropy(Sxx_power, compatibility='seewave')
        #EAS, ECU, ECV, EPS, EPS_KURT, EPS_SKEW = features.spectral_entropy(Sxx_power, fn, flim=self.flim) 
        
        # ----------------------- vector features
        self.freq    = fft_frequencies(sr=self.fs, n_fft=self.NN) 
        self.FF_coef = np.abs(stft(y=self.s, n_fft=self.NN, hop_length=self.NN//2, win_length=None, 
                     center=True, dtype=None, pad_mode='constant'))
        self.FF_coef /= np.max(self.FF_coef)
        self.FF_time = np.linspace(0,self.time[-1], self.FF_coef.shape[1]) #times_like(self.FF_coef, sr=self.fs, hop_length=self.NN//2, n_fft=self.NN)
        
        centroid      = feature.spectral_centroid(y=self.s, sr=self.fs, S=self.Sxx_dB, n_fft=self.NN, 
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
    
    def MotorGestures(self, sampling=44100, oversamp=20, prct_noise=0):
        N_total = len(self.s)

        #sampling and necessary constants
        sampling, oversamp = sampling, oversamp
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
                out[n_out]   = RB*v[4]*10   # out =  
                n_out       += 1
                self.Vs.append(v);
                
                beta      = self.beta[n_out]     + prct_noise*noise
                amplitud  = self.envelope[n_out] + prct_noise*noise
                alpha     = self.alpha[n_out]
                taux      = 0
            t += 1;   taux += 1;
        self.Vs = np.array(self.Vs)

        # pre processing synthetic data
        out = sound.normalize(out, max_amp=1)
        self.synth_env = self.Enve(out)
        self.out_amp           = np.zeros_like(out)
        not_zero               = np.where(self.synth_env > 0.005)
        self.out_amp[not_zero] = out[not_zero] * self.envelope[not_zero] / self.synth_env[not_zero]
        
        
        self.synth = Syllable(self.out_amp, self.fs, self.t0,  Nt=self.Nt, llambda=self.llambda, NN=self.NN)
        
    def Enve(self, out):
        synth_env = sound.envelope(out, Nt=self.Nt) 
        t_env = np.arange(0,len(synth_env),1)*len(out)/self.fs/len(synth_env)
        t_env[-1] = self.time[-1] 
        fun_s = interp1d(t_env, synth_env)
        return fun_s(self.time)
        
        
    def WriteAudio(self):
        wavfile.write('{}/synth4_amp_{}_{}.wav'.format(self.paths.examples,self.no_syllable), self.fs, np.asarray(self.out_amp,  dtype=np.float32))
        wavfile.write('{}/song_{}_{}.wav'.format(self.paths.examples,self.no_syllable),       self.fs, np.asarray(self.s,     dtype=np.float32))
    
    # -------------- --------------
    def Solve(self, p):
        self.p = p
        self.AlphaBeta()
        self.MotorGestures() # solve the problem and define the synthetic syllable
        
        #deltaFF     = 
        #deltaSCI    = np.abs(self.synth.SCI-self.SCI)
        #deltaNoHarm = np.abs(self.synth.NoHarm_out-self.NoHarm).astype(float)
        deltaSxx    = np.abs(self.synth.Sxx-self.Sxx)
        deltaMel    = np.abs(self.synth.FF_coef-self.FF_coef)
        
        self.deltaEnv   = np.abs(self.synth.envelope-self.envelope)
        self.deltaFF     = 1e-4*np.abs(self.synth.FF-self.FF) #/np.max(deltaFF)
        #self.deltaSCI    = deltaSCI     #/np.max(deltaSCI)
        #self.DeltaNoHarm = deltaNoHarm*10**(deltaNoHarm-2)
        self.deltaSxx    = deltaSxx/np.max(deltaSxx)
        self.deltaMel    = deltaMel/np.max(deltaMel)
            
        #self.scoreSCI    = np.norm(self.deltaSCI)#/self.deltaSCI.size
        self.scoreFF     = Norm(np.abs(self.deltaFF),ord=1)/self.deltaFF.size
        self.scoreEnv    = Norm(self.deltaEnv,ord=1)/self.deltaEnv.size
        self.scoreSxx    = Norm(self.deltaSxx, ord=np.inf)#/self.DeltaSxx.size
        self.scoreMel    = Norm(self.deltaMel, ord=np.inf)#/self.DeltaSxx.size
        
        
    def residualSCI(self, p):
        self.Solve(p)
        return self.scoreSCI
    
    def residualFF(self, p):
        self.Solve(p)
        return self.scoreFF
    
    def residualFFandSCI(self, p):
        self.Solve(p)
        return self.scoreFF#+self.scoreSCI+self.DeltaNoHarm
    
    # ----------- OPTIMIZATION FUNCTIONS --------------
    def OptimalGamma(self, method_kwargs):
        kwargs = {k: method_kwargs[k] for k in set(list(method_kwargs.keys())) - set(["method"])}
    
        start = time.time()
        self.p["gamma"].set(vary=True)
        mi    = lmfit.minimize(self.residualFFandSCI, self.p, nan_policy='omit', method=method_kwargs["method"], **kwargs) 
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
    def CompleteSolution(self, kwargs):
        start = time.time()
        # add params:   (NAME   VALUE    VARY    MIN  MAX  EXPR BRUTE_STEP)
        self.p.add_many(('a0',   0.11,   True ,   0, 0.25,  None, 0.01), 
                        ('a1',   0.05,   True,   -2,    2,  None, 0.1),  
                        ('b0',   -0.1,   True,   -1,  0.5,  None, 0.03),  
                        ('b1',      1,   True,  0.2,    2,  None, 0.04), 
                        ('gamma', 4e4,   True,  1e4,  1e5,  None, 1000),
                        ('b2',     0.,   False, None, None, None, None), 
                        ('a2',     0.,   False, None, None, None, None))
        mi    = lmfit.minimize(self.residualFFandSCI, self.p, nan_policy='omit', **kwargs) 
        self.p["a0"].set(   vary=False, value=mi.params["a0"].value)
        self.p["a1"].set(   vary=False, value=mi.params["a1"].value)
        self.p["b0"].set(   vary=False, value=mi.params["b0"].value)
        self.p["b1"].set(   vary=False, value=mi.params["b1"].value)
        self.p["gamma"].set(vary=False, value=mi.params["gamma"].value)
        
        self.Solve(self.p)
        end = time.time()
        
        print("Time of execution = {0:.4f}".format(end-start))
    
    
    ## ----------------------- PLOT FUNCTIONS --------------------------
    def PlotSynth(self):
        
        fig, ax = plt.subplots(2, 2, figsize=(12, 5), sharex=True, sharey='col')
        fig.subplots_adjust(top=0.85)     
        
        ax[0][0].plot(self.time, self.s, label='canto', c='b')
        ax[0][0].set_title('Real')
        ax[0][0].plot(self.time, self.envelope, label='envelope', c='k')
        ax[0][0].legend(); ax[0][0].set_ylabel("Amplitud (a.u.)")
        ax[1][0].plot(self.synth.time, self.synth.s, label='synthetic', c='g')
        ax[1][0].set_title('Synthetic') 
        ax[1][0].plot(self.synth.time, self.synth.envelope
, label='envelope', c='k')
        ax[1][0].legend(); ax[1][0].set_xlabel('t (s)'); ax[1][0].set_ylabel("Amplitud (a.u.)")

        Delta_tu   = self.tu[-1] - self.tu[0]
        Delta_tu_s = 1#tu_s[-1] - tu_s[0]

        pcm = ax[0][1].pcolormesh(self.tu, self.fu*1e-3, self.Sxx, cmap=plt.get_cmap('Greys'), rasterized=True)#, vmin=10, vmax=70)
        fig.colorbar(pcm, ax=ax[0,1], location='right', label='Power')
        ax[0][1].plot(self.timeFF, self.FF*1e-3, 'bo-', lw=2)
        ax[0][1].set_title('Real'); ax[0][1].set_ylabel('f (khz)'); ax[0][1].set_ylim(0, 20);

        pcm = ax[1][1].pcolormesh(self.synth.tu, self.synth.fu*1e-3, self.synth.Sxx, cmap=plt.get_cmap('Greys'), rasterized=True)#, vmin=10, vmax=70)
        fig.colorbar(pcm, ax=ax[1,1], location='right', label='Power')
        ax[1][1].plot(self.synth.timeFF, self.synth.FF*1e-3, 'go-', lw=2)
        ax[1][1].set_title('Synthetic') 
        ax[1][1].set_ylim(0, 20);   ax[1][1].set_xlim(min(self.synth.time), max(self.synth.time))
        ax[1][1].set_xlabel('t (s)'); ax[1][1].set_ylabel('f (khz)');

        #fig.tight_layout(); 
        fig.suptitle('Sound Waves and Spectrograms', fontsize=20)#, family='fantasy');
        plt.show()
        
    def PlotAlphaBeta(self, xlim=(-0.05,.2), ylim=(-0.2,0.9)):
        fig = plt.figure(constrained_layout=True, figsize=(10, 6))
        gs  = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1:, 0])
        ax3 = fig.add_subplot(gs[:, 1])

        fig.suptitle("GridSpec")
        fig.tight_layout(pad=3.0)        #fig.subplots_adjust(top=0.85)
        
        viridis = cm.get_cmap('Blues')
        c       = viridis(np.linspace(0.3, 1, np.size(self.time)))
        
        ax1.scatter(self.time, self.alpha, c=c, label="alfa")
        ax1.set_title('Air-Sac Pressure')
        ax1.grid()
        ax1.set_ylabel('α (a.u.)'); #ax1.set_xlabel('Time (s)'); 
        ax1.set_ylim(xlim);        #ax[0].legend()

        ax2.scatter(self.time, self.beta, c=c, label="beta")
        ax2.set_title('Labial Tension')
        ax2.grid()
        ax2.set_xlabel(r'Time (s)'); ax2.set_ylabel('β (a.u.)')
        ax2.set_ylim(ylim);         ax2.sharex(ax1);        #ax[0].legend()

        # ------------- Bogdanov–Takens bifurcation ------------------
        mu2 = np.linspace(-2.5, 1/3, 1000)
        xmas = (1+np.sqrt(1-3*mu2))/3
        xmen = (1-np.sqrt(1-3*mu2))/3
        mu1_mas = -mu2*xmas - xmas**3 + xmas**2
        mu1_men = -mu2*xmen - xmen**3 + xmen**2
        #---------------------------------------------------------------

        ax3.scatter(self.alpha, self.beta, c=c, label="Parameters", marker="_")
        ax3.plot(-1/27, 1/3, 'ko')#, label="Cuspid Point"); 
        ax3.axvline(0, color='red', lw=1)#, label="Hopf Bifurcation")
        ax3.plot(mu1_men, mu2, '-g', lw=1)#, label="Saddle-Noddle\nBifurcation"); 
        ax3.text(-0.01,0.6, "Hopf",rotation=90, color="r")
        ax3.text(-0.04,0.39,"CP",  rotation=0,  color="k")
        ax3.text(-0.03,0.2,  "SN",  rotation=0,  color="g")
        ax3.text(0.1, 0.005, "SN",  rotation=0,  color="g")
        
        ax3.plot(mu1_mas, mu2, '-g', lw=1)
        ax3.fill_between(mu1_mas, mu2, 10, where=mu1_mas > 0, color='gray', alpha=0.25)#, label='BirdSongs')
        ax3.set_ylabel('Tension (a.u.)'); ax3.set_xlabel('Pressure (a.u.)')
        ax3.set_title('Parameter Space')
        ax3.legend()
        ax3.set_xlim(xlim); ax3.set_ylim(ylim)
        fig.suptitle("Air-Sac Pressure (α) and Labial Tension (β) Parameters", fontsize=20)#, family='fantasy')
        plt.show()

    def Plot(self, flag=False):       
        fig = plt.figure(constrained_layout=True, figsize=(30, 12))
        gs  = fig.add_gridspec(nrows=4, ncols=5, hspace=0.2)
        
        # ----- FF ---------------
        ax1 = fig.add_subplot(gs[0:2, :3])
        pcm = ax1.pcolormesh(self.tu, self.fu*1e-3, self.Sxx, cmap=plt.get_cmap('Greys'), rasterized=True)#, vmin=10, vmax=70)
        plt.colorbar(pcm, ax=ax1, location='left', label='Power', pad=0.05)
        ax1.plot(self.timeFF, self.FF*1e-3, 'bo-', label='Real',ms=10)
        ax1.plot(self.synth.timeFF, self.synth.FF*1e-3, 'go-', label='Synthetic', ms=6)
        ax1.legend(); ax1.set_ylim((1, 15)); 
        ax1.set_ylim((1, 15)); ax1.set_xlim((self.tu[0], self.tu[-1]))
        ax1.set_ylabel('f (khz)'); #ax1.set_xlabel('time (s)');     
        ax1.set_title('Spectrogram - Fundamental Frequency (FF)')
        
        ax2 = fig.add_subplot(gs[0:2, 3:])
        ax2.plot(self.timeFF, self.deltaFF, "-o", color="k", label='Σ ΔFF= {:.4f}'.format(self.scoreFF)); ax2.set_ylim((0,15e3))
        ax2.set_xlabel('time (s)'); ax2.set_ylabel('f (kHz)'); ax2.legend()
        ax2.set_title('Fundamental Frequency Error (ΔFF)'); ax2.set_ylim((0,1))
        
        # ------------------ spectgrograms
        ax3 = fig.add_subplot(gs[2, 0])
        pcm = ax3.pcolormesh(self.tu, self.fu*1e-3, self.Sxx, cmap=plt.get_cmap('Greys'), rasterized=True)#, vmin=10, vmax=70)
        plt.colorbar(pcm, ax=ax3, location='left', label='Power', pad=0.05)
        ax3.plot(self.timeFF, self.FF*1e-3, 'bo-', label='Real', ms=10)
        ax3.set_ylim((1, 15)); ax3.set_xlim((self.tu[0], self.tu[-1]))
        ax3.set_ylabel('f (khz)');    
        ax3.set_title('Spectrogram Real (FF-R)')
        
        ax4 = fig.add_subplot(gs[3, 0])
        pcm = ax4.pcolormesh(self.synth.tu, self.synth.fu*1e-3, self.synth.Sxx, cmap=plt.get_cmap('Greys'), rasterized=True)#, vmin=10, vmax=70)
        plt.colorbar(pcm, ax=ax4, location='left', label='Power', pad=0.0)
        ax4.plot(self.synth.timeFF, self.synth.FF*1e-3, 'go-', label='synthetic', ms=6)
        ax4.set_xlim((self.tu[0], self.tu[-1])); ax4.set_ylim((1, 15)); 
        ax4.set_ylabel('f (khz)'); ax4.set_xlabel('time (s)');     
        ax4.set_title('Spectrogram Synthetic (FF-S)')
        ax4.sharex(ax3)
        
        # ------------------ Mel spectgrograms
        ax5 = fig.add_subplot(gs[2, 2])
        pcm = ax5.pcolormesh(self.FF_time, self.freq*1e-3, self.FF_coef, rasterized=True) #cmap=plt.get_cmap('Greys'))#, vmin=10, vmax=70)
        plt.colorbar(pcm, ax=ax5, location='left', label='Power', pad=0.0)
        ax5.set_xlim((self.tu[0], self.tu[-1])); ax5.set_ylim((1, 15)); 
        ax5.set_ylabel('f (khz)'); #ax5.set_xlabel('time (s)');     
        ax5.set_title('Mel-Spectrogram Frequency Real (FF-R)')
        #ax5.sharex(ax3)
        
        ax6 = fig.add_subplot(gs[3, 2])
        pcm = ax6.pcolormesh(self.synth.FF_time, self.synth.freq*1e-3, self.synth.FF_coef, rasterized=True)# cmap=plt.get_cmap('Greys'))#, vmin=10, vmax=70)
        fig.colorbar(pcm, ax=ax6, location='left', label='Power', pad=0.0)
        ax6.set_xlim((self.tu[0], self.tu[-1])); ax5.set_ylim((1, 15)); 
        ax6.set_ylabel('f (khz)'); ax6.set_xlabel('time (s)');     
        ax6.set_title('Mel-Spectrogram Frequency Synth (FF-S)')
        ax6.sharex(ax5)
        
        # ------------------ Delta Sxx - Mell
        ax7 = fig.add_subplot(gs[2, 1])
        pcm = ax7.pcolormesh(self.tu, self.fu*1e-3, self.deltaSxx, cmap=plt.get_cmap('Greys'), rasterized=True)#, vmin=0, vmax=1)
        plt.colorbar(pcm, ax=ax7, location='left', label='Power', pad=0.0)
        #ax[1][0].plot(self.timeFF, self.deltaSCI, "-o", color="k", label='Σ R(SCI) = {:.4f}'.format(self.scoreSCI))
        ax7.set_ylabel('f (khz) (s)'); ax7.set_xlabel('t (s) (s)'); 
        ax7.set_ylim((1, 15)); ax7.set_title('Sxx Error (ΔSxx)')
        #ax7.sharex(ax6)
        
        ax8 = fig.add_subplot(gs[3, 1])
        pcm = ax8.pcolormesh(self.FF_time, self.freq*1e-3, self.deltaMel,  rasterized=True)# cmap=plt.get_cmap('Greys'),, vmin=0, vmax=1)
        plt.colorbar(pcm, ax=ax8, location='left', label='Power', pad=0.0)
        ax8.set_ylabel('f (khz) (s)'); ax8.set_xlabel('t (s) (s)'); 
        ax8.set_ylim((1, 15)); ax8.set_title('Mel Normalized Error (ΔMel)')
        
        # ------------------ sound
        ax9 = fig.add_subplot(gs[2, 3])
        ax9.plot(self.time,     self.s,        label='real',     c='b')
        ax9.plot(self.time,     self.envelope, label='real_env', c='k')
        ax9.plot(self.synth.time, self.synth.s,  label='syn_env',  c='g')
        ax9.plot(self.synth.time, self.synth.envelope,label='synth',    c='g')
        ax9.legend(); ax9.set_ylabel("Amplitud (a.u.)")
        ax9.set_title("Sound Waves")
        
        ax10 = fig.add_subplot(gs[3, 3])
        ax10.plot(self.time, self.deltaEnv, 'ko-', label='Σ Δenv = {:.4f}'.format(self.scoreEnv))
        ax10.set_xlabel("t (s)"); ax10.set_ylabel("Amplitud (a.u.)"); 
        ax10.set_title("Envelope Difference (Δ env)"); 
        ax10.set_ylim((0,1)); ax10.legend()
        ax10.sharex(ax9)
        
        # ------------------ SIC
        ax11 = fig.add_subplot(gs[2, 4])
        
        ax12 = fig.add_subplot(gs[3, 4])
        

        
        fig.suptitle("SCORES", fontsize=20)#, family='fantasy')
        plt.show()
        
    
    def PlotVs(self, xlim=(0,0.025)):
        fig, ax = plt.subplots(3, 1, figsize=(12, 9))
        fig.subplots_adjust(wspace=0.35, hspace=0.4)

        time = self.time[:self.Vs.shape[0]]
        
        ax[0].plot(time, self.Vs[:,4], color='b')
        #ax[0].set_xlim((0,1e5))
        ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("$P_{out}$");
        ax[0].set_title("Trachea Output Pressure")

        ax[1].plot(time, self.Vs[:,1], color='g') 
        ax[1].set_xlim(xlim)
        ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("$P_{in}$");
        ax[1].set_title("Trachea Input Pressure")

        ax[2].plot(time, self.Vs[:,0], color='r')
        ax[2].set_xlim(xlim)
        ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("$x(t)$");
        ax[2].set_title("Labial position")
        ax[2].sharex(ax[1])

        fig.suptitle('Labial Parameters (vector $v$)', fontsize=20)
        plt.show()