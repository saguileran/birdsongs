from .util import *
from pathlib import Path

class Syllable(object):
    #%%
    """
    Store and define a syllable and its properties
    INPUT:
        s  = signal
        fs = sampling rate
        t0 = initial time of the syllable
    """ 
    def __getstate__(self):
        state = self.__dict__.copy()
        #del state[''] # remove the unpicklable progress attribute
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        # restore the progress from the progress integer
        #self.progress = make_progress(self.progress_int)
    
    #%%
    def __init__(self, birdsong=None, t0=0, Nt=200, llambda=1.5, NN=0, overlap=0.5, flim=(1.5e3,2e4), n_mels=4, umbral_FF=1, tlim=[], sfs=[], no_syllable=0, ide="", file_name="syllable"):
        ## The bifurcation can be cahge modifying the self.f2 and self.f1 functions
        ## ------------- Bogdanov–Takens bifurcation ------------------
        self.beta_bif = np.linspace(-2.5, 1/3, 1000)  # mu2:beta,  mu1:alpha
        xs, ys, alpha, beta, gamma = sym.symbols('x y alpha beta gamma')
        # ---------------- Labia EDO's Bifurcation -----------------------
        self.f1 = ys
        self.f2 = (-alpha-beta*xs-xs**3+xs**2)*gamma**2 -(xs+1)*gamma*xs*ys
        x01 = sym.solveset(self.f1, ys)+sym.solveset(self.f1, xs)  # find root f1
        f2_x01 = self.f2.subs(ys,x01.args[0])                     # f2(root f1)
        f  = sym.solveset(f2_x01, alpha)                         # root f2 at root f1, alpha=f(x,beta)
        g  = alpha                                               # g(x) = alpha, above
        df = f.args[0].diff(xs)                                   # f'(x)
        dg = g.diff(xs)                                           # g'(x)
        roots_bif = sym.solveset(df-dg, xs)                       # bifurcation roots sets (xmin, xmas)
        self.mu1_curves = [] 
        for ff in roots_bif.args:                                       # roots as arguments (expr)
            x_root = np.array([float(ff.subs(beta, mu2)) for mu2 in self.beta_bif], dtype=float)    # root evaluatings beta
            mu1    = np.array([f.subs([(beta,self.beta_bif[i]),(xs,x_root[i])]).args[0] for i in range(len(self.beta_bif))], dtype=float)
            self.mu1_curves.append(mu1)
        self.f1 = sym.lambdify([xs, ys, alpha, beta, gamma], self.f1)
        self.f2 = sym.lambdify([xs, ys, alpha, beta, gamma], self.f2)
        ## -------------------------------------------------------------------------------------
        self.p = lmfit.Parameters()
        # add params:   (NAME   VALUE    VARY    MIN  MAX  EXPR BRUTE_STEP)
        self.p.add_many(('a0', 0.11, False, 0.01, 0.25, None, None), 
                        ('a1', 0.05, False,   -2,    2, None, None),
                        ('a2',   0., False,    0,    2, None, None),
                        ('b0', -0.1, False,   -1,  0.5, None, None),  
                        ('b1',    1, False,    0,    2, None, None), 
                        ('b2',   0., False,    0,    2, None, None), 
                        ('gm',  4e4, False,  1e4,  1e5, None, None))
        # -------------------------------------------------------------------        
        self.n_mfcc     = 8
        self.Nt         = Nt
        
        self.n_mels     = n_mels
        self.flim       = flim
        self.llambda    = llambda
        self.umbral_FF  = umbral_FF
        
        # define a syllable by entering the amplitude array (out)
        if birdsong!=None: 
            self.birdsong   = birdsong
            self.fs         = self.birdsong.fs
            self.center     = self.birdsong.center
            self.no_file    = self.birdsong.no_file
            self.paths      = self.birdsong.paths
            self.file_name  = self.birdsong.file_name
            self.umbral     = self.birdsong.umbral
            s = self.birdsong.s; 
            NN = self.birdsong.NN
        elif len(sfs)!=0:           
            s, fs           = sfs
            self.fs         = fs
            self.center     = False
            self.no_file    = 0
            self.file_name  = file_name
            self.umbral     = 0.05
        
        
        # ------ define syllable by time interval [tini, tend] --------
        if len(tlim)==0 and t0!=0: 
            self.s  = sound.normalize(s, max_amp=1.0)
            self.t0 = t0
        elif len(tlim)==0 and t0==0: 
            self.t0 = 0
            self.s  = sound.normalize(s, max_amp=1.0)
        elif len(tlim)!=0:
            self.s  = sound.normalize(s[int(tlim[0]*self.fs):int(tlim[1]*self.fs)], max_amp=1.0)
            self.t0 = tlim[0]
            self.tlim = tlim
        
        self.time_s   = np.linspace(0, len(self.s)/self.fs, len(self.s))
        self.envelope = Enve(self.s, self.fs, self.Nt)
        self.T        = self.s.size/self.fs
        self.time0    = np.linspace(0, len(self.s)/self.fs, len(self.s))
        
        if NN==0:
            if self.s.size < self.fs/5: # a decimal of a second (0.2 s)
                self.id          = "chunck"
                self.NN          = 128
            else:
                self.id          = "syllable"
                self.NN          = 1024
        else: self.NN=NN
        if ide!="": self.id = ide
        self.no_syllable = no_syllable
        self.win_length  = self.NN
        self.hop_length  = self.NN//4
        self.no_overlap  = int(overlap*self.NN)        
        
        # -------------------------------------------------------------------
        # ------------- ACOUSTIC FEATURES -----------------------------------
        # -------------------------------------------------------------------
        self.stft = librosa.stft(y=self.s, n_fft=self.NN, hop_length=self.hop_length, win_length=self.NN, window='hann',
                                 center=self.center, dtype=None, pad_mode='constant')
        freqs, times, mags = librosa.reassigned_spectrogram(self.s, sr=self.fs, S=self.stft, n_fft=self.NN,
                                        hop_length=self.hop_length, win_length=self.win_length, window='hann', 
                                        center=self.center, reassign_frequencies=True, reassign_times=True,
                                        ref_power=1e-06, fill_nan=True, clip=True, dtype=None, pad_mode='constant')
        
        self.freqs   = freqs  
        self.times   = times 
        self.Sxx     = mags 
        self.Sxx_dB  = librosa.amplitude_to_db(mags, ref=np.max)
        self.FF_coef = np.abs(self.stft)
        self.freq = librosa.fft_frequencies(sr=self.fs, n_fft=self.NN) 
        self.time = librosa.times_like(X=self.stft,sr=self.fs, hop_length=self.hop_length, n_fft=self.NN) #, axis=-1
        self.time -= self.time[0]
        
        self.f_msf   = np.array([Norm(self.FF_coef[:,i]*self.freq, 1)/Norm(self.FF_coef[:,i], 1) for i in range(self.FF_coef.shape[1])])        
        
        self.centroid =  feature.spectral_centroid(y=self.s, sr=self.fs, S=np.abs(self.stft), n_fft=self.NN,
                                            hop_length=self.hop_length, freq=self.freqs, win_length=self.win_length, 
                                            window='hann',center=self.center, pad_mode='constant')[0]
        self.mfccs = feature.mfcc(y=self.s, sr=self.fs, S=self.stft, n_mfcc=self.n_mfcc, dct_type=2, norm='ortho', lifter=0)
        self.rms   = feature.rms(y=self.s, S=self.stft, frame_length=self.NN, hop_length=self.hop_length,
                                 center=self.center, pad_mode='constant')[0]
        self.s_mel = feature.melspectrogram(y=self.fs, sr=self.fs, S=self.stft, n_fft=self.NN, hop_length=self.hop_length,
                                            win_length=self.win_length, window='hann', center=self.center, pad_mode='constant', power=2.0)
       
        # pitches[..., f, t] contains instantaneous frequency at bin f, time t
        # magnitudes[..., f, t] contains the corresponding magnitudes.
        # Both pitches and magnitudes take value 0 at bins of non-maximal magnitude.
        # pitches, magnitudes = librosa.piptrack(y=self.s, sr=self.fs, S=self.stft, n_fft=self.NN, hop_length=self.hop_length,
        #                        fmin=self.flim[0], fmax=self.flim[1], threshold=0.01, win_length=self.win_length, 
        #                        center=self.center, pad_mode='constant', ref=None)
        # self.zcr = librosa.feature.zero_crossing_rate(y=self.s,frame_length=self.NN, hop_length=self.hop_length, center=self.center)
        # self.rolloff     = feature.spectral_rolloff(y=self.s, sr=self.fs, S=self.stft, n_fft=self.NN, 
        #                                             hop_length=self.hop_length, win_length=self.win_length, center=self.center,
        #                                             pad_mode='constant', freq=self.freqs, roll_percent=0.6)[0]
        # self.rolloff_min = feature.spectral_rolloff(y=self.s, sr=self.fs, S=self.stft, n_fft=self.NN, 
        #                                          hop_length=self.hop_length, win_length=self.win_length, 
        #                          center=self.center, pad_mode='constant', freq=self.freqs, roll_percent=0.2)[0]
        # self.onset_env   = onset.onset_strength(y=self.s, sr=self.fs, S=self.stft, lag=1, max_size=1, fmax=self.flim[1],
        #                                          ref=None, detrend=False, center=self.center, feature=None, aggregate=None)

#         # ----------------------- matrix  ----------------------------
#         self.contrast = feature.spectral_contrast(y=self.s, sr=self.fs, S=self.stft, n_fft=self.NN, hop_length=self.hop_length, 
#                                     win_length=self.win_length, center=self.center, pad_mode='constant', freq=self.freqs, 
#                                     fmin=self.flim[0], n_bands=4,quantile=0.02, linear=False)
#         self.s_mel    = feature.melspectrogram(y=self.s, sr=self.fs, S=self.stft, n_fft=self.NN, 
#                                        hop_length=self.hop_length, win_length=self.win_length, center=self.center, 
#                                        pad_mode='constant', power=2.0, n_mels=self.n_mels,
#                                        fmin=self.flim[0], fmax=self.flim[1])
#         self.s_sal    = librosa.salience(S=self.FF_coef, freqs=self.freqs, harmonics=[1, 2, 3, 4], weights=[1,1,1,1], 
#                                          aggregate=None, filter_peaks=True, fill_value=0, kind='linear', axis=-2)
#         self.C        = librosa.cqt(y=self.s, sr=self.fs, hop_length=self.hop_length, fmin=self.flim[0], n_bins=32, 
#                                     bins_per_octave=12, tuning=0.0, filter_scale=1, norm=1, sparsity=0.01, 
#                                     window='hann', scale=True, pad_mode='constant', dtype=None)
#         self.D        = librosa.iirt(y=self.s, sr=self.fs, win_length=self.win_length, hop_length=self.hop_length, center=self.center, 
#                          tuning=0.0, pad_mode='constant', flayout='sos')
        
#         self.pitches    = pitches
#         self.magnitudes = magnitudes
        
        # # self.features  = [energy, Ht, Hf]
        # # self.entropies = [EAS, ECU, ECV, EPS, EPS_KURT, EPS_SKEW]
        # # self.times_on = times_on
        
        # # ------------- "better method" --------------
        # self.FF     = pyin(self.s, fmin=self.flim[0], fmax=self.flim[1], sr=self.fs, frame_length=self.NN, 
        #                win_length=self.win_length, hop_length=self.hop_length, n_thresholds=100, beta_parameters=(2, 18), 
        #                boltzmann_parameter=2, resolution=0.1, max_transition_rate=35.92, switch_prob=0.01, 
        #                no_trough_prob=0.01, fill_na=0, center=self.center, pad_mode='constant')
        self.FF     = yin(self.s, fmin=self.flim[0], fmax=self.flim[1], sr=self.fs, frame_length=self.NN, 
                          win_length=self.NN//2, hop_length=self.NN//4, trough_threshold=self.umbral_FF, center=self.center, pad_mode='constant')
        
#         # # remove atypical data
#         df = pd.DataFrame(data={"FF":self.FF, "time":self.time})
#         q  = df["FF"].quantile(0.99)
#         df[df["FF"] < q]
#         q_low, q_hi = df["FF"].quantile(0.1), df["FF"].quantile(0.99)
#         df_filtered = df[(df["FF"] < q_hi) & (df["FF"] > q_low)]

#         self.time   = self.time[df_filtered["FF"].index]
#         self.FF = self.FF[df_filtered["FF"].index]

        
        
        
        self.timeFF = np.linspace(0,self.time[-1],self.FF.size)
        self.FF_fun = interp1d(self.timeFF, self.FF)
        self.SCI    = self.f_msf / self.FF_fun(self.time)
    
    #%%
    def AlphaBeta(self):
        a = np.array([self.p["a0"].value, self.p["a1"].value, self.p["a2"].value]);   
        b = np.array([self.p["b0"].value, self.p["b1"].value, self.p["b2"].value])
        
        t_1   = np.linspace(0,self.T,len(self.s))   
        t_par = np.array([np.ones(t_1.size), t_1, t_1**2])
        
        self.alpha = np.dot(a, t_par);  # lines (or parabolas)
        
        # define by same shape as fudamenta frequency
        if "syllable" in self.id: 
            poly = Polynomial.fit(self.timeFF, self.FF, deg=10)
            x, y = poly.linspace(np.size(self.s))
            self.beta  = b[0] + b[1]*(1e-4*y) + b[2]*(1e-4*y)**2   
        elif "chunck" in self.id: 
            self.beta = np.dot(b, t_par);
            
    #%%
    ##  ------------------------- BIRDS -------------------------
    def MotorGestures(self, alpha, beta, gamma, ovfs=20, prct_noise=0):  # ovfs:oversamp
        t, tmax, dt = 0, int(self.s.size)*ovfs-1, 1./(ovfs*self.fs) # t0, tmax, td
        # pback and pin vectors initialization
        pi, pb, out = np.zeros(tmax), np.zeros(tmax), np.zeros(int(self.s.size))
        # initial vector ODEs (v0), it is not too relevant
        v = 1e-4*np.array([1e2, 1e1, 1, 1, 1, 1]);  self.Vs = [v];
        # ------------- BIRD PARAMETERS -----------
        BirdData = pd.read_csv(self.paths.auxdata/'ZonotrichiaData.csv')
        c, L, r, Ch, MG, MB, RB, Rh = BirdData['value'] # c, L, r, c, L1, L2, r2, rd 
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
        # ------------------------------ ODEs -----------------------------
        def ODEs(v):
            dv, [x, y, pout, i1, i2, i3] = np.zeros(6), v  # (x, y, pout, i1, i2, i3)'
            # ----------------- direct implementation of the EDOs -----------
            # dv[0] = y
            # dv[1] = (-self.alpha[t//ovfs]-self.beta[t//ovfs]*x-x**3+x**2)*self.p["gm"].value**2 - (x**2*y+x*y)*self.p["gm"].value
            dv[0] = self.f1(x, y, alpha[t//ovfs], beta[t//ovfs], gamma)
            dv[1] = self.f2(x, y, alpha[t//ovfs], beta[t//ovfs], gamma)
            # ------------------------- trachea ------------------------
            pbold = pb[t]                                 # pressure back before
            # Pin(t) = Ay(t)+pback(t-L/C) = envelope_Signal*v[1]+pb[t-L/C/dt]
            pi[t] = (.5*self.envelope[t//ovfs])*dv[1] + pb[t-int(L/c/dt)] 
            pb[t] = -r*pi[t-int(L/c/dt)]                          # pressure back after: -rPin(t-L/C) 
            pout  = (1-r)*pi[t-int(L/c/dt)]                       # pout
            # ---------------------------------------------------------------
            dv[2] = (pb[t]-pbold)/dt                      # dpout
            dv[3] = i2
            dv[4] = -(1/Ch/MG)*i1 - Rh*(1/MB+1/MG)*i2 +(1/MG/Ch+Rh*RB/MG/MB)*i3 \
                    +(1/MG)*dv[2] + (Rh*RB/MG/MB)*pout
            dv[5] = -(MG/MB)*i2 - (Rh/MB)*i3 + (1/MB)*pout
            return dv        
        # ----------------------- Solving EDOs ----------------------
        while t < tmax: # and v[1] > -5e6:  # labia velocity not too fast
            v = rk4(ODEs, v, dt);  self.Vs.append(v)  # RK4 - step
            out[t//ovfs] = RB*v[-1]               # output signal (synthetic) 
            t += 1;
            
            # # if the bird OEC change of size in time
            # BirdData = pd.read_csv(self.paths.auxdata/'ZonotrichiaData.csv')
            # c, L, r, Ch, MG, MB, RB, Rh = BirdData['value'] # c, L, r, c, L1, L2, r2, rd 
        
        # ------------------------------------------------------------
        self.Vs = np.array(self.Vs)
        # define solution (synthetic syllable) as a Syllable object 
        synth = Syllable(Nt=self.Nt, llambda=self.llambda, NN=self.NN, overlap=0.5, flim=self.flim, sfs=[out, self.fs])
        
        
        # synth.no_syllable = self.no_syllable
        # synth.no_file     = self.no_file
        # synth.p           = self.p
        # synth.paths       = self.paths
        synth.id          = self.id+"-synth"
        synth.Vs          = self.Vs
        synth.alpha       = self.alpha
        synth.beta        = self.beta
        
        df_MotorGestures = pd.DataFrame(data={"time":self.time_s, "alpha":self.alpha, "beta":self.beta})
        name = self.file_name[:-4] + "-MG.csv"
        df_MotorGestures.to_csv(self.paths.results / Path("MotorGestures-parameters") / name, index=False)

        synth.timesVs     = np.linspace(0, len(self.s)/self.fs, len(self.s)*ovfs)
        
        delattr(self,"alpha"); delattr(self,"beta")
        
        return synth

    #%%
    ##  --------------- Anphibius -------------------------
    def MotorGesturesAnphibius(self, alpha, beta, gamma, ovfs=20, prct_noise=0):  # ovfs:oversamp
        t, tmax, dt = 0, int(self.s.size)*ovfs-1, 1./(ovfs*self.fs) # t0, tmax, td
        # pback and pin vectors initialization
        pi, pb, out = np.zeros(tmax), np.zeros(tmax), np.zeros(int(self.s.size))
        # initial vector ODEs (v0), it is not too relevant
        v = 1e-4*np.array([1e2, 1e1, 1]);  self.Vs = [v];
        # ------------- BIRD PARAMETERS -----------
        c, L, r = 3.43E+02, 2.50E-02, 6.50E-01 
        # , Ch, MG, MB, RB, Rh = BirdData['value'] # c, L, r, c, L1, L2, r2, rd 
        # BirdData = pd.read_csv(self.paths.auxdata/'ZonotrichiaData.csv')
        # - Trachea:
        #           r: reflection coeficient    [adimensionelss]
        #           L: trachea length           [m]
        #           c: speed of sound in media  [m/s]
        def ODEs(v):
            dv, [x, y, pout] = np.zeros(3), v  # (x, y, pout)'
            # ----------------- direct implementation of the EDOs -----------
            # dv[0] = y
            # dv[1] = (-self.alpha[t//ovfs]-self.beta[t//ovfs]*x-x**3+x**2)*self.p["gm"].value**2 - (x**2*y+x*y)*self.p["gm"].value
            dv[0] = self.f1(x, y, alpha[t//ovfs], beta[t//ovfs], gamma)
            dv[1] = self.f2(x, y, alpha[t//ovfs], beta[t//ovfs], gamma)
            # ------------------------- trachea ------------------------
            pbold = pb[t]                                 # pressure back before
            # Pin(t) = Ay(t)+pback(t-L/C) = envelope_Signal*v[1]+pb[t-L/C/dt]
            pi[t] = (.5*self.envelope[t//ovfs])*dv[1] + pb[t-int(L/c/dt)] 
            pb[t] = -r*pi[t-int(L/c/dt)]                          # pressure back after: -rPin(t-L/C) 
            pout  = (1-r)*pi[t-int(L/c/dt)]                       # pout
            # ---------------------------------------------------------------
            dv[2] = (pb[t]-pbold)/dt                      # dpout
            return dv        
        # ----------------------- Solving EDOs ----------------------
        while t < tmax: # and v[1] > -5e6:  # labia velocity not too fast
            v = rk4(ODEs, v, dt);  self.Vs.append(v)  # RK4 - step
            out[t//ovfs] = v[-1]               # output signal (synthetic) 
            t += 1;
        # ------------------------------------------------------------
        self.Vs = np.array(self.Vs)
        # define solution (synthetic syllable) as a Syllable object 
        synth = Syllable(Nt=self.Nt, llambda=self.llambda, NN=self.NN, overlap=0.5, flim=self.flim, sfs=[out, self.fs])
        
        synth.id          = self.id+"-synth"
        synth.Vs          = self.Vs
        synth.alpha       = self.alpha
        synth.beta        = self.beta
        synth.timesVs     = np.linspace(0, len(self.s)/self.fs, len(self.s)*ovfs)
        
        delattr(self,"alpha"); delattr(self,"beta");
        
        return synth

        
    # def Enve(self, out, fs, Nt):
    #     time = np.linspace(0, len(out)/fs, len(out))
    #     out_env = sound.envelope(out, Nt=Nt) 
    #     t_env = np.arange(0,len(out_env),1)*len(out)/fs/len(out_env)
    #     t_env[-1] = time[-1] 
    #     fun_s = interp1d(t_env, out_env)
    #     return fun_s(time)

    #%%    
    def WriteAudio(self):
        name = '{}/{}-{}-{}.wav'.format(self.paths.examples, self.file_name[:-4], self.id, self.no_syllable)
        WriteAudio(name, fs=self.fs, s=self.s)

    #%%    
    def Solve(self, p, orde=2):
        self.p = p;  # define parameteres to optimize
        self.ord = orde; 
        
        if self.s.size < 2*self.fs/100: self.id = "chunck"
        else:                           self.id = "syllable"
                
        self.AlphaBeta()             # define alpha and beta parameters
        synth = self.MotorGestures(self.alpha, self.beta, self.p["gm"].value) # solve the problem and define the synthetic syllable
        synth = self.SynthScores(synth, orde=orde) # compute differences and score variables
        synth.paths = self.paths
        synth.file_name = self.file_name[:-4]# + "-synth    "
        return synth
    
    #%%
    def SolveAB(self, alpha, beta, gamma, orde=2):
        self.alpha = alpha; self.beta  = beta;
        
        synth = self.MotorGestures(alpha, beta, gamma)
        synth = self.SynthScores(synth, orde=orde)
        synth.id = "synth-birdsongs"
        
        return synth
    
    #%%
    def Play(self): playsound(self.file_name)
    
    #%%
    def SynthScores(self, synth, orde=2):
        synth.ord=self.ord=orde;  # order of score norms
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
        synth.correlation = np.zeros_like(self.time)
        synth.Df          = np.zeros_like(self.time)
        synth.SKL         = np.zeros_like(self.time)
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
        
        synth.scoreCorrelation = Norm(synth.correlation, ord=self.ord)/synth.correlation.size
        synth.scoreSKL         = Norm(synth.SKL, ord=self.ord)/synth.SKL.size
        synth.scoreDF          = Norm(synth.Df, ord=self.ord)/synth.Df.size
        
        
        synth.residualCorrelation = synth.scoreFF-np.mean(synth.correlation+synth.Df +synth.scoreSKL)
        
        synth.SCIFF = synth.scoreSCI + synth.scoreFF
        
        return synth

    #%%
    def Set(self, p_array):
        self.p["a0"].set(value=p_array[0])
        self.p["a1"].set(value=p_array[1])
        self.p["a2"].set(value=p_array[2])
        self.p["b0"].set(value=p_array[3])
        self.p["b1"].set(value=p_array[4])
        self.p["b2"].set(value=p_array[4])