from .functions import *
from .paths import *

class Syllable(object):
    """
    Store and define a syllable and its properties
    INPUT:
        s  = signal
        fs = sampling rate
        t0 = initial time of the syllable
        window_time = window time lenght to make chuncks
        IL = interval length to calculate the envelope
    """
    def __init__(self, s, fs, t0, window_time=0.005, Nt=200, llambda=1.5, NN=512, overlap=0.5):
        self.t0 = t0
        self.Nt = Nt
        self.s  = sound.normalize(s, max_amp=1.0)
        self.fs = fs
        self.llambda = llambda
        
        #self.params  = self.p.valuesdict()
        self.NN         = NN
        self.no_overlap = int(overlap*self.NN )
        self.time       = np.linspace(0, len(self.s)/self.fs, len(self.s))
        self.window_time   = window_time # 0.005 # 0.01
        
        Sxx_power, tn, fn, ext = sound.spectrogram (self.s, self.fs, nperseg=self.NN, noverlap=self.no_overlap, mode='psd')  
        Sxx_dB = util.power2dB(Sxx_power) + 96
        Sxx_dB_noNoise, noise_profile, _ = sound.remove_background(Sxx_dB, gauss_std=25, gauss_win=50, llambda=self.llambda)

        self.fu  = fn
        self.tu  = np.linspace(0, self.time[-1], Sxx_power.shape[1]) #tn #*self.time[-1]/tn[-1]
        self.Sxx = sound.smooth(Sxx_dB_noNoise, std=0.5)
        self.Sxx_dB = util.power2dB(self.Sxx) +96
        
        S = -35         # Sensbility microphone-35dBV (SM4) / -18dBV (Audiomoth)
        G = 26          # Amplification gain (26dB (SM4 preamplifier))

        ## acoustic indices in spectral domain
        # df_spec_ind, df_spec_ind_per_bin = features.all_spectral_alpha_indices(self.Sxx_dB,self.tu,self.fu, 
        #                             R_compatible = 'seewave', gain = G, sensitivity = S, verbose = False,
        #                             flim_low = [0,2000], flim_mid = [2000,12000], flim_hi  = [12,20000], 
        #                             mask_param1 = 6, mask_param2=0.5, display = False)
        # self.df_spec_ind = df_spec_ind
        

        self.envelope = sound.envelope(self.s, Nt=self.Nt) 
        t_env = np.arange(0,len(self.envelope),1)*len(self.s)/self.fs/len(self.envelope)
        t_env[-1] = self.time[-1] 
        fun_s = interp1d(t_env, self.envelope)
        self.envelope = fun_s(self.time)
        
        
        self.T        = self.tu[-1]-self.tu[0]
        self.fs_new   = int(self.T*self.fs)   # number of points for the silabale/chunck

        #f0 = pyin(self.s, fmin=1000, fmax=15000, sr=self.fs, frame_length=128, win_length=None, hop_length=None, n_thresholds=100, beta_parameters=(2, 18), boltzmann_parameter=2, resolution=0.1, max_transition_rate=35.92, switch_prob=0.01, no_trough_prob=0.01, fill_na=nan, center=True, pad_mode='constant')
        f0 = yin(self.s, fmin=1000, fmax=15000, sr=self.fs, frame_length=128, win_length=None, hop_length=None, trough_threshold=1, center=False, pad_mode='constant')
        self.timeFF = np.linspace(0,self.time[-1],f0.size)
        
        self.FF = f0
        #self.timeFF = timeFF
        
        self.SCI = self.FF#SCI
        self.NoHarm = 1# NoHarm
    
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
                #if (n_out*100/N_total) % 5 == 0:  print('{:.0f}%'.format(100*n_out/N_total)) # dp = 5 (() % dp == 0)
                
            t += 1;   taux += 1;
        #print('100%')

        # pre processing synthetic data
        out = sound.normalize(out, max_amp=1)
        self.synth_env = sound.envelope(out, Nt=self.Nt) 
        t_env = np.arange(0,len(self.synth_env),1)*len(out)/self.fs/len(self.synth_env)
        t_env[-1] = self.time[-1] 
        fun_s = interp1d(t_env, self.synth_env)
        self.synth_env = fun_s(self.time)
        
        
        self.out_amp           = np.zeros_like(out)
        not_zero               = np.where(self.synth_env > 0.005)
        self.out_amp[not_zero] = out[not_zero] * self.envelope[not_zero] / self.synth_env[not_zero]
        
        self.out_amp = sound.normalize(self.out_amp, max_amp=1.0)
        
        self.s_amp_env = sound.envelope(self.out_amp, Nt=self.Nt) 
        ## acoustic indices in spectral domain
        
        t_env = np.arange(0,len(self.s_amp_env),1)*len(self.out_amp)/self.fs/len(self.s_amp_env)
        t_env[-1] = self.time[-1] 
        fun_s = interp1d(t_env, self.s_amp_env)
        self.s_amp_env = fun_s(self.time)
              
        Sxx_power, tn, fn, ext = sound.spectrogram (self.out_amp, self.fs, nperseg=self.NN, noverlap=self.no_overlap, mode='psd')  
        Sxx_dB = util.power2dB(Sxx_power) + 96
        Sxx_dB_noNoise, noise_profile, _ = sound.remove_background(Sxx_dB, 
                gauss_std=25, gauss_win=50, llambda=self.llambda//3)

        self.fu_out  = fn
        self.tu_out  = np.linspace(0, self.time[-1], Sxx_power.shape[1]) #tn#*self.time[-1]/tn[-1]
        self.Sxx_out = sound.smooth(Sxx_dB_noNoise, std=0.5)
        self.Sxx_dB_out = util.power2dB(self.Sxx) +96
        
        S = -35         # Sensbility microphone-35dBV (SM4) / -18dBV (Audiomoth)
        G = 26          # Amplification gain (26dB (SM4 preamplifier))

        
        # df_spec_ind, df_spec_ind_per_bin = features.all_spectral_alpha_indices(self.Sxx_out,self.tu_out,self.fu_out, 
        #                             R_compatible = 'seewave', gain = G, sensitivity = S, verbose = False,
        #                             flim_low = [0,2000], flim_mid = [2000,12000], flim_hi  = [12,20000], 
        #                             mask_param1 = 6, mask_param2=0.5, display = False)
        # self.df_spec_ind = df_spec_ind
        
        
        self.time_out = np.linspace(0, len(self.out_amp)/self.fs, len(self.out_amp))
        self.Vs       = np.array(self.Vs)
        
        f0_out = yin(self.out_amp, fmin=1000, fmax=15000, sr=self.fs, frame_length=128,
            win_length=None, hop_length=None, trough_threshold=1, center=False, pad_mode='constant')
        self.timeFF_out = np.linspace(0,self.time[-1],f0_out.size)
        
        self.FF_out     = f0_out
        self.SCI_out    = self.FF_out#timeFF_out #SCI_out
        self.NoHarm_out = 1#NoHarm_out
        self.time_out   = np.linspace(0, self.tu_out[-1], len(self.out_amp)) 
    
    def WriteAudio(self):
        wavfile.write('{}/synth4_amp_{}_{}.wav'.format(self.paths.examples,self.no_syllable), self.fs, np.asarray(self.out_amp,  dtype=np.float32))
        wavfile.write('{}/song_{}_{}.wav'.format(self.paths.examples,self.no_syllable),       self.fs, np.asarray(self.s,     dtype=np.float32))
    
    # -------------- --------------
    def Solve(self, p):
        self.p = p
        self.AlphaBeta()
        self.MotorGestures()
        
        deltaFF     = np.abs(self.FF_out-self.FF)
        deltaSCI    = np.abs(self.SCI_out-self.SCI)
        deltaNoHarm = np.abs(self.NoHarm_out-self.NoHarm).astype(float)
        
        self.deltaSCI    = deltaSCI     #/np.max(deltaSCI)
        self.deltaFF     = 1e-4*deltaFF #/np.max(deltaFF)
        self.scoreSCI    = np.sum(self.deltaSCI)#/self.deltaSCI.size
        self.scoreFF     = np.sum(abs(self.deltaFF))/self.deltaFF.size
        self.DeltaNoHarm = deltaNoHarm*10**(deltaNoHarm-2)
    
    def residualSCI(self, p):
        self.Solve(p)
        return self.scoreSCI
    
    def residualFF(self, p):
        self.Solve(p)
        return self.scoreFF
    
    def residualFFandSCI(self, p):
        self.Solve(p)
        return self.scoreFF+self.scoreSCI+self.DeltaNoHarm
    
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
        ax[1][0].plot(self.time_out, self.out_amp, label='synthetic', c='g')
        ax[1][0].set_title('Synthetic') 
        ax[1][0].plot(self.time_out, self.s_amp_env
, label='envelope', c='k')
        ax[1][0].legend(); ax[1][0].set_xlabel('t (s)'); ax[1][0].set_ylabel("Amplitud (a.u.)")

        Delta_tu   = self.tu[-1] - self.tu[0]
        Delta_tu_s = 1#tu_s[-1] - tu_s[0]

        ax[0][1].pcolormesh(self.tu, self.fu*1e-3, self.Sxx, cmap=plt.get_cmap('Greys'), rasterized=True)
        ax[0][1].plot(self.timeFF, self.FF*1e-3, 'bo-', label='Smoothed and Interpolated\nto {0} fs'.format(self.fs), lw=2)
        ax[0][1].set_title('Real'); ax[0][1].set_ylabel('f (khz)'); ax[0][1].set_ylim(0, 20);

        ax[1][1].pcolormesh(self.tu_out, self.fu_out*1e-3, self.Sxx_out, cmap=plt.get_cmap('Greys'), rasterized=True, label='output normalizado amplitud')
        ax[1][1].plot(self.timeFF_out, self.FF_out*1e-3, 'go-', label='Smoothed and Interpolated\nto {0} fs'.format(self.fs), lw=2)
        ax[1][1].set_title('Synthetic') 
        ax[1][1].set_ylim(0, 20);   ax[1][1].set_xlim(min(self.time_out), max(self.time_out))
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

    def Plot(self, flag=0):
        fig, ax = plt.subplots(2, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [3, 2]},sharex=True)
        fig.subplots_adjust(top=0.85);   #fig.tight_layout(pad=3.0)
        
        ax[0][0].pcolormesh(self.tu, self.fu*1e-3, self.Sxx, cmap=plt.get_cmap('Greys'), rasterized=True)
        ax[0][0].plot(self.timeFF, self.FF*1e-3, 'bo-', label='Real',ms=10)
        ax[0][0].plot(self.timeFF_out, self.FF_out*1e-3, 'go-', label='Synthetic', ms=6)
        #ax[0][0].plot(self.time_ampl, self.freq_amp*1e-3, 'r+', label='sampled ff', ms=5)
        ax[0][0].legend()#title="Fundamenta Frequency")
        ax[0][0].set_ylim((1, 15)); ax[0][0].set_xlim((self.tu[0], self.tu[-1]))
        ax[0][0].set_ylabel('f (khz)'); #ax[0][0].set_xlabel('time (s)');     
        ax[0][0].set_title('Fundamental Frequency (FF)')

        ax[0][1].plot(self.timeFF, self.SCI, 'o', ms=3, label='Real mean {0:.2f}'.format(np.mean(self.SCI)))
        ax[0][1].plot(self.timeFF_out, self.SCI_out, 'o', ms=3, label='Synth mean {0:.2f}'.format(np.mean(self.SCI_out)))
        if flag: 
            m,  b  = np.polyfit(self.timeFF, self.SCI, 1)
            mo, bo = np.polyfit(self.timeFF_out, self.SCI_out, 1)
            ax[0][1].plot(self.timeFF, m*self.timeFF+b, 'b-', label='m={0:.2f}, b={1:.2f} '.format(m,b))
            ax[0][1].plot(self.timeFF, mo*self.timeFF+bo, 'r-', label='m={0:.2f}, b={1:.2f} '.format(mo,bo))
        ax[0][1].legend()
        ax[0][1].set_ylabel('SCI (adimensionless)'); #ax[0][1].set_xlabel('time (s)'); 
        ax[0][1].set_title('Spectral Content Index (SCI)')
        
        ax[1][0].plot(self.timeFF, self.deltaFF, "-o", color="k", label='Σ R(FF)/len(s)= {:.4f}'.format(self.scoreFF))
        ax[1][0].set_xlabel('time (s)'); ax[1][0].set_ylabel('Error (\\times 10^{4}$Hz)'); ax[1][0].legend()
        ax[1][0].set_title('Fundamental Frequency Error (ΔFF)'); ax[1][0].set_ylim((0,1))
        
        ax[1][1].plot(self.timeFF, self.deltaSCI, "-o", color="k", label='Σ R(SCI)/len(s) = {:.4f}'.format(self.scoreSCI))
        ax[1][1].set_xlabel('time (s)'); ax[1][1].set_ylabel('Error (adimensionless)'); ax[1][1].legend(); ax[1][1].set_ylim((0,1))
        ax[1][1].set_title('Spectral Content Index Error (ΔSCI)')
        
        fig.suptitle("Scored Variables", fontsize=20)#, family='fantasy')
        plt.show()
    
    def PlotVs(self, xlim=(0,0.025)):
        fig, ax = plt.subplots(3, 1, figsize=(12, 9))
        fig.subplots_adjust(wspace=0.4, hspace=0.4)

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
