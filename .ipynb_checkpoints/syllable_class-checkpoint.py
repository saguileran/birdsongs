from functions import *
from paths import *

class Syllable:    
    """
    Store and define a syllable and its properties
    INPUT:
        s  = signal
        fs = sampling rate
        p  = lmfit parameters object (defines who is gonna be calculated)
        window_time = window time lenght to make chuncks
    """
    def __init__(self, s, fs, t0, p, window_time=0.005, IL=0.01):
        self.t0 = t0
        self.s  = s
        self.fs = fs
        self.p  = p
        
        #self.params  = self.p.valuesdict()
        self.NN      = 256 
        self.sigma   = self.NN/10
        self.overlap = 1/1.1
        
        self.window_time   = window_time # 0.005#0.01
        
        sil_filtered = butter_lowpass_filter(self.s, self.fs, lcutoff=15000.0, order=6)
        self.s = butter_highpass_filter(sil_filtered, self.fs, hcutoff=2000.0, order=5)
        
        fu_sil, tu_sil, Sxx_sil = get_spectrogram(self.s, self.fs, window=self.NN, overlap=self.overlap, sigma=self.sigma) #espectro silaba
        
        self.envelope = normalizar(envelope_cabeza(self.s,intervalLength=IL*np.size(self.s)), minout=0)
        self.time     = np.linspace(0, len(self.s)/self.fs, len(self.s))
        self.fu       = fu_sil
        self.tu       = tu_sil-tu_sil[0]#+self.t0-self.window_time/2#+self.t0-self.window_time/2
        self.Sxx      = Sxx_sil
        self.T        = self.tu[-1]-self.tu[0]
        self.fs_new   = int(self.T*self.fs)   # number of points for the silabale/chunck
        
        SCI, time_ampl, freq_amp, Ampl_freq, freq_amp_int , time_inter = FFandSCI(self.s, self.time, self.fs, self.t0, window_time=self.window_time)#, method="synth")
        
        self.time_ampl          = time_ampl-time_ampl[0]
        self.time_inter         = time_inter-time_inter[0]
        self.Ampl_freq_filtered = Ampl_freq     #_filtered
        self.freq_amp_smooth    = freq_amp_int  #_smooth           # satisfies fs
        self.freq_amp           = freq_amp                         # less values than smooth
        self.SCI                = SCI
        self.time               = np.linspace(0, self.tu[-1], len(self.s))
    
    def AlphaBeta(self): 
        #self.gamma = param['gamma']
        a          = np.array([self.p["a0"].value, self.p["a1"].value, 0]);   
        b          = np.array([self.p["b0"].value, self.p["b1"].value, 0])
        
        t_1   = np.linspace(0,self.T,len(self.s))   
        t_par = np.array([np.ones(t_1.size), t_1, t_1**2])
        # approx alpha and beta as polynomials
        poly = Polynomial.fit(self.time_inter, self.freq_amp_smooth, deg=10)
        x, y = poly.linspace(np.size(self.s))
        
        self.beta  = 1e-4*y*b[1] + b[0]   # self.beta = np.dot(b, t_par)
        self.alpha = np.dot(a, t_par);  
    
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
        
        BirdData = pd.read_csv(auxdata_path+'CopetonData.csv')
        ancho, largo = BirdData['value'][:2]
        s1overCH, s1overLB, s1overLG, RB, r, rdis = BirdData['value'][8:]#[3:8]#[8:]
        
        c = 3.5e4
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
        
        while t < tmax and v[1] > -5000000:
            # -------- trachea ---------------
            dbold  = db[t]                              # forcing 1, before
            a[t]   = (.50)*(1.01*A1*v[1]) + db[t-tau]   # a = Pin, pressure:v(t)y(t) + Pin(t-tau) # envelope*
            db[t]  = -r*a[t-tau]                        # forcing 1, after: -rPin(t-tau)
            ddb    = (db[t]-dbold)/dt                   # Derivada, dPout/dt=Delta forcing1/dt

            #  -rPin(t-tau),  dPout/dt,   v(t)y(t) + Pin(t-tau)
            forcing1, forcing2, PRESSURE = db[t], ddb, a[t] 
            
            tiempot += dt
            v = rk4(dxdt_synth, v, dt);   
            #s1overCH, s1overLB, s1overLG, RB, r, rdis = BirdData["value"][8:]

            noise    = 0.21*(uniform(0, 1)-0.5)
            A1       = amplitud + prct_noise*noise

            if taux == oversamp and n_out<self.fs-1:
                out[n_out]   = RB*v[4]*10
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
        self.synth_env         = normalizar(envelope_cabeza(out, intervalLength=0.05*np.size(self.s)), minout=0)
        self.out_amp           = np.zeros_like(out)
        not_zero               = np.where(self.synth_env > 0.005)
        self.out_amp[not_zero] = out[not_zero] * self.envelope[not_zero] / self.synth_env[not_zero]
        self.s_amp_env         = normalizar(envelope_cabeza(self.out_amp, intervalLength=0.01*np.size(self.s)), minout=0)
        
        sil_filtered_out = butter_lowpass_filter(self.out_amp, self.fs, lcutoff=15000.0, order=6)
        self.out_amp = butter_highpass_filter(sil_filtered_out, self.fs, hcutoff=2000.0, order=5)
        
        
        fu_s, tu_s, Sxx_s = get_spectrogram(self.out_amp, self.fs, window=self.NN,  overlap=self.overlap, sigma=self.sigma)  # out normalized
        
        self.tu_out   = tu_s-tu_s[0]#-self.window_time/2 # -self.t0
        self.fu_out   = fu_s
        self.Sxx_out  = Sxx_s
        self.time_out = np.linspace(0, len(self.out_amp)/self.fs, len(self.out_amp))
        self.Vs       = np.array(self.Vs)
        
        SCI, time_ampl, freq_amp, Ampl_freq, freq_amp_int, time_inter = FFandSCI(self.out_amp, self.time, self.fs, self.t0, window_time=self.window_time) #  method="synth",
        
        self.time_ampl_out          = time_ampl
        self.time_inter_out         = time_inter-time_inter[0]
        self.Ampl_freq_filtered_out = Ampl_freq #Ampl_freq_filtered
        self.freq_amp_smooth_out    = freq_amp_int           # satisfies fs
        self.freq_amp_out           = freq_amp                         # less values than smooth
        self.SCI_out                = SCI
        self.time_out = np.linspace(0, self.tu_out[-1], len(self.out_amp)) 
    
    def Audio(self,num_file,no_syllable):
        wavfile.write('{}/synth4_amp_{}_{}.wav'.format(examples_path,num_file,no_syllable), self.fs, np.asarray(normalizar(self.out_amp),  dtype=np.float32))
        wavfile.write('{}/song_{}_{}.wav'.format(examples_path,num_file,no_syllable),       self.fs, np.asarray(normalizar(self.s),        dtype=np.float32))
    
    # -------------- --------------
    def Solve(self, p):
        self.p = p
        self.AlphaBeta()
        self.MotorGestures()
        
        deltaFF  = np.abs(self.freq_amp_smooth_out-self.freq_amp_smooth)
        deltaSCI = np.abs(self.SCI_out-self.SCI)
        
        self.deltaSCI = deltaSCI#/np.max(deltaSCI)
        self.deltaFF  = 1e-4*deltaFF#/np.max(deltaFF)
        self.scoreSCI = np.sum(self.deltaSCI)/self.deltaSCI.size
        self.scoreFF  = np.sum(abs(self.deltaFF))/self.deltaFF.size
    
    # share methods with the other class
    def residualSCI(self, p):
        self.Solve(p)
        return self.scoreSCI
    
    def residualFF(self, p):
        self.Solve(p)
        return self.scoreFF
    
    # ----------- OPTIMIZATION FUNCTIONS --------------
    def OptimalGamma(self, kwargs):
        start = time.time()
        self.p["gamma"].set(vary=True)
        mi    = lmfit.minimize(self.residualSCI, self.p, nan_policy='omit', **kwargs) 
        self.p["gamma"].set(value=mi.params["gamma"].value, vary=False)
        end   = time.time()
        print("γ* =  {0:.0f}, t={1:.4f} min".format(self.p["gamma"].value, (end-start)/60))
        return mi.params["gamma"].value
    
    def OptimalBs(self, kwargs):
        # ---------------- b0--------------------
        start0 = time.time()
        self.p["b0"].set(vary=True)
        mi0    = lmfit.minimize(self.residualFF, self.p, nan_policy='omit', **kwargs) 
        self.p["b0"].set(vary=False, value=mi0.params["b0"].value)
        end0   = time.time()
        print("b_0*={0:.4f}, t={1:.4f} min".format(self.p["b0"].value, (end0-start0)/60))
        # ---------------- b1--------------------
        start1 = time.time()
        self.p["b1"].set(vary=True)
        mi1    = lmfit.minimize(self.residualFF, self.p, nan_policy='omit', **kwargs) 
        self.p["b1"].set(vary=False, value=mi1.params["b1"].value)
        end1   = time.time()
        print("b_1*={0:.4f}, t={1:.4f} min".format(self.p["b1"].value, (end1-start1)/60))
        #return self.p["b0"].value, self.p["b1"].value #end0-start0, end1-start1
    
    def OptimalParams(self, num_file, kwargs):
        self.Solve(self.p)  # solve first syllable
        
        kwargs["Ns"] = 51;   self.OptimalGamma(kwargs)
        kwargs["Ns"] = 21;   self.OptimalBs(kwargs)
        self.syllable.Audio(num_file, 0)
        
    
    
    
    
    ## ----------------------- PLOT FUNCTIONS --------------------------
    def PlotSynth(self):
        
        fig, ax = plt.subplots(2, 2, figsize=(12, 5), sharex=True, sharey='col')
        fig.subplots_adjust(top=0.85)     
        
        ax[0][0].plot(self.time, normalizar(self.s), label='canto')
        ax[0][0].set_title('Real')
        ax[0][0].plot(self.time, self.envelope, label='envelope')
        ax[0][0].legend(); ax[0][0].set_ylabel("Amplitud (a.u.)")
        
        out = normalizar(self.out_amp) 
        ax[1][0].plot(self.time_out, out+np.abs(np.mean(out)), label='synthetic')
        ax[1][0].set_title('Synthetic') 
        ax[1][0].plot(self.time_out, self.s_amp_env, label='envelope')
        ax[1][0].legend(); ax[1][0].set_xlabel('t (s)'); ax[1][0].set_ylabel("Amplitud (a.u.)")

        Delta_tu   = self.tu[-1] - self.tu[0]
        Delta_tu_s = 1#tu_s[-1] - tu_s[0]

        ax[0][1].pcolormesh(self.tu, self.fu*1e-3, np.log(self.Sxx), cmap=plt.get_cmap('Greys'), rasterized=True)
        ax[0][1].plot(self.time_inter, self.freq_amp_smooth*1e-3, 'b-', label='Smoothed and Interpolated\nto {0} fs'.format(self.fs), lw=2)
        ax[0][1].set_title('Real'); ax[0][1].set_ylabel('f (khz)'); ax[0][1].set_ylim(0, 20);

        ax[1][1].pcolormesh(self.tu_out, self.fu_out*1e-3, np.log(self.Sxx_out), cmap=plt.get_cmap('Greys'), rasterized=True, label='output normalizado amplitud')
        ax[1][1].plot(self.time_inter, self.freq_amp_smooth_out*1e-3, 'b-', label='Smoothed and Interpolated\nto {0} fs'.format(self.fs), lw=2)
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
        c       = viridis(np.linspace(0.3, 1, np.size(self.time_inter)))
        
        ax1.scatter(self.time_inter, self.alpha, c=c, label="alfa")
        ax1.set_title('Air-Sac Pressure')
        ax1.grid()
        ax1.set_ylabel('α (a.u.)'); #ax1.set_xlabel('Time (s)'); 
        ax1.set_ylim(xlim);        #ax[0].legend()

        ax2.scatter(self.time_inter, self.beta, c=c, label="beta")
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
        
        ax[0][0].pcolormesh(self.tu, self.fu*1e-3, np.log(self.Sxx), cmap=plt.get_cmap('Greys'), rasterized=True)
        ax[0][0].plot(self.time_inter, self.freq_amp_smooth*1e-3, 'bo', label='Real',ms=10)
        ax[0][0].plot(self.time_inter, self.freq_amp_smooth_out*1e-3, 'gx', label='Synthetic', ms=6)
        ax[0][0].plot(self.time_ampl, self.freq_amp*1e-3, 'r+', label='sampled ff', ms=5)
        ax[0][0].legend()#title="Fundamenta Frequency")
        ax[0][0].set_ylim((1, 15)); ax[0][0].set_xlim((self.tu[0], self.tu[-1]))
        ax[0][0].set_ylabel('f (khz)'); #ax[0][0].set_xlabel('time (s)');     
        ax[0][0].set_title('Fundamental Frequency (FF)')

        ax[0][1].plot(self.time_ampl, self.SCI, 'o', ms=3, label='Real mean {0:.2f}'.format(np.mean(self.SCI)))
        ax[0][1].plot(self.time_ampl, self.SCI_out, 'o', ms=3, label='Synth mean {0:.2f}'.format(np.mean(self.SCI_out)))
        if flag: 
            m,  b  = np.polyfit(self.time_ampl, self.SCI, 1)
            mo, bo = np.polyfit(self.time_ampl, self.SCI_out, 1)
            ax[0][1].plot(self.time_ampl, m*self.time_ampl+b, 'b-', label='m={0:.2f}, b={1:.2f} '.format(m,b))
            ax[0][1].plot(self.time_ampl, mo*self.time_ampl+bo, 'r-', label='m={0:.2f}, b={1:.2f} '.format(mo,bo))
        ax[0][1].legend()
        ax[0][1].set_ylabel('SCI (adimensionless)'); #ax[0][1].set_xlabel('time (s)'); 
        ax[0][1].set_title('Spectral Content Index (SCI)')
        
        ax[1][0].plot(self.time_inter, self.deltaFF, "-o", color="k", label='Σ R(FF)/len(s)= {:.4f}'.format(self.scoreFF))
        ax[1][0].set_xlabel('time (s)'); ax[1][0].set_ylabel('Error (\\times 10^{4}$Hz)'); ax[1][0].legend()
        ax[1][0].set_title('Fundamental Frequency Error (ΔFF)'); ax[1][0].set_ylim((0,1))
        
        ax[1][1].plot(self.time_ampl, self.deltaSCI, "-o", color="k", label='Σ R(SCI)/len(s) = {:.4f}'.format(self.scoreSCI))
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