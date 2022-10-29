from syllable_class import *

class Song(Syllable):
    """
    Store a song and its properties in a class 
    INPUT:
        file_name = song audio file name path
    """
    def __init__(self, file_name):
        fs, s = wavfile.read(file_name)
        if len(np.shape(s))>1 and s.shape[1]==2 :s = (s[:,1]+s[:,0])/2 # two channels to one
        
        self.NN       = 1024 
        self.overlap  = 1/1.1
        self.sigma    = self.NN/10
        self.umbral   = 0.05
        
        self.TimesInd = []
        self.fs       = fs
        self.s        = s
        self.envelope = normalizar(envelope_cabeza(self.s,intervalLength=0.01*self.fs), minout=0, method='extremos')
        self.time     = np.linspace(0, len(self.s)/self.fs, len(self.s))
        
        fu_all, tu_all, Sxx_all = get_spectrogram(s, fs, window=self.NN, overlap=self.overlap, sigma=self.sigma)
        self.fu       = fu_all
        self.tu       = tu_all
        self.Sxx      = Sxx_all
        
        self.p = lmfit.Parameters()
        # add wi  : (NAME    VALUE  VARY   MIN    MAX  EXPR  BRUTE_STEP)
        self.p.add_many(('a0',   0.11,   False, 0,   0.25,   None, None), 
                        ('a1',    0.05,  False, -2,     2,  None, None),  
                        ('b0',    -0.1,  False, -1,   0.5,  None, None),  
                        ('b1',    1,     False,  0.2,   2,  None, None), 
                        ('gamma', 40000, False,  0.1,  1e5, None, None),
                        ('b2',    0.,    False, None, None, None, None), 
                        ('a2',    0.,    False, None, None, None, None))
        #self.parametros = self.p.valuesdict()
        self.window_time = 0.005
        self.syllables   = self.Syllables()
        
    def Syllables(self):
        supra      = np.where(self.envelope > self.umbral)[0]
        syllables  = consecutive(supra, min_length=100)
        return [ss for ss in syllables if len(ss) > self.NN] # elimino silabas cortas    
    
    def SyllableNo(self, no_syllable):
        self.no_syllable   = no_syllable
        ss                 = self.syllables[no_syllable-1]  # silbido in indexes 
        syllable           = self.s[ss[0]:ss[-1]]       # silaba en audio
        self.silb_complet  = syllable
        self.time_syllable = self.time[ss[0]:ss[-1]]# -0.02/2
        self.t0            = self.time[ss[0]]
        self.syllable      = Syllable(self.silb_complet, self.fs, self.t0, self.window_time, self.p)
        self.TimesInd.append([ss])
    
    def Chunck(self, no_chunck):
        self.no_chunck     = no_chunck
        chunks_s, chunks_t = Windows(self.silb_complet, self.time_syllable, self.fs, window_time=0.02, overlap=1) # overla=1 not overlap
        self.chunck        = Sillable(chunks_s[no_chunck], self.fs, self.t0, self.window_time, self.p)
        self.chunck.t0     = chunks_t[no_chunck][0]
    
    # ------------- solver for some parameters -----------------
    def SolveSyllable(self, p):
        self.syllable.p = p
        self.syllable.AlphaBeta()
        self.syllable.MotorGestures()
        
        deltaFF  = np.abs(self.syllable.freq_amp_smooth_out-self.syllable.freq_amp_smooth)
        deltaSCI = np.abs(self.syllable.SCI_out-self.syllable.SCI)
        self.syllable.deltaSCI = deltaSCI#/np.max(deltaSCI)
        self.syllable.deltaFF  = 1e-4*deltaFF#/np.max(deltaFF)
        self.syllable.scoreSCI = np.sum(self.syllable.deltaSCI)/self.syllable.deltaSCI.size
        self.syllable.scoreFF  = np.sum(abs(self.syllable.deltaFF))/self.syllable.deltaFF.size
        
    # score functions, they are a real number (we are interest that both are one)    
    def residualSCI(self, p):
        self.SolveSyllable(p)
        return self.syllable.scoreSCI
    
    def residualFF(self, p):
        self.SolveSyllable(p)
        return self.syllable.scoreFF
    
    # ----------- OPTIMIZATION FUNCTIONS --------------
    def OptimalGamma(self, kwargs):
        start = time.time()
        self.syllable.p["gamma"].set(vary=True)
        
        mi    = lmfit.minimize(self.residualSCI, self.syllable.p, nan_policy='omit', **kwargs) 
        self.syllable.p["gamma"].set(value=mi.params["gamma"].value, vary=False)
        end   = time.time()
        print("γ* =  {0:.4f}".format(self.syllable.p["gamma"].value))
        
        return end-start
    
    def OptimalBs(self, kwargs):
        # ---------------- b0--------------------
        start0 = time.time()
        self.syllable.p["b0"].set(vary=True)
        mi0    = lmfit.minimize(self.residualFF, self.syllable.p, nan_policy='omit', **kwargs) #, max_nfev=500) #dual_annealing # , Ns=200
        self.syllable.p["b0"].set(vary=False, value=mi0.params["b0"].value)
        end0   = time.time()
        print("b_0*={0:.4f}".format(self.syllable.p["b0"].value))
        # ---------------- b1--------------------
        start1 = time.time()
        self.syllable.p["b1"].set(vary=True)
        mi1    = lmfit.minimize(self.residualFF, self.syllable.p, nan_policy='omit', **kwargs) 
        self.syllable.p["b1"].set(vary=False, value=mi1.params["b1"].value)
        end1   = time.time()
        print("b_1*={0:.4f}".format(self.p["b1"].value))
        
        return end0-start0, end1-start1
    
            
        
    def Plot(self, file_name, flag=0): #flag = 1 or 0
        fig, ax = plt.subplots(3, 1, figsize=(12, 9))#, sharex=True, gridspec_kw={'width_ratios': [1, 1]})
        fig.subplots_adjust(hspace=0.4)
        
        ax[0].pcolormesh(self.tu, self.fu/1000, np.log(self.Sxx), cmap=plt.get_cmap('Greys'), rasterized=True)
        ax[0].plot(self.syllable.time_inter+self.t0, self.syllable.freq_amp_smooth*1e-3, 'b-', label='FF'.format(self.syllable.fs), lw=2)
        ax[0].plot(self.syllable.time_ampl+self.t0, self.syllable.freq_amp*1e-3, 'r+', label='sampled FF', ms=2)
        for ss in self.syllables:   ax[0].plot([self.time[ss[0]], self.time[ss[-1]]], [0, 0], 'k', lw=5)
        
        ax[0].set_ylim(0, 12.000); ax[0].set_xlim(min(self.time), max(self.time));
        ax[0].set_title("Complete Song Spectrum"); 
        ax[0].set_xlabel('t (s)'); ax[0].set_ylabel('f (kHz)')

        ax[1].plot(self.time, self.s/np.max(self.s),'k', label='audio')
        ax[1].plot(self.time, np.ones(len(self.time))*self.umbral, '--', label='umbral')
        ax[1].plot(self.time, self.envelope, label='envelope')
        ax[1].legend()#loc=1, title='Data')
        ax[1].set_title("Complete Song Sound Wave")
        ax[1].set_xlabel('t (s)'); ax[1].set_ylabel('Amplitud normalaized');
        ax[1].sharex(ax[0])

        if flag==1:
            #ax[2].pcolormesh(self.tu, self.fu, np.log(self.Sxx), cmap=plt.get_cmap('Greys'), rasterized=True)
            ax[0].plot(self.chunck.time_inter+self.chunck.t0, self.chunck.freq_amp_smooth*1e-3, 'g-', label='Chunck', lw=4)
            ax[2].plot(self.chunck.time_inter+self.chunck.t0, self.chunck.freq_amp_smooth*1e-3, 'g-', label='Chunck', lw=10)
        
        ax[2].pcolormesh(self.syllable.tu+self.t0, self.syllable.fu*1e-3, np.log(self.syllable.Sxx), cmap=plt.get_cmap('Greys'), rasterized=True) 
        ax[2].plot(self.syllable.time_inter+self.t0, self.syllable.freq_amp_smooth*1e-3, 'b-', lw=5, label='Smoothed and Interpolated\nto {0}=fs'.format(self.syllable.fs))
        ax[2].plot(self.syllable.time_ampl+self.t0, self.syllable.freq_amp*1e-3, 'r+', label='sampled FF', ms=3)
        ax[2].set_ylim((2, 11)); 
        ax[2].set_xlim((self.syllable.time_inter[0]-0.001+self.t0, self.syllable.time_inter[-1]+0.001+self.t0));
        ax[2].legend()
        ax[2].set_xlabel('t (s)'); ax[2].set_ylabel('f (khz)')
        ax[2].set_title('Single Syllable Spectrum, No {}'.format(self.no_syllable))
        
        ax[0].legend()
        fig.suptitle('Audio: {}'.format(file_name[39:]), fontsize=20)#, family='fantasy')
        print('Number of syllables {}'.format(len(self.syllables)))