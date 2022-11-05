from syllable_class import *
Pool() # crea pool to parallel programming for optimization

class Song(Syllable):
    """
    Store a song and its properties in a class 
    INPUT:
        file_name = song audio file name path
        window_time = window time length to compute the fundamenta frequency
    """
    def __init__(self, file_name, window_time=0.005):
        fs, s = wavfile.read(file_name)
        if len(np.shape(s))>1 and s.shape[1]==2 :s = (s[:,1]+s[:,0])/2 # two channels to one
        
        self.NN       = 1024 
        self.overlap  = 0.8 #1/1.1
        self.sigma    = self.NN/10
        self.umbral   = 0.05
        
        self.SylInd   = []
        self.fs       = fs
        self.s        = s
        self.envelope = normalizar(envelope_cabeza(self.s,intervalLength=0.01*self.fs), minout=0, method='extremos')
        self.time     = np.linspace(0, len(self.s)/self.fs, len(self.s))
        
        fu_all, tu_all, Sxx_all = get_spectrogram(s, fs, window=self.NN, overlap=self.overlap, sigma=self.sigma)
        self.fu       = fu_all
        self.tu       = tu_all
        self.Sxx      = Sxx_all
        
        self.p = lmfit.Parameters()
        # add params:   (NAME   VALUE    VARY    MIN  MAX  EXPR BRUTE_STEP)
        self.p.add_many(('a0',   0.11,   False ,   0, 0.25, None, None), 
                        ('a1',   0.05,   False,   -2,    2, None, None),  
                        ('b0',   -0.1,   False,   -1,  0.5, None, None),  
                        ('b1',      1,   False,  0.2,    2, None, None), 
                        ('gamma', 4e4,   False,  1e4,  1e5, None, None),
                        ('b2',     0.,   False, None, None, None, None), 
                        ('a2',     0.,   False, None, None, None, None))
        #self.parametros = self.p.valuesdict()
        self.window_time = window_time
        self.syllables   = self.Syllables()
        
    def Syllables(self):
        supra      = np.where(self.envelope > self.umbral)[0]
        syllables  = consecutive(supra, min_length=100)
        return [ss for ss in syllables if len(ss) > self.NN] # remove short syllables
    
    def SyllableNo(self, no_syllable):
        self.no_syllable   = no_syllable
        ss                 = self.syllables[no_syllable-1]  # syllable indexes 
        syllable           = self.s[ss[0]:ss[-1]]       # audios syllable
        self.silb_complet  = syllable
        self.time_syllable = self.time[ss[0]:ss[-1]]
        self.t0            = self.time[ss[0]]
        self.syllable      = Syllable(self.silb_complet, self.fs, self.t0, self.p, self.window_time)
        self.SylInd.append([ss])
        
        return self.syllable
    
    def Chunck(self, no_chunck):
        self.no_chunck     = no_chunck
        chunks_s, chunks_t = Windows(self.silb_complet, self.time_syllable, self.fs, window_time=0.02, overlap=1) # overla=1 not overlap
        self.chunck        = Syllable(chunks_s[no_chunck], self.fs, self.t0, self.p, self.window_time, IL=0.005)
        self.chunck.t0     = chunks_t[no_chunck][0]
        
        self.chunck.NN      = 1024/8 
        self.chunck.sigma   = self.NN/10
        self.chunck.overlap = 0.8
        
        self.chunck.p["b0"].set(value=-0.4)
        
        return self.chunck
    
    # ------------- solver for some parameters -----------------
    def WholeSong(self, num_file, kwargs, plot):#, maxi=5):
        self.SyllableNo(1)          # first syllable
        self.syllable.Solve(self.p)  # solve first syllable
        
        kwargs["Ns"] = 51;   self.syllable.OptimalGamma(kwargs)
        kwargs["Ns"] = 21;   self.syllable.OptimalBs(kwargs)
        self.syllable.Audio(num_file, 0)
            
        for i in range(2,self.syllables.size+1): # maxi+1):#
            self.SyllableNo(i)
            self.syllable.Solve(self.p)
            self.syllable.OptimalBs(kwargs)
            self.syllable.Audio(num_file, i)
            if plot:
                self.syllable.PlotAlphaBeta()
                self.syllable.PlotSynth()
                self.syllable.Plot(0)

    def SyntheticSyllable(self):
        self.s_synth = np.empty_like(self.s)
        for i in range(self.syllables.size):
            self.s_synth[self.SylInd[i]] = self.syllables[i]
        
    # -------------------- PLOT --------------    
    def Plot(self, file_name, flag=0): 
        fig, ax = plt.subplots(3, 1, figsize=(12, 9))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        #fig.tight_layout(pad=3.0)
        
        ax[0].pcolormesh(self.tu, self.fu/1000, np.log(self.Sxx), cmap=plt.get_cmap('Greys'), rasterized=True)
        ax[0].plot(self.syllable.time_inter+self.t0, self.syllable.freq_amp_smooth*1e-3, 'b-', label='FF'.format(self.syllable.fs), lw=2)
        ax[0].plot(self.syllable.time_ampl+self.t0, self.syllable.freq_amp*1e-3, 'r+', label='sampled FF', ms=2)
        for i in range(len(self.syllables)):#for ss in self.syllables:   
            ax[0].plot([self.time[self.syllables[i][0]], self.time[self.syllables[i][-1]]], [0, 0], 'k', lw=5)
            ax[0].text((self.time[self.syllables[i][-1]]-self.time[self.syllables[i][0]])/2, 0.5, str(i))
        
        ax[0].set_ylim(0, 12.000); ax[0].set_xlim(min(self.time), max(self.time));
        ax[0].set_title("Complete Song Spectrum"); 
        ax[0].set_ylabel('f (kHz)'); #ax[0].set_xlabel('t (s)'); 

        ax[1].plot(self.time, self.s/np.max(self.s),'k', label='audio')
        ax[1].plot(self.time, np.ones(len(self.time))*self.umbral, '--', label='umbral')
        ax[1].plot(self.time, self.envelope, label='envelope')
        ax[1].legend(loc='upper right')#loc=1, title='Data')
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
        ax[2].legend(loc='upper right')
        ax[2].set_xlabel('t (s)'); ax[2].set_ylabel('f (khz)')
        ax[2].set_title('Single Syllable Spectrum, No {}'.format(self.no_syllable))
        
        ax[0].legend(loc='upper right')
        fig.suptitle('Audio: {}'.format(file_name[39:]), fontsize=20)#, family='fantasy')
        plt.show()
        print('Number of syllables {}'.format(len(self.syllables)))