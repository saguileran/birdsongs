from .syllable import *
Pool() # crea pool to parallel programming for optimization

class Song(Syllable):
    """
    Store a song and its properties in a class 
    INPUT:
        pahts = paths object with all the folder paths requiered
        no_file = number of the file to analyze
        umbral = threshold to detect a syllable, less than one
    """
    def __init__(self, paths, no_file, umbral=0.05, llambda=1.5):
        self.no_file   = no_file
        self.paths     = paths
        self.llambda   = llambda
        
        self.file_name = self.paths.sound_files[self.no_file-1]
        s, fs = sound.load(self.file_name)
        if len(np.shape(s))>1 and s.shape[1]==2 :s = (s[:,1]+s[:,0])/2 # two channels to one
        
        self.NN       = 1024 
        self.no_overlap  = self.NN//2 ##0.8 
        self.umbral   = umbral
        
        self.SylInd   = []
        self.fs       = fs
        self.s        = sound.normalize(s, max_amp=1.0)
        self.time     = np.linspace(0, len(self.s)/self.fs, len(self.s))
        
        self.envelope = sound.envelope(self.s, Nt=500) 
        t_env = np.arange(0,len(self.envelope),1)*len(self.s)/self.fs/len(self.envelope)
        t_env[-1] = self.time[-1] 
        fun_s = interp1d(t_env, self.envelope)
        self.envelope = fun_s(self.time)
        
        
        Sxx_power, tn, fn, ext = sound.spectrogram (self.s, self.fs, nperseg=self.NN, noverlap=self.no_overlap, mode='psd')  
        Sxx_dB = util.power2dB(Sxx_power) + 96
        Sxx_dB_noNoise, noise_profile, _ = sound.remove_background(Sxx_dB, 
                gauss_std=25, gauss_win=50, llambda=self.llambda)

        self.fu  = fn
        self.tu  = tn
        self.Sxx = sound.smooth(Sxx_dB_noNoise, std=0.5)   
        self.Sxx_dB = util.power2dB(self.Sxx)+96
        
        self.p = lmfit.Parameters()
        # add params:   (NAME   VALUE    VARY    MIN  MAX  EXPR BRUTE_STEP)
        self.p.add_many(('a0',   0.11,   False ,   0, 0.25, None, None), 
                        ('a1',   0.05,   False,   -2,    2, None, None),  
                        ('b0',   -0.1,   False,   -1,  0.5, None, None),  
                        ('b1',      1,   False,  0.2,    2, None, None), 
                        ('gamma', 4e4,   False,  1e4,  1e5, None, None),
                        ('b2',     0.,   False, None, None, None, None), 
                        ('a2',     0.,   False, None, None, None, None))
        self.syllables    = [syl for syl in self.Syllables() if len(syl)>self.NN]
        self.no_syllables = len(self.syllables)
        
        
    def Syllables(self):
        supra      = np.where(self.envelope > self.umbral)[0]
        syllables  = consecutive(supra, min_length=100)
        return [ss for ss in syllables if len(ss) > self.NN/5] # remove short syllables
    
    def Syllables2(self):
        im_bin = rois.create_mask(Sxx_dB_blurred, bin_std=1.5, bin_per=0.5, mode='relative')
    
    
    def Syllable(self, no_syllable, window_time=0.01, Nt=200, llambda=1.5):
        self.no_syllable   = no_syllable
        ss                 = self.syllables[self.no_syllable-1]  # syllable indexes 
        syllable           = self.s[ss[0]:ss[-1]]       # audios syllable
        self.syll_complet  = syllable
        self.time_syllable = self.time[ss[0]:ss[-1]]
        self.t0            = self.time_syllable[0]
        
        self.syllable      = Syllable(self.syll_complet, self.fs, self.time[ss[0]], window_time=window_time, Nt=Nt, llambda=llambda)
        
        self.syllable.no_syllable = self.no_syllable
        self.syllable.no_file     = self.no_file
        self.syllable.p           = self.p
        self.syllable.paths       = self.paths
        
        self.SylInd.append([[no_syllable], [ss]])
        
        return self.syllable
    
    def Chunck(self, no_chunck, window_time=0.005, Nt=20, llambda=1.5, len_win=0.01):
        self.no_chunck     = no_chunck
        chuncks = sound.wave2frames(self.syll_complet,  Nt=512)
        times   = sound.wave2frames(self.time_syllable, Nt=512)
        
        
        self.chunck        = Syllable(chuncks[:,self.no_chunck-1], self.fs, times[self.no_chunck-1,0], NN=256, llambda=llambda, Nt=5)
        
        self.chunck.no_syllable = self.no_chunck
        self.chunck.no_file     = self.no_file
        self.chunck.p           = self.p
        self.chunck.paths       = self.paths
        
        
        
        return self.chunck
    
    # ------------- solver for some parameters -----------------
    def WholeSong(self, method_kwargs, plot=False, syll_max=0):
        self.OptGamma = self.AllGammas(method_kwargs)
        self.p["gamma"].set(value=self.OptGamma)
        if syll_max==0: syll_max=self.syllables.size+1
        for i in range(1,syll_max): # maxi+1):#
            syllable = self.Syllable(i)
            syllable.Solve(self.p)
            syllable.OptimalBs(method_kwargs)
            syllable.Audio()
            if plot:
                self.syllable.PlotAlphaBeta()
                self.syllable.PlotSynth()
                self.syllable.Plot(0)

    def AllGammas(self, method_kwargs):
        self.Gammas = np.zeros(self.no_syllables)
        for i in range(1,self.no_syllables+1):
            syllable = self.Syllable(i)
            syllable.Solve(self.p)
            syllable.OptimalGamma(method_kwargs)
            self.Gammas[i-1] = syllable.p["gamma"].value
        return np.mean(self.Gammas)
            
    def SyntheticSyllable(self):
        self.s_synth = np.empty_like(self.s)
        for i in range(self.syllables.size):
            self.s_synth[self.SylInd[i][1]] = self.syllables[i]
        
    # -------------------- PLOT --------------    
    def Plot(self, flag=0): 
        fig, ax = plt.subplots(3, 1, figsize=(12, 9))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        #fig.tight_layout(pad=3.0)
        
    
        ax[0].pcolormesh(self.tu, self.fu/1000, self.Sxx, cmap=plt.get_cmap('Greys'), rasterized=True)
        
        ax[0].plot(self.syllable.timeFF+self.syllable.t0, self.syllable.FF*1e-3, 'bo', label='FF'.format(self.syllable.fs), lw=1)
        #ax[0].plot(self.syllable.timeFF+self.syllable.t0, self.syllable.FF*1e-3, 'r+', label='sampled FF', ms=2)
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
            #ax[2].pcolormesh(self.chunck.tu, self.chunck.fu, self.chunck.Sxx, cmap=plt.get_cmap('Greys'), rasterized=True)
            ax[0].plot(self.chunck.timeFF+self.chunck.t0, self.chunck.FF*1e-3, 'g-', label='Chunck', lw=4)
            ax[2].plot(self.chunck.timeFF+self.chunck.t0, self.chunck.FF*1e-3, 'g-', label='Chunck', lw=8)
        
        ax[2].pcolormesh(self.syllable.tu+self.syllable.t0, self.syllable.fu*1e-3, self.syllable.Sxx, cmap=plt.get_cmap('Greys'), rasterized=True) 
        #ax[2].plot(self.syllable.timeFF+self.syllable.t0, self.syllable.FF*1e-3, 'b-', lw=5, label='Smoothed and Interpolated\nto {0}=fs'.format(self.syllable.fs))
        ax[2].plot(self.syllable.timeFF+self.syllable.t0, self.syllable.FF*1e-3, 'r+', label='FF', ms=10)
        ax[2].set_ylim((2, 11)); 
        ax[2].set_xlim((self.syllable.timeFF[0]-0.001+self.syllable.t0, self.syllable.timeFF[-1]+0.001+self.syllable.t0));
        ax[2].legend(loc='upper right')
        ax[2].set_xlabel('t (s)'); ax[2].set_ylabel('f (khz)')
        ax[2].set_title('Single Syllable Spectrum, No {}'.format(self.no_syllable))
        
        ax[0].legend(loc='upper right')
        fig.suptitle('Audio: {}'.format(self.file_name[39:]), fontsize=20)#, family='fantasy')
        plt.show()
        print('Number of syllables {}'.format(len(self.syllables)))
