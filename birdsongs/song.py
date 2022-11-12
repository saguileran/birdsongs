from .syllable import *
from .utils import *

class Song(Syllable):
    """
    Store a song and its properties in a class 
    INPUT:
        pahts = paths object with all the folder paths requiered
        no_file = number of the file to analyze
        umbral = threshold to detect a syllable, less than one
    """
    def __init__(self, paths, no_file, umbral=0.05, llambda=1.5, NN=512, overlap=0.5, umbral_FF=0.1, flim=(1.5e3,2e4), center=False):
        self.no_file = no_file
        self.paths   = paths
        self.llambda = llambda
        self.flim    = flim
        self.center  = center
        
        self.file_name = self.paths.sound_files[self.no_file-1]
        s, fs = sound.load(self.file_name)
        if len(np.shape(s))>1 and s.shape[1]==2 :s = (s[:,1]+s[:,0])/2 # two channels to one
        
        self.NN       = NN 
        self.no_overlap = int(overlap*self.NN) ##0.8 
        self.umbral   = umbral
        
        self.SylInd   = []
        self.fs       = fs
        self.s        = sound.normalize(s, max_amp=1.0)
        self.time     = np.linspace(0, len(self.s)/self.fs, len(self.s))
        self.id       = "song"
        
        self.envelope = sound.envelope(self.s, Nt=500) 
        t_env = np.arange(0,len(self.envelope),1)*len(self.s)/self.fs/len(self.envelope)
        t_env[-1] = self.time[-1] 
        fun_s = interp1d(t_env, self.envelope)
        self.envelope = fun_s(self.time)
        
        
        Sxx_power, tn, fn, ext = sound.spectrogram (self.s, self.fs, nperseg=self.NN, noverlap=self.no_overlap, mode='psd', flims=flim)  
        Sxx_dB = util.power2dB(Sxx_power) + 96
        Sxx_dB_noNoise, noise_profile, _ = sound.remove_background(Sxx_dB, 
                gauss_std=25, gauss_win=50, llambda=self.llambda)

        self.FF = yin(self.s, fmin=self.flim[0], fmax=self.flim[1], sr=self.fs, frame_length=2*self.NN, 
                win_length=None, hop_length=self.NN//2, trough_threshold=umbral_FF, 
                 center=self.center, pad_mode='constant')
        self.freq    = fft_frequencies(sr=self.fs, n_fft=self.NN)
        self.timeFF = np.linspace(0,self.time[-1], self.FF.size) 
        
        self.fu  = fn
        self.tu  = tn
        self.Sxx = sound.smooth(Sxx_dB_noNoise, std=0.5)   
        self.Sxx_dB = util.power2dB(self.Sxx)+96
        
        self.syllables    = self.Syllables(method="freq")
        self.no_syllables = len(self.syllables)
        print('The son has {} syllables'.format(len(self.syllables)))
        
    def Syllables(self, method="amplitud"):
        if method=="amplitud":
            supra      = np.where(self.envelope > self.umbral)[0]
            candidates = np.split(supra, np.where(np.diff(supra) != 1)[0]+1)
            syllables = [x for x in candidates if len(x) > 2*self.NN] 
        
            return syllables 
        elif method=="freq":
            ss = np.where((self.FF < 15e3) & (self.FF>2e3)) # filter frequency
        
            ff_t   = self.timeFF[ss]                        # cleaning timeFF
            FF_new = self.FF[ss]                            # cleaning FF
            FF_dif = np.abs(np.diff(FF_new))                # find where is it cutted

    #         # alternative form with pandas
    #         df = pd.DataFrame(data={"FF":bird.FF, "time":bird.timeFF})
    #         q  = df["FF"].quantile(0.99)
    #         df[df["FF"] < q]
    #         q_low, q_hi = df["FF"].quantile(0.05), df["FF"].quantile(0.99)
    #         df_filtered = df[(df["FF"] < q_hi) & (df["FF"] > q_low)]

    #         plt.plot(bird.FF, 'o')
    #         plt.plot(df_filtered["FF"], 'o')

            peaks, _ = find_peaks(FF_dif, distance=10, height=500)
            syl = [np.arange(peaks[i]+1,peaks[i+1]) for i in range(len(peaks)-1)]
            syl = [np.arange(0,peaks[0])]+syl+[np.arange(peaks[-1]+1,len(ff_t))]

            syl_intervals = np.array([[ff_t[s][0], ff_t[s][-1]] for s in syl])
            indexes = np.int64(self.fs*syl_intervals)
            indexes = [np.arange(ind[0],ind[1],1) for ind in indexes]
            
            return [x for x in indexes if len(x) > 2*self.NN]
        
        elif "maad":
            im_bin = rois.create_mask(self.Sxx_dB, bin_std=1.5, bin_per=0.5, mode='relative')
        
    
    def Syllable(self, no_syllable, NN=512):
        self.no_syllable   = no_syllable
        ss                 = self.syllables[self.no_syllable-1]  # syllable indexes 
        self.syll_complet  = self.s[ss]       # audios syllable
        self.time_syllable = self.time[ss]
        self.t0            = self.time_syllable[0]
        
        self.syllable      = Syllable(self.syll_complet, self.fs, self.t0, center=self.center, flim=self.flim, NN=NN)
            
        self.syllable.no_syllable  = self.no_syllable
        self.syllable.no_file      = self.no_file
        self.syllable.paths        = self.paths
        self.syllable.id           = "syllable"
        
        self.SylInd.append([[no_syllable], [ss]])
        
        fraction = self.syll_complet.size/1024
        Nt_new = int(((fraction%1)/fraction+1)*1024)
        self.chuncks    = sound.wave2frames(self.syll_complet,  Nt=Nt_new)
        self.times_chun = sound.wave2frames(self.time_syllable, Nt=Nt_new)
        self.no_chuncks = len(self.chuncks)
        
        return self.syllable
    
    def Chunck(self, no_chunck, Nt=int(256/10), llambda=1.5, NN=256, overlap=0.5):
        self.no_chunck     = no_chunck
        
        self.chunck        = Syllable(self.chuncks[:,self.no_chunck-1], self.fs, self.times_chun[0,self.no_chunck-1], NN=NN, llambda=llambda, Nt=Nt, overlap=overlap, center=self.center, flim=self.flim)
        
        self.chunck.no_syllable  = self.no_chunck
        self.chunck.no_file      = self.no_file
        self.chunck.paths        = self.paths
        self.chunck.id           = "chunck"
        
        return self.chunck
    
    # ------------- solver for some parameters -----------------
    def WholeSong(self, method_kwargs, plot=False, syll_max=0):
        self.OptGamma = self.AllGammas(method_kwargs)
        self.p["gamma"].set(value=opt_gamma)
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

    def SyntheticSyllable(self):
        self.s_synth = np.empty_like(self.s)
        for i in range(self.syllables.size):
            self.s_synth[self.SylInd[i][1]] = self.syllables[i]