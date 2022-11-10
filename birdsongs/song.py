from .syllable import *
from .functions import *

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
        self.p.add_many(('a0', 0.11, False ,   0, 0.25, None, None), 
                        ('a1', 0.05, False,   -2,    2, None, None),
                        ('a2',   0., False,    0,    2, None, None),
                        ('b0', -0.1, False,   -1,  0.5, None, None),  
                        ('b1',    1, False,  0.2,    2, None, None), 
                        ('b2',   0., False,    0,    3, None, None), 
                        ('gm',  4e4, False,  1e4,  1e5, None, None))
        self.syllables    = [syl for syl in self.Syllables() if len(syl)>self.NN]
        self.no_syllables = len(self.syllables)
        print('The son has {} syllables'.format(len(self.syllables)))
        
    def Syllables(self, min_length=64, stepsize=1):
        supra      = np.where(self.envelope > self.umbral)[0]
        candidates = np.split(supra, np.where(np.diff(supra) != stepsize)[0]+1)
        syllables = [x for x in candidates if len(x) > min_length]
        
        return [ss for ss in syllables if len(ss) > self.NN/5] # remove short syllables
    
    def Syllables2(self):
        im_bin = rois.create_mask(Sxx_dB_blurred, bin_std=1.5, bin_per=0.5, mode='relative')
    
    
    def Syllable(self, no_syllable, Nt=200, llambda=1.5, NN=512):
        self.no_syllable   = no_syllable
        ss                 = self.syllables[self.no_syllable-1]  # syllable indexes 
        syllable           = self.s[ss[0]:ss[-1]]       # audios syllable
        self.syll_complet  = syllable
        self.time_syllable = self.time[ss[0]:ss[-1]]
        self.t0            = self.time_syllable[0]
        
        self.syllable      = Syllable(self.syll_complet, self.fs, self.time[ss[0]], Nt=Nt, llambda=llambda)
            
        self.syllable.no_syllable  = self.no_syllable
        self.syllable.no_file      = self.no_file
        self.syllable.p            = self.p
        self.syllable.paths        = self.paths
        self.syllable.id           = "syllable"
        
        self.SylInd.append([[no_syllable], [ss]])
        
        self.chuncks    = sound.wave2frames(self.syll_complet,  Nt=Nt)
        self.times_chun = sound.wave2frames(self.time_syllable, Nt=Nt)
        self.no_chuncks = len(self.chuncks)
        
        return self.syllable
    
    def Chunck(self, no_chunck, Nt=5, llambda=1.5, NN=64):
        self.no_chunck     = no_chunck
        
        self.chunck        = Syllable(self.chuncks[:,self.no_chunck-1], self.fs, self.times_chun[self.no_chunck-1,0], NN=NN, llambda=llambda, Nt=Nt)
        
        self.chunck.no_syllable  = self.no_chunck
        self.chunck.no_file      = self.no_file
        self.chunck.p            = self.p
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