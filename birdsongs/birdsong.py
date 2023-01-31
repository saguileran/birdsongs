from .syllable import *
from .util import *

class BirdSong(Syllable, object):
    #%%
    """
    Store a song and its properties in a class 
    INPUT:
        pahts = paths object with all the folder paths requiered
        no_file = number of the file to analyze
        umbral = threshold to detect a syllable, less than one
    """
    def __init__(self, paths, no_file, sfs=[], umbral=0.05, llambda=1., NN=1024, overlap=0.5, center=False,
                umbral_FF=1.5, flim=(1.5e3,2e4),  tlim=[], split_method="freq", Nt=500, syll_times=[]):
        self.no_file = no_file
        self.paths   = paths
        self.llambda = llambda
        self.flim    = flim
        self.center  = center
        self.file_path = self.paths.sound_files[self.no_file]
        self.file_name =  str(self.paths.sound_files[self.no_file])[len(str(self.paths.audios))+1:]
        
        if len(sfs)==0:
            #s, fs = sound.load(self.file_path)
            s, fs = librosa.load(self.file_path, sr=None)
            s = librosa.to_mono(s)
            self.id = "birdsong"
        else:
            s, fs   = sfs
            self.id = "birdsong-synth"
        
        if len(tlim)==0: 
            self.s  = sound.normalize(s, max_amp=1.0)
            self.t0 = 0
        else:          
            self.s  = sound.normalize(s[int(tlim[0]*fs):int(tlim[1]*fs)], max_amp=1.0)
            self.t0 = tlim[0]
            self.tlim = tlim
        
        self.NN         = NN
        self.win_length = self.NN//2
        self.hop_length = self.NN//4
        self.center     = center
        self.no_overlap = int(overlap*self.NN)
        self.umbral_FF  = umbral_FF
        
        self.umbral   = umbral
        
        self.SylInd   = []
        self.fs       = fs
        self.time_s   = np.linspace(0, len(self.s)/self.fs, len(self.s))
        self.envelope = Enve(self.s, self.fs, Nt=Nt)
        
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
        self.freq    = librosa.fft_frequencies(sr=self.fs, n_fft=self.NN) 
        self.time    = librosa.times_like(X=self.stft,sr=self.fs, hop_length=self.hop_length, n_fft=self.NN) #, axis=-1
        self.time -= self.time[0]
        
        self.FF     = yin(self.s, fmin=self.flim[0], fmax=self.flim[1], sr=self.fs, frame_length=self.NN, 
                          win_length=self.win_length, hop_length=self.hop_length, trough_threshold=self.umbral_FF, center=self.center, pad_mode='constant')
        
        if len(syll_times)==0:
            self.syllables = self.Syllables(method=split_method)
        else:
            indexes = np.int64(self.fs*np.array(syll_times))
            indexes = [np.arange(ind[0],ind[1],1) for ind in indexes]
            
            self.syllables = [x for x in indexes if len(x) > 2*self.NN]
        
        self.no_syllables = len(self.syllables)
        print('The son has {} syllables'.format(self.no_syllables))
    
    #%%
    def Syllables(self, method="freq"):
        if method=="amplitud":
            supra      = np.where(self.envelope > self.umbral)[0]
            candidates = np.split(supra, np.where(np.diff(supra) != 1)[0]+1)
            
            return [x for x in candidates if len(x) > 2*self.NN] 
        elif method=="freq":
            # ss = np.where((self.FF < self.flim[1]) & (self.FF>self.flim[0])) # filter frequency
            # ff_t   = self.time[ss]                        # cleaning timeFF
            # FF_new = self.FF[ss]                            # cleaning FF
            # FF_dif = np.abs(np.diff(FF_new))                # find where is it cutted
            # # alternative form with pandas
            df = pd.DataFrame(data={"FF":self.FF, "time":self.time})
            q  = df["FF"].quantile(0.99)
            df[df["FF"] < q]
            q_low, q_hi = df["FF"].quantile(0.1), df["FF"].quantile(0.99)
            df_filtered = df[(df["FF"] < q_hi) & (df["FF"] > q_low)]
            
            ff_t   = self.time[df_filtered["FF"].index]
            FF_new = self.FF[df_filtered["FF"].index]
            FF_dif = np.abs(np.diff(FF_new))
            # plt.plot(self.FF, 'o');  plt.plot(df_filtered["FF"], 'o')
            
            peaks, _ = find_peaks(FF_dif, distance=10, height=500) # FF_dif
            syl = [np.arange(peaks[i]+1,peaks[i+1]) for i in range(len(peaks)-1)]
            syl = [np.arange(0,peaks[0])]+syl+[np.arange(peaks[-1]+1,len(ff_t))]

            syl_intervals = np.array([[ff_t[s][0], ff_t[s][-1]] for s in syl])
            indexes = np.int64(self.fs*syl_intervals)
            indexes = [np.arange(ind[0],ind[1],1) for ind in indexes]
            
            return [x for x in indexes if len(x) > 2*self.NN]
        
        elif "maad":
            im_bin = rois.create_mask(self.Sxx_dB, bin_std=1.5, bin_per=0.5, mode='relative')
        
    #%%
    def Syllable(self, no_syllable, NN=1024):
        self.no_syllable   = no_syllable
        ss                 = self.syllables[self.no_syllable-1]  # syllable indexes 
        self.syll_complet  = self.s[ss]       # audios syllable
        self.time_syllable = self.time_s[ss]
        self.t0            = self.time_syllable[0]
        
        self.syllable      = Syllable(self, tlim=(self.time_syllable[0], self.time_syllable[-1]), flim=self.flim, NN=NN, file_name=self.file_name+"synth")
            
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
    
    #%%
    def Chunck(self, no_chunck, Nt=int(256/10), llambda=1.5, NN=256, overlap=0.5):
        self.no_chunck     = no_chunck
        
        self.chunck        = Syllable(self, t0 = self.times_chun[0,self.no_chunck-1], NN=NN, llambda=llambda, Nt=Nt, overlap=overlap, flim=self.flim, out=self.chuncks[:,self.no_chunck-1], file_name=self.file_name+"-chunck")
        
        self.chunck.no_syllable  = self.no_chunck
        #self.chunck.no_file      = self.no_file
        #self.chunck.paths        = self.paths
        self.chunck.id           = "chunck"
        
        return self.chunck
    
    def WriteAudio(self):
        name = '{}/{}-{}.wav'.format(self.paths.examples,self.file_name, self.id)
        WriteAudio(name, fs=self.fs, s=self.s)
    
    #%%
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

    #%%
    def SyntheticSyllable(self):
        self.s_synth = np.empty_like(self.s)
        for i in range(self.syllables.size):
            self.s_synth[self.SylInd[i][1]] = self.syllables[i]

    #%%
    def Play(self): playsound(self.file_path)

    #%%
    def Set(self, p_array):
        self.p["a0"].set(value=p_array[0])
        self.p["a1"].set(value=p_array[1])
        self.p["a2"].set(value=p_array[2])
        self.p["b0"].set(value=p_array[3])
        self.p["b1"].set(value=p_array[4])
        self.p["b2"].set(value=p_array[5])