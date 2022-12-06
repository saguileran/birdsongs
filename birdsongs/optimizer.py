from .syllable import *
from .birdsong import *
from .utils import *

class Optimizer(Syllable, object):
    def __init__(self, obj, method_kwargs):
        self.obj    = obj
        self.obj0   = obj
        self.method = method_kwargs["method"]
        self.kwargs = {key:value for key, value in method_kwargs.items() if key!="method"}#method_kwargs
        self.syllables = []
        
    def residualSCI(self, p):
        syllable_synth = self.obj.Solve(p)
        return syllable_synth.SCIFF #scoreSCI +  syllable_synth.scoreFF
    # return scoreSxx + syllable_synth.scoreMfccs + syllable_synth.scoreMel # scoreCorrelation #scoreSCI 
    
    def residualFF(self, p):
        syllable_synth = self.obj.Solve(p)
        return syllable_synth.scoreFF # + syllable_synth.scoreCentroid
    
    # def residualIndexes(self, p):
    #     syllable_synth = self.obj.Solve(p)
    #     #self.entropies = [EAS, ECU, ECV, EPS, EPS_KURT, EPS_SKEW]
    #     return syllable_synth.scoreACI_sum + syllable_synth.scoreBI + syllable_synth.entropies
    
    def residualCorrelation(self, p):
        syllable_synth = self.obj.Solve(p)
        return syllable_synth.residualCorrelation
        # return syllable_synth.scoreFF -np.mean(syllable_synth.correlation+syllable_synth.Df+syllable_synth.scoreSKL)

    # making peackeable
    def residualCorrelation_p(p): return Optimizer, (p.a,)
    # print("pickling a C instance...")
    
    
    
    
    # -----------------------------------------------------------------------
    # ---------------- OPTIMAL PARAMETERS ------------------------------
    def OptimalBs(self, obj):
        self.obj = obj
        if "syllable" in self.obj.id:
            # ---------------- b0 and b2 --------------------
            start02 = time.time()
            self.obj.p["b0"].set(vary=True);  obj.p["b2"].set(vary=True);
            mi02    = lmfit.minimize(self.residualFF, self.obj.p, nan_policy='omit', method=self.method, **self.kwargs) 
            self.obj.p["b0"].set(vary=False, value=mi02.params["b0"].value)
            self.obj.p["b2"].set(vary=False, value=mi02.params["b2"].value)
            end02   = time.time()
            print(r"$b_0*$"+"={:.4f},\nb_2*={:.4f}, t={:.4f} min".format(self.obj.p["b0"].value, self.obj.p["b2"].value, (end02-start02)/60))
        elif "chunck" in self.obj.id:
            # ---------------- b0--------------------
            start0 = time.time()
            self.obj.p["b0"].set(vary=True)
            mi0    = lmfit.minimize(self.residualFF, self.obj.p, nan_policy='omit', method=self.method, **self.kwargs) 
            self.obj.p["b0"].set(vary=False, value=mi0.params["b0"].value)
            end0   = time.time()
            print(r"$b_0*$"+"={0:.4f}, t={1:.4f} min".format(self.obj.p["b0"].value, (end0-start0)/60))
        # ---------------- b1--------------------
        start1 = time.time()
        self.obj.p["b1"].set(vary=True)
        mi1    = lmfit.minimize(self.residualFF, self.obj.p, nan_policy='omit', method=self.method, **self.kwargs) 
        self.obj.p["b1"].set(vary=False, value=mi1.params["b1"].value)
        end1   = time.time()
        print(r"$b_1*$"+"={0:.4f}, t={1:.4f} min".format(self.obj.p["b1"].value, (end1-start1)/60))
        #return self.obj.p["b0"].value, self.obj.p["b1"].value #end0-start0, end1-start1
        #return self.obj.p
        obj = self.obj
        
    def OptimalAs(self, obj):
        self.obj = obj
        # ---------------- a0--------------------
        start0 = time.time()
        self.obj.p["a0"].set(vary=True)
        
        # def ResidualCo(c):
        #     print("pickling a C instance...")
        #     return Syllable, (c.residualCorrelation,)
        # copyreg.pickle(Syllable, ResidualCo)
        
        
        mi0    = lmfit.minimize(self.residualCorrelation, self.obj.p, nan_policy='omit', method=self.method, **self.kwargs) 
        self.obj.p["a0"].set(vary=False, value=mi0.params["a0"].value)
        end0   = time.time()
        print(r"$a_0*$"+"={0:.4f}, t={1:.4f} min".format(self.obj.p["a0"].value, (end0-start0)/60))
        # ---------------- a1--------------------
        start1 = time.time()
        self.obj.p["a1"].set(vary=True)
        
        mi1    = lmfit.minimize(self.residualCorrelation, self.obj.p, nan_policy='omit', method=self.method, **self.kwargs) 
        self.obj.p["a1"].set(vary=False, value=mi1.params["a1"].value)
        end1   = time.time()
        
        print(r"$a_1*$"+"={0:.4f}, t={1:.4f} min".format(self.obj.p["a1"].value, (end1-start1)/60))
        #return self.obj.p["b0"].value, self.obj.p["b1"].value #end0-start0, end1-start1
        obj = self.obj
        
    def OptimalGamma(self, obj):
        self.obj = obj
        start = time.time()
        self.obj.p["gm"].set(vary=True)
        mi    = lmfit.minimize(self.residualSCI, self.obj.p, nan_policy='omit', method=self.method, **self.kwargs) 
        self.obj.p["gm"].set(value=mi.params["gm"].value, vary=False)
        end   = time.time()
        print("γ* =  {0:.0f}, t={1:.4f} min".format(self.obj.p["gm"].value, (end-start)/60))
        
        obj = self.obj
        return mi.params["gm"].value
    
#     def OptimalParams1(self, NsGamma=51, NsPar=21):
#         obj_synth = self.obj.Solve(self.obj.p)     #
        
#         #self.kwargs["Ns"] = NsGamma;   
#         #self.obj.OptimalGamma()
        
#         #self.kwargs["Ns"] = NsPar;     
#         self.obj.OptimalAs(obj_synth)
#         self.obj.OptimalBs(obj_synth)
        
#         self.p = self.obj.p
        
#         return self.obj.p
#         #self.obj.WriteAudio()
    
    
    # ----------- OPTIMIZATION FUNCTIONS --------------
    # Solve the minimization problem at once
    def CompleteSolution(self):
        start = time.time()
        
        self.obj.p = lmfit.Parameters()
        # add params:   (NAME   VALUE    VARY    MIN  MAX  EXPR BRUTE_STEP)
        self.obj.p.add_many(('a0', 0.11, True ,   0, 0.25,  None, None),#0.01), 
                            ('a1', 0.05, True,   -2,    2,  None, None),#0.1),  
                            ('a2',   0., True,    0,    2,  None, None),
                            ('b0', -0.1, True,   -1,  0.5,  None, None),#0.03),  
                            ('b1',   1,  True,  0.2,     2,  None, None),#0.04), 
                            ('b2',   0., True,    0,    2,  None, None), 
                            ('gm',   gm, True,  1e4,  1e5, None, 2000))
        mi    = lmfit.minimize(self.residualFFandSCI, self.obj.p, nan_policy='omit', method=self.method, **self.kwargs) 
        self.obj.p["a0"].set(vary=False, value=mi.params["a0"].value)
        self.obj.p["a1"].set(vary=False, value=mi.params["a1"].value)
        self.obj.p["a2"].set(vary=False, value=mi.params["a2"].value)
        self.obj.p["b0"].set(vary=False, value=mi.params["b0"].value)
        self.obj.p["b1"].set(vary=False, value=mi.params["b1"].value)
        self.obj.p["b2"].set(vary=False, value=mi.params["b2"].value)
        self.obj.p["gm"].set(vary=False, value=mi.params["gm"].value)
        
        synth = self.obj.Solve(self.obj.p)
        end = time.time()
        print("Time of execution = {0:.4f}".format(end-start))
        return synth
        
    def AllGammas(self, bird, Ns=21):
        start = time.time()
        if self.method=="brute": self.kwargs["Ns"] = Ns
        Gammas = np.zeros(bird.no_syllables)
        for i in range(1,bird.no_syllables+1):
            print("Syllable {}/{}".format(i,bird.no_syllables))
            obj         = bird.Syllable(i)
            Gammas[i-1] = self.OptimalGamma(obj)
            
        self.optimal_gamma = np.mean(Gammas)
        self.Gammas = Gammas
        self.obj    = self.obj0
        #self.obj.p["gm"].set(value=self.optimal_gamma, vary=False)
        end = time.time()
        print("Time of execution = {0:.4f} minutes".format((end-start)/60))
        return self.optimal_gamma
        
    def OptimalParams(self, obj, Ns=21):
        if self.method=="brute": self.kwargs["Ns"] = Ns     
        start = time.time()
        print("As");   self.OptimalAs(obj);   
        print("Bs");   self.OptimalBs(obj);
        print("end")
        
        obj.p = self.obj.p
        
        end = time.time()
        print("Time of execution = {0:.4f} minutes".format((end-start)/60))
        #obj.p["gamma"].set(value=optimal_gm)
        #obj_synth = obj.Solve(obj.p)     #
        
        #return obj.p
    
    def AllOptimals(self, bird):  # optimal for all syllables
        s_synth_song = np.zeros_like(bird.s)   # synthetic song init
        indexes = bird.Syllables()
        self.AllGammas(bird)
        Display(bird)
        
        for i in range(1,bird.no_syllables+1):
            obj       = bird.Syllable(i)
            print("Syllable: {}".format(i))
            self.OptimalParams(obj=obj)
            obj_synth = obj.Solve(obj.p)            
            
            s_synth_song[indexes[i-1]] = obj_synth.s
            
        self.bird_synth = BirdSong(paths=bird.paths, no_file=bird.paths, sfs=[s_synth_song,bird.fs]) 
        self.bird_synth.id += "synth"
        #self.bird_synth.WriteAudio()
        
        return self.bird_synth
    
    
    
    
    def AllGammasByTimes(self, times, Ns=21, NN=1024):
        if self.method=="brute": self.kwargs["Ns"] = Ns     

        times   = np.array(times)
        Gammas = np.zeros(times.shape[0])
        for i in range(times.shape[0]):
            print("Syllable {}/{}".format(i+1,times.shape[0]))
            print(times[i,:])
            syllable = Syllable(self.obj, tlim=times[i,:], NN=NN)
            Gammas[i-1] = self.OptimalGamma(syllable)
            self.obj = self.obj0
            
        self.optimal_gamma = np.mean(Gammas)
        self.Gammas = Gammas
        #self.obj.p["gm"].set(value=self.optimal_gamma)
        #bird.obj.p["gm"].set(value=self.optimal_gamma)
        
        return self.optimal_gamma
    
    def SongByTimes(self, times, Ngm=11, Nba=11, NN=512, optimal_gamm=-1): 
        times   = np.array(times)
        indexes = np.int64(times*self.obj.fs)
        # times = [[t0_1, tend1],...,[t0_N, tendN]]
        print("Looking for optial time scale constant (γ*)")
        tstart = time.time()
        
        if optimal_gamm==-1:  optimal_gamm = self.AllGammasByTimes(times, Ns=Ngm, NN=NN)

        tend = time.time()
        print("γ found {} over {} syllables. Time of execution {:.4f} min".format(optimal_gamm, times.shape[0], (tend-tstart)/60))
        
        self.bird_s = np.zeros_like(self.obj.s)
        self.bird_t = np.copy(self.obj.time_s)
        
        self.synth_bird_s = np.zeros_like(self.obj.s)
        self.synth_bird_t = np.copy(self.obj.time_s)
        self.alphas       = np.zeros_like(self.obj.s)
        self.betas        = np.zeros_like(self.obj.s)
        self.ps = []
        
        start = time.time()
        print("Start syllables optimization")
        for i in range(times.shape[0]):
            print("Syllable {}/{}".format(i+1,times.shape[0]))
            obj       = Syllable(self.obj, tlim=times[i,:], umbral_FF=self.obj.umbral_FF, Nt=30, NN=NN)
            obj.p["gm"].set(value=optimal_gamm)
            obj_synth = obj.Solve(obj.p)
            self.OptimalParams(obj, Ns=Nba)
            obj_synth = obj.Solve(obj.p)
            self.syllables.append(obj_synth.s)
            
            index_0, index_end = indexes[i,0], int(indexes[i,0]+obj_synth.s.size)
            
            self.bird_s[index_0: index_end]       = obj.s
            self.synth_bird_s[index_0: index_end] = obj_synth.s
            self.alphas[index_0: index_end]       = obj_synth.alpha
            self.betas[index_0: index_end]        = obj_synth.beta
            
            self.ps.append(obj.p)
            self.obj=self.obj0
            
        # self.synth_bird = BirdSong(self.obj.paths, self.obj.no_file, sfs=[self.synth_bird_s, self.obj.fs], split_method="amplitud", umbral=-.01)
        self.synth_bird = Syllable(self.obj0, NN=NN, sfs=[self.synth_bird_s, obj.fs])
        self.bird       = Syllable(self.obj0, NN=NN, sfs=[self.bird_s, obj.fs])
        
        #self.synth_bird.synth    = self.synth_bird 
        # self.synth_bird.file_name = self.obj.file_name
        self.synth_bird.alpha    = self.alphas
        self.synth_bird.beta     = self.betas
        self.synth_bird.ps       = self.ps
        self.synth_bird.id       = "birdsong-synth"
        self.bird.id             = "birdsong"
        
        end = time.time()
        print("Optimal parameters found. The time of execution was {:.4f} hours".format((end-start)/60/60))
        
        return self.bird, self.synth_bird
        
    def SmoothAB(self, times, smooth=3, fraction=5):
        times   = np.array(times)
        indexes = np.int64(times*self.obj.fs)
        self.alphas_smooth = np.copy(self.alphas)
        self.betas_smooth  = np.copy(self.betas)
        
        for i in range(times.shape[0]):
            index_0, index_end = indexes[i,0], int(indexes[i,1])-1
            
            NoSamples = (index_end-index_0)//fraction  # syllable samples length fraction
            x = self.synth_bird_t
            yleft  = smoothstep(x, x_min=x[int(index_0-NoSamples/2)], x_max=x[index_0], N=smooth)
            yright = smoothstep(x, x_min=x[index_end], x_max=x[int(index_end+NoSamples/2)], N=smooth)
            
            self.synth_bird_s[int(index_0-NoSamples/2):index_0]      = self.synth_bird_s[index_0]*yleft[int(index_0-NoSamples/2):index_0]
            self.synth_bird_s[index_end:int(index_end+NoSamples/2)] = self.synth_bird_s[index_end]*yright[index_end:int(index_end+NoSamples/2)][::-1]
            
            # self.alphas_smooth[int(index_0-NoSamples/2):index_0]     = self.alphas[index_0]*yleft[int(index_0-NoSamples/2):index_0]
            # self.betas_smooth[int(index_0-NoSamples/2):index_0]      = self.betas[index_0]*yleft[int(index_0-NoSamples/2):index_0]
            # self.alphas_smooth[index_end:int(index_end+NoSamples/2)] = self.alphas[index_end]*yright[index_end:int(index_end+NoSamples/2)][::-1]
            # self.betas_smooth[index_end:int(index_end+NoSamples/2)]  = self.betas[index_end]*yright[index_end:int(index_end+NoSamples/2)][::-1]
            # ----------------------------------------------
        
        # self.synth_bird.alphas_smooth = self.alphas_smooth
        # self.synth_bird.betas_smooth  = self.betas_smooth
        
        # self.synth_bird_s = self.synth_bird_s
        
        return self.synth_bird_s