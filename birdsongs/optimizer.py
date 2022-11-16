from .syllable import *
from .song import *
from .utils import *

class Optimizer(Syllable):
    def __init__(self, obj, method_kwargs):
        self.obj       = obj
        self.obj0      = obj
        self.method    = method_kwargs["method"]
        del method_kwargs["method"]
        self.kwargs = method_kwargs
        
    def residualSCI(self, p):
        syllable_synth = self.obj.Solve(p)
        return syllable_synth.scoreSCI +  syllable_synth.scoreFF
    # return scoreSxx + syllable_synth.scoreMfccs + syllable_synth.scoreMel # scoreCorrelation #scoreSCI 
    
    def residualFF(self, p):
        syllable_synth = self.obj.Solve(p)
        return syllable_synth.scoreFF # + syllable_synth.scoreCentroid
    
    def residualIndexes(self, p):
        syllable_synth = self.obj.Solve(p)
        #self.entropies = [EAS, ECU, ECV, EPS, EPS_KURT, EPS_SKEW]
        return syllable_synth.scoreACI_sum + syllable_synth.scoreBI + syllable_synth.entropies
    
    def residualCorrelation(self, p):
        syllable_synth = self.obj.Solve(p)
        return syllable_synth.scoreFF -np.mean(syllable_synth.correlation+syllable_synth.Df+syllable_synth.scoreSKL)

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
            print("b_0*={:.4f},\nb_2*={:.4f}, t={:.4f} min".format(self.obj.p["b0"].value, self.obj.p["b2"].value, (end02-start02)/60))
        elif "chunck" in self.obj.id:
            # ---------------- b0--------------------
            start0 = time.time()
            self.obj.p["b0"].set(vary=True)
            mi0    = lmfit.minimize(self.residualFF, self.obj.p, nan_policy='omit', method=self.method, **self.kwargs) 
            self.obj.p["b0"].set(vary=False, value=mi0.params["b0"].value)
            end0   = time.time()
            print("b_0*={0:.4f}, t={1:.4f} min".format(self.obj.p["b0"].value, (end0-start0)/60))
        # ---------------- b1--------------------
        start1 = time.time()
        self.obj.p["b1"].set(vary=True)
        mi1    = lmfit.minimize(self.residualFF, self.obj.p, nan_policy='omit', method=self.method, **self.kwargs) 
        self.obj.p["b1"].set(vary=False, value=mi1.params["b1"].value)
        end1   = time.time()
        print("b_1*={0:.4f}, t={1:.4f} min".format(self.obj.p["b1"].value, (end1-start1)/60))
        #return self.obj.p["b0"].value, self.obj.p["b1"].value #end0-start0, end1-start1
        #return self.obj.p
        obj = self.obj
        
    def OptimalAs(self, obj):
        self.obj = obj
        # ---------------- a0--------------------
        start0 = time.time()
        self.obj.p["a0"].set(vary=True)
        mi0    = lmfit.minimize(self.residualCorrelation, self.obj.p, nan_policy='omit', method=self.method, **self.kwargs) 
        self.obj.p["a0"].set(vary=False, value=mi0.params["a0"].value)
        end0   = time.time()
        print("a_0*={0:.4f}, t={1:.4f} min".format(self.obj.p["a0"].value, (end0-start0)/60))
        # ---------------- a1--------------------
        start1 = time.time()
        self.obj.p["a1"].set(vary=True)
        mi1    = lmfit.minimize(self.residualCorrelation, self.obj.p, nan_policy='omit', method=self.method, **self.kwargs) 
        self.obj.p["a1"].set(vary=False, value=mi1.params["a1"].value)
        end1   = time.time()
        
        print("a_1*={0:.4f}, t={1:.4f} min".format(self.obj.p["a1"].value, (end1-start1)/60))
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
    
    def OptimalParams1(self, NsGamma=51, NsPar=21):
        obj_synth = self.obj.Solve(self.obj.p)     #
        
        #self.kwargs["Ns"] = NsGamma;   
        self.obj.OptimalGamma()
        
        #self.kwargs["Ns"] = NsPar;     
        self.obj.OptimalAs(obj_synth)
        self.obj.OptimalBs(obj_synth)
        
        self.p = self.obj.p
        
        return self.obj.p
        #self.obj.WriteAudio()
    
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
        
    def AllGammas(self, bird):
        Gammas = np.zeros(bird.no_syllables)
        for i in range(1,bird.no_syllables+1):
            obj         = bird.Syllable(i)
            Gammas[i-1] = self.OptimalGamma(obj)
            
        self.optimal_gamma = np.mean(Gammas)
        self.Gammas = Gammas
        self.obj    = self.obj0
        self.obj.p["gm"].set(value=self.optimal_gamma, vary=False)
        return Gammas
        
    def OptimalParams(self, obj, Ns=21):         # optimal_gm
        if self.method=="brute": self.kwargs["Ns"] = Ns     
        
        print("As")
        self.OptimalAs(obj)    
        print("Bs")
        self.OptimalBs(obj)
        print("end")
        obj.p = self.obj.p
        
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
            
        self.bird_synth = Song(paths=bird.paths, no_file=bird.paths, sfs=[s_synth_song,bird.fs]) 
        self.bird_synth.id += "synth"
        #self.bird_synth.WriteAudio()
        
        return self.bird_synth