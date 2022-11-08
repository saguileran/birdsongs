from .syllable import *
from .functions import *

class Optimizer(Syllable):
    def __init__(self, obj, method_kwargs):
        self.obj       = obj
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
        return syllable_synth.scoreSCI
    # syllable_synth.scoreSxx + syllable_synth.scoreCentroid
    
    def residualCorrelation(self, p):
        syllable_synth = self.obj.Solve(p)
        return np.mean(syllable_synth.correlation+syllable_synth.DF+syllable_synth.scoreSKL)

    # -----------------------------------------------------------------------
    # ---------------- OPTIMAL PARAMETERS ------------------------------
    def OptimalBs(self, synth):
        if "syllable" in self.id:
            # ---------------- b2--------------------
            start0 = time.time()
            self.obj.p["b2"].set(vary=True)
            mi2    = lmfit.minimize(synth.residualFF, self.obj.p, nan_policy='omit', method=self.method, **self.kwargs) 
            self.obj.p["b2"].set(vary=False, value=mi2.params["b2"].value)
            end2   = time.time()
            print("b_2*={0:.4f}, t={1:.4f} min".format(self.obj.p["b2"].value, (end0-start0)/60))
        # ---------------- b0--------------------
        start0 = time.time()
        self.obj.p["b0"].set(vary=True)
        mi0    = lmfit.minimize(synth.residualFF, self.obj.p, nan_policy='omit', method=self.method, **self.kwargs) 
        self.obj.p["b0"].set(vary=False, value=mi0.params["b0"].value)
        end0   = time.time()
        print("b_0*={0:.4f}, t={1:.4f} min".format(self.obj.p["b0"].value, (end0-start0)/60))
        # ---------------- b1--------------------
        start1 = time.time()
        self.obj.p["b1"].set(vary=True)
        mi1    = lmfit.minimize(synth.residualFF, self.obj.p, nan_policy='omit', method=self.method, **self.kwargs) 
        self.obj.p["b1"].set(vary=False, value=mi1.params["b1"].value)
        end1   = time.time()
        print("b_1*={0:.4f}, t={1:.4f} min".format(self.obj.p["b1"].value, (end1-start1)/60))
        #return self.obj.p["b0"].value, self.obj.p["b1"].value #end0-start0, end1-start1
        
    def OptimalAs(self, synth):
        # ---------------- a0--------------------
        start0 = time.time()
        self.obj.p["a0"].set(vary=True)
        mi0    = lmfit.minimize(synth.residualCorrelation, self.obj.p, nan_policy='omit', method=self.method, **self.kwargs) 
        self.obj.p["a0"].set(vary=False, value=mi0.params["a0"].value)
        end0   = time.time()
        print("a_0*={0:.4f}, t={1:.4f} min".format(self.obj.p["a0"].value, (end0-start0)/60))
        # ---------------- a1--------------------
        start1 = time.time()
        self.obj.p["a1"].set(vary=True)
        mi1    = lmfit.minimize(synth.residualCorrelation, self.obj.p, nan_policy='omit', method=self.method, **self.kwargs) 
        self.obj.p["a1"].set(vary=False, value=mi1.params["a1"].value)
        end1   = time.time()
        print("a_1*={0:.4f}, t={1:.4f} min".format(self.obj.p["a1"].value, (end1-start1)/60))
        #return self.obj.p["b0"].value, self.obj.p["b1"].value #end0-start0, end1-start1
        
    def OptimalGamma(self):
        start = time.time()
        self.obj.p["gm"].set(vary=True)
        mi    = lmfit.minimize(self.residualSCI, self.obj.p, nan_policy='omit', method=self.method, **self.kwargs) 
        self.obj.p["gm"].set(value=mi.params["gm"].value, vary=False)
        end   = time.time()
        print("γ* =  {0:.0f}, t={1:.4f} min".format(self.obj.p["gm"].value, (end-start)/60))
        return mi.params["gm"].value
    
    def OptimalParams(self, NsGamma=51, NsPar=21):
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
                            ('b1',   1, True,  0.2,     2,  None, None),#0.04), 
                            ('b2',   0., True,    0,    2,  None, None), 
                            ('gm',   gm,  True,  1e4,  1e5, None, 2000))
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
            self.obj  = bird.Syllable(i)
            
            Gammas[i-1]   = self.OptimalGamma()#Gammas[i-1] = opt_gamma # syllable_synth.p["gm"].value
        
        self.optimal_gamma = np.mean(Gammas)
        self.obj.p["gm"].set(value=self.optimal_gamma)
        
        return Gammas