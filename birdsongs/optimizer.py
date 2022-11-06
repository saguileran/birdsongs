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
        return syllable_synth.scoreSCI + syllable_synth.scoreSCI
    # syllable_synth.scoreSxx + syllable_synth.scoreCentroid
    

    
    def OptimalBs(self, synth):
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
    
    def OptimalParams(self, NsGamma=51, NsPar=21):
        self.obj.Solve(self.obj.p)  # solve first syllable
        
        self.kwargs["Ns"] = NsGamma;   self.obj.OptimalGamma()
        self.kwargs["Ns"] = NsPar;     self.obj.OptimalBs()
        self.obj.WriteAudio()
    
    # Solve the minimization problem at once
    def CompleteSolution(self):
        gm = self.opt_gamma
        start = time.time()
        
        self.obj.p = lmfit.Parameters()
        # add params:   (NAME   VALUE    VARY    MIN  MAX  EXPR BRUTE_STEP)
        self.obj.p.add_many(('a0',   0.11, True ,   0, 0.25,  None, None),#0.01), 
                            ('a1',   0.05, True,   -2,    2,  None, None),#0.1),  
                            ('b0',   -0.1, True,   -1,  0.5,  None, None),#0.03),  
                            ('b1',      1, True,  0.2,    2,  None, None),#0.04), 
                            ('gamma', gm,  False,  1e4,  1e5, None, 1000),
                            ('b2',     0., False, None, None, None, None), 
                            ('a2',     0., False, None, None, None, None))
        mi    = lmfit.minimize(self.residualFFandSCI, self.obj.p, nan_policy='omit', method=self.method, **self.kwargs) 
        self.obj.p["a0"].set(   vary=False, value=mi.params["a0"].value)
        self.obj.p["a1"].set(   vary=False, value=mi.params["a1"].value)
        self.obj.p["b0"].set(   vary=False, value=mi.params["b0"].value)
        self.obj.p["b1"].set(   vary=False, value=mi.params["b1"].value)
        self.obj.p["gamma"].set(vary=False, value=mi.params["gamma"].value)
        
        self.obj.Solve(self.obj.p)
        end = time.time()
        print("Time of execution = {0:.4f}".format(end-start))
        
        # ----------- OPTIMIZATION FUNCTIONS --------------
    def OptimalGamma(self):
        start = time.time()
        self.obj.p["gamma"].set(vary=True)
        mi    = lmfit.minimize(self.residualSCI, self.obj.p, nan_policy='omit', method=self.method, **self.kwargs) 
        self.obj.p["gamma"].set(value=mi.params["gamma"].value, vary=False)
        end   = time.time()
        print("γ* =  {0:.0f}, t={1:.4f} min".format(self.obj.p["gamma"].value, (end-start)/60))
        return mi.params["gamma"].value
        
    def AllGammas(self, bird):
        Gammas = np.zeros(bird.no_syllables)
        for i in range(1,bird.no_syllables+1):
            self.obj  = bird.Syllable(i)
            
            opt_gamma = self.OptimalGamma()
            Gammas[i-1]    = opt_gamma # syllable_synth.p["gamma"].value
        
        self.optimal_gamma = np.mean(Gammas)
        
        return Gammas