from .syllable import *
from .functions import *

class Solution(Syllable):
    def __init__(self, obj):
        self.obj       = obj
        #self.obj_synth = obj_synth
        
    def residualSCI(self, p):
        syllable_synth = self.obj.Solve(p)
        return syllable_synth.scoreSCI
    
    def residualFF(self, p):
        syllable_synth = self.obj.Solve(p)
        return syllable_synth.scoreFF
    
    def residualIndexes(self, p):
        syllable_synth = self.obj.Solve(p)
        return syllable_synth.scoreSCI+ syllable_synth.scoreSCI
    # syllable_synth.scoreSxx + syllable_synth.scoreMel
    
    
    # ----------- OPTIMIZATION FUNCTIONS --------------
    def OptimalGamma(self, method_kwargs):
        kwargs = {k: method_kwargs[k] for k in set(list(method_kwargs.keys())) - set(["method"])}
    
        start = time.time()
        self.p["gamma"].set(vary=True)
        mi    = lmfit.minimize(self.residualSCI, self.p, nan_policy='omit', method=method_kwargs["method"], **kwargs) 
        self.p["gamma"].set(value=mi.params["gamma"].value, vary=False)
        end   = time.time()
        print("γ* =  {0:.0f}, t={1:.4f} min".format(self.p["gamma"].value, (end-start)/60))
        return mi.params["gamma"].value
    
    def OptimalBs(self, method_kwargs):
        kwargs = {k: method_kwargs[k] for k in set(list(method_kwargs.keys())) - set(["method"])}
        # ---------------- b0--------------------
        start0 = time.time()
        self.p["b0"].set(vary=True)
        mi0    = lmfit.minimize(self.residualFF, self.p, nan_policy='omit', method=method_kwargs["method"], **kwargs) 
        self.p["b0"].set(vary=False, value=mi0.params["b0"].value)
        end0   = time.time()
        print("b_0*={0:.4f}, t={1:.4f} min".format(self.p["b0"].value, (end0-start0)/60))
        # ---------------- b1--------------------
        start1 = time.time()
        self.p["b1"].set(vary=True)
        mi1    = lmfit.minimize(self.residualFF, self.p, nan_policy='omit', method=method_kwargs["method"], **kwargs) 
        self.p["b1"].set(vary=False, value=mi1.params["b1"].value)
        end1   = time.time()
        print("b_1*={0:.4f}, t={1:.4f} min".format(self.p["b1"].value, (end1-start1)/60))
        #return self.p["b0"].value, self.p["b1"].value #end0-start0, end1-start1
    
    def OptimalParams(self, method_kwargs):
        self.Solve(self.p)  # solve first syllable
        
        kwargs["Ns"] = 51;   self.OptimalGamma(method_kwargs)
        kwargs["Ns"] = 21;   self.OptimalBs(method_kwargs)
        self.WriteAudio()
    
    # Solve the minimization problem at once
    def CompleteSolution(self, opt_gamma, kwargs):
        start = time.time()
        # add params:   (NAME   VALUE    VARY    MIN  MAX  EXPR BRUTE_STEP)
        self.p.add_many(('a0',   0.11,         True ,   0, 0.25,  None, 0.01), 
                        ('a1',   0.05,         True,   -2,    2,  None, 0.1),  
                        ('b0',   -0.1,         True,   -1,  0.5,  None, 0.03),  
                        ('b1',      1,         True,  0.2,    2,  None, 0.04), 
                        ('gamma', opt_gamma,   False,  1e4,  1e5, None, 1000),
                        ('b2',     0.,         False, None, None, None, None), 
                        ('a2',     0.,         False, None, None, None, None))
        mi    = lmfit.minimize(self.residualFFandSCI, self.p, nan_policy='omit', **kwargs) 
        self.p["a0"].set(   vary=False, value=mi.params["a0"].value)
        self.p["a1"].set(   vary=False, value=mi.params["a1"].value)
        self.p["b0"].set(   vary=False, value=mi.params["b0"].value)
        self.p["b1"].set(   vary=False, value=mi.params["b1"].value)
        self.p["gamma"].set(vary=False, value=mi.params["gamma"].value)
        
        self.Solve(self.p)
        end = time.time()
        
        print("Time of execution = {0:.4f}".format(end-start))