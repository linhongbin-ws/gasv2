class TSmooth():
    def __init__(self, alpha=0.9):
        self.T_prv = None
        self.alpha = alpha
        assert alpha>=0 and alpha <=1
    def smooth(self, T):
        if self.T_prv is None:
            return T
        else:
            p = T[0:3,3]
            p_prv = self.T_prv[0:3, 3]
            out = p*self.alpha + p_prv*(1-self.alpha)
            T_out = T.copy()
            T_out[0:3, 3] = out
            self.T_prv = T_out
            return T_out
