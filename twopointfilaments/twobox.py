import numpy as np
from twopointfilaments import constants, TwoPoint
import mpmath as mp
import warnings

class TBTPW:
    """
    Two-Box Two-Point model with Wall fluxes.

    Two-box extention of the two-point model including near-SOL and far-SOL,
    ballooning and fluxes to the walls.

    Input parameters:
        coll: Reference collisionality
        what: Perpendicular to parallel transport time
        gamma: Sheath heat coeffificient
        tau: Ion to electron temperature ratio. 
        ion_species: Default to deuterium.
        doL: normalized parallel ballooning length.
        lhat: far-SOL width divided by near-SOL width.
    """
    def __init__(self,coll,what,gamma=7,tau=1,ion_species='deuterium',doL=0.33,lhat=5):
        self.coll = mp.mpf(coll)
        self.xi = mp.mpf(constants.xi_factor(tau,ion_species)*coll)
        self.what = mp.mpf(what)
        self.ghat = mp.mpf(gamma/(1+tau))
        self.doL = mp.mpf(doL)
        self.lhat = mp.mpf(lhat)
        self.status = np.nan
        self.message = ''

        self.ndN=np.nan
        self.TuN=np.nan
        self.TdN=np.nan

        self.nuF=np.nan
        self.ndF=np.nan
        self.TuF=np.nan
        self.TdF=np.nan


        if what>0:
            try:
                self._calculate_near_sol()
                self._calculate_far_sol()
                self.status = 0
            except AssertionError as e:
                self.message = e.args[0][0]
                self.status = e.args[0][1]

        else:
            TP = TwoPoint(coll, gamma, tau, ion_species)
            self.nuN = 1
            self.ndN = TP.ndN
            self.TuN = TP.TuN
            self.TdN = TP.TdN
            self.status = 0

    def _calculate_near_sol(self):
        # The three near-SOL equations combined give for the upstream temperature:
        def find_TuN():
            '''
            Uses mpmath to find the root of the polynomial P for t=T_uN^0.5. 
            We find the root of P(t)/t^14 instead, since this function has a unique simple root for positive t.

            The variables z0, z1, e0, e1 are collections of model parameters as defined below.
            '''
            # Define constants
            Z0=2.5*self.what*(1+self.lhat)*self.doL*self.xi
            Z1=(2/(self.ghat*self.xi))**7
            E0=1-mp.exp(-1/self.doL)
            E1=1-self.doL*(1-mp.exp(-1/self.doL))

            f = lambda t:t**7-Z1*(t**(-2)-Z0*E0)**7+Z0*E1*t**2-1
            df = lambda t: 7*t**6+14*Z1*t**(-3)*(t**(-2)-Z0*E0)**6+2*Z0*E1*t
            d2f = lambda t: 42*t**5-42*Z1*t**(-4)*(5*t**(-2)-Z0*E0)(t**(-2)-Z0*E0)**5+2*Z0*E1
            try:
                res = mp.findroot(f,1,solver='halley',df=df, d2f=d2f, tol=1e-6, maxsteps=1000)
                return res**2
            except ValueError:
                assert False, ('Root not numerically verified',-3)

        self.TuN = find_TuN()
        assert (1-2.5*self.what*(1+self.lhat)*self.doL*self.xi*self.TuN*(1-mp.exp(-1/self.doL))>0), ('Near-SOL sheath dissipation is negative.', -2)
        self.TdN = (2*(1-2.5*self.what*(1+self.lhat)*self.doL*self.xi*self.TuN*(1-mp.exp(-1/self.doL)))/(self.ghat*self.xi*self.TuN))**(2) 
        self.ndN = 0.5*self.TuN/self.TdN
        self.nuN = 1.

    def _calculate_far_sol(self):
        def find_TdF():
            """
            See find_TuN. Same principle.
            """
            # Define constants
            A=5*self.what*(1+1./self.lhat)*self.doL*(1-mp.exp(-1./self.doL))/self.ghat
            B=(5/14)*self.xi*(self.ghat-3)*self.what*(1+self.lhat)*self.doL**2*(1-mp.exp(-1./self.doL))**2*mp.exp(1./self.doL)*self.TuN/self.ghat

            f = lambda t:t**7 + A*t**6-B
            df = lambda t: 7*t**6 + 6*A*t**5
            d2f = lambda t: 42*t**5 + 30*A*t**4
            try:
                res = mp.findroot(f,1,solver='halley',df=df, d2f=d2f, tol=1e-6, maxsteps=1000)
                return res**2
            except ValueError:
                assert False, ('Root not numerically verified',-3)
        
        self.TdF = find_TdF()
        self.ndF = 7*mp.exp(-1./self.doL)*self.TdF**2/( self.xi*(self.ghat-3)*self.doL*(1-mp.exp(-1./self.doL)) )
        self.nuF = (1.+self.lhat- self.ndF*self.TdF**0.5/(self.what*self.doL*(1.-mp.exp(-1./self.doL))))/(1+1./self.lhat)
        assert (self.nuF>0), ('Far-SOL upstream density is negative.',-4)
        assert (self.nuF<self.lhat), ('Far-SOL particle source is negative.',-5)
        assert (self.nuF<1), ('Far-SOL upstream density is higher than near-SOL', -6)
        self.TuF = 2*self.ndF*self.TdF/self.nuF
        assert (self.TuN>self.nuF*self.TuF/self.lhat), ('Far-SOL heat source is negative.',-7)
