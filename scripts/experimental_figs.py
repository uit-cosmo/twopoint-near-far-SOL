import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import cosmoplots
plt.style.use(["cosmoplots.default"])
from twopointfilaments import xi_factor
from scipy.integrate import quad

dataloc='/home/ath019/Documents/manuscripts/two-point/code/data/'
saveloc='/home/ath019/Documents/manuscripts/two-point/code/figures/'

# Linestyle for plotting wrt what
linestyle = ['--', ':']
symbstyle = ['v','^']

class Plot:
    """
    Container class to make plots. 
    *_values indicates which values (of the full scans) to use.
    The full set of coll-values is always assumed, although plotting may truncate.

    Although the computation is done with respect to w_hat = v_perp L_par/ (c_{s,ref} lambda), 
    the what_values used here are 
    the plotting is done with respect
    """
    def __init__(self,lhat_vals, doL_vals,
                 gamma=7,tau=1,ion_species='deuterium',
                 ):
        """
        lhat_vals and doL_vals are lists of indices.
        """
        self.xi_factor = xi_factor(tau,ion_species)
        self.ghat = gamma/(1+tau)

        self.standard_twopoint = np.load(dataloc+"TwoPoint.npz")
        self.scan = np.load(dataloc+"TBTPW_scan.npz")
        
        self.coll = self.scan['coll']
        self.what = self.scan['what']
        self.lhat = self.scan['lhat']
        self.doL = self.scan['doL']

        self.ndN = self.scan['ndN']
        self.TuN = self.scan['TuN']
        self.TdN = self.scan['TdN']
        self.nuF = self.scan['nuF']
        self.ndF = self.scan['ndF']
        self.TuF = self.scan['TuF']
        self.TdF = self.scan['TdF']

        self.P = self.scan['P']

        self._select_scan_indices(lhat_vals, doL_vals)

    def _select_scan_indices(self, lhat_vals, doL_vals):
        lhat_ind = [np.argmin(np.abs(self.lhat-l)) for l in lhat_vals]
        doL_ind = [np.argmin(np.abs(self.doL-d)) for d in doL_vals]
        self.lhat = self.lhat[lhat_ind]
        self.doL = self.doL[doL_ind]
        
        attributes = [
            "ndN", "TuN", "TdN", "nuF",
            "ndF", "TuF", "TdF", "P",
        ]

        for name in attributes:
            d = getattr(self, name)
            d = np.take(d, lhat_ind, axis=2)
            d = np.take(d, doL_ind, axis=3)
            setattr(self, name, d)

    def find_what_uF_indices(self, 
                          what_vals = None, what_err = 2e-2):
        # Find combined nuref, what_ref indices which correspond to what_uf in what_vals, with 
        # error as a percentage set by what_err.

        # Returns an array of the same shape as ndN, TdN etc of booleans
        C, W = np.meshgrid(self.coll, self.what, indexing='ij')
        
        self.what_ind = np.array([np.argmin(np.abs(self.what-w)) for w in what_vals])
        self.what_uF_good_ind = {key:np.zeros_like(self.scan['P'], dtype=bool) for key in self.what_ind}
        self.what_uF = np.zeros_like(self.scan['P'])


        for li in range(len(self.lhat)):
            for dj in range(len(self.doL)):
                self.what_uF[:,:,li,dj] = W/self.TuF[:,:,li,dj]**0.5
                
                for wk in self.what_ind:
                    self.what_uF_good_ind[wk][:,:,li,dj] = (
                                                            (self.what_uF[:,:,li,dj]>(1-what_err)*self.what[wk])
                                                            &(self.what_uF[:,:,li,dj]<(1+what_err)*self.what[wk])
                                                            )
        
        self.W = lambda wk,li,dj: W[self.what_uF_good_ind[wk][:,:,li,dj]]
        self.C = lambda wk,li,dj: C[self.what_uF_good_ind[wk][:,:,li,dj]]

    def _get_plot_params(self, what_uF):
        if what_uF == True:
            good = lambda data, wk, li, dj: data[:,:,li,dj][self.what_uF_good_ind[wk][:,:,li,dj]]
            #C = lambda wk, li, dj: np.stack((self.coll,)*len(self.what),axis=1)[self.what_uF_good_ind[wk][:,:,li,dj]]
            C = self.C
            def plot(Ax, X, Y, col, styleind, label=None):
                Ax.plot(X, Y, c=col, ls=linestyle[styleind], label=label)
            fname = '_what_uF'
        else:
            C = lambda wk, li, dj: self.coll
            good = lambda data, wk, li, dj: data[:,wk,li,dj]
            def plot(Ax, X, Y, col, styleind, label=None):
                Ax.plot(X, Y, c=col, ls=linestyle[styleind], label=label)
            fname = ''

        return C, good, plot, fname

    def plot_good_params(self, truncated=True):   
        X, Y = np.meshgrid(np.log10(self.scan['coll']), 
                           np.log10(self.scan['what']), indexing='ij')
        
        if truncated:
            lhat = self.lhat
            doL = self.doL
            P = self.P
            fname = '_trunc'
        else:
            lhat = self.scan['lhat']
            doL = self.scan['doL']
            P = self.scan['P']
            fname = ''

        fig, ax = cosmoplots.figure_multiple_rows_columns(len(doL),len(lhat))
        for li in range(len(lhat)):
            for dj in range(len(doL)):
                ind = li + len(lhat)*dj
                ax[ind].contourf(X, Y, P[:,:,li,dj], cmap='cividis', vmin=-7,vmax=0, levels=8)
                #fig.colorbar(plot, ax=ax[ind])
                ax[ind].text(0.62, 0.1, rf'$\hat{{\lambda}}={lhat[li]},\, \hat{{\delta}}={doL[dj]}$', bbox={'facecolor':'white'},
                             transform = ax[ind].transAxes)
                if li == 0:
                    ax[ind].set_ylabel(r'$\log_{10} \hat{\omega}$')
                if dj == len(doL)-1:
                    ax[ind].set_xlabel(r'$\log_{10} \nu_{\mathrm{ref}}$')

                if truncated:
                    ax[ind].text(0.1, 0.1, r'$\mathrm{I}$', c='white', transform = ax[ind].transAxes)
                    ax[ind].text(0.9, 0.65, r'$\mathrm{IV}$', transform = ax[ind].transAxes)
                    ax[ind].text(0.9, 0.9, r'$\mathrm{V}$', transform = ax[ind].transAxes)
                    if li==0:
                        ax[ind].text(0.6, 0.6, r'$\mathrm{II}$',transform = ax[ind].transAxes)
                    if ind==1:
                        ax[ind].text(0.65, 0.35, r'$\mathrm{II}$',transform = ax[ind].transAxes)
                        ax[ind].text(0.65, 0.55, r'$\mathrm{III}$', c = 'white',transform = ax[ind].transAxes)
                        for jj,symb in zip([100,280,350,420,500],
                                           ['o','d','*','<','>']):
                            ax[ind].scatter(X[jj,jj], Y[jj,jj],marker=symb,s=10)
                            print(f'lognu={np.log10(self.scan['coll'][jj])}, logw={np.log10(self.scan['what'][jj])}, P={P[jj,jj,li,dj]}')
                    if ind==3:
                        ax[ind].text(0.7, 0.5, r'$\mathrm{II}$',transform = ax[ind].transAxes)
                        ax[ind].text(0.05, 0.9, r'$\mathrm{III}$',c='white',transform = ax[ind].transAxes)

        fig.savefig(saveloc+'good_params'+fname+'.eps')

    def plot_good_params_what_uF(self):   
        X, Y = np.meshgrid(np.log10(self.scan['coll']), 
                           np.log10(self.scan['what']), indexing='ij')
        
        fig, ax = cosmoplots.figure_multiple_rows_columns(len(self.doL),len(self.lhat))
        for li in range(len(self.lhat)):
            for dj in range(len(self.doL)):
                ind = li + len(self.lhat)*dj
                ax[ind].pcolor(X, np.log10(self.what_uF[:,:,li,dj]), np.ones_like(X), cmap = 'cividis', vmin=0, vmax=1)
                ax[ind].text(0.62, 0.1, rf'$\hat{{\lambda}}={self.lhat[li]},\, \hat{{\delta}}={self.doL[dj]}$', bbox={'facecolor':'white'},
                             transform = ax[ind].transAxes)
                for kk, wj in enumerate(self.what_ind):
                    ax[ind].axhline(np.log10(self.what[wj]), c='k', ls = linestyle[kk])
                if li == 0:
                    ax[ind].set_ylabel(r'$\log_{10} \hat{\omega}$')
                if dj == len(self.doL)-1:
                    ax[ind].set_xlabel(r'$\log_{10} \nu_{\mathrm{ref}}$')
                
                #ax[ind].set_xlim(0,2)
                ax[ind].set_ylim(-2,2)
        fig.savefig(saveloc+'good_params_ofar.eps')

    def plot_near_SOL(self, what_uF=False):
        fig, ax = cosmoplots.figure_multiple_rows_columns(len(self.doL),len(self.lhat))
        C, good, plot, fname = self._get_plot_params(what_uF)
        for li in range(len(self.lhat)):
            for dj in range(len(self.doL)):
                ind = li + len(self.lhat)*dj
                cosmoplots.change_log_axis_base(ax[ind],'y')
                ax[ind].plot(self.coll, self.standard_twopoint['ndN'], 'C0', label = r'$n_{d,N}/n_\mathrm{u,N}$')
                ax[ind].plot(self.coll, self.standard_twopoint['TdN']/self.standard_twopoint['TuN'], 'C1', label = r'$T_{d,N}/T_{u,N}$')
                ax[ind].plot(self.coll, self.standard_twopoint['TuN'], 'C2', label = r'$T_{u,N}/T_\mathrm{ref}$')

                for styleind, wk in enumerate(self.what_ind):
                    plot(ax[ind], C(wk,li,dj), good(self.ndN,wk,li,dj), 'C0', styleind)
                    plot(ax[ind], C(wk,li,dj), good(self.TdN/self.TuN,wk,li,dj), 'C1', styleind)
                    plot(ax[ind], C(wk,li,dj), good(self.TuN,wk,li,dj), 'C2', styleind)

                    ax[ind].text(0.62, 0.1, rf'$\hat{{\lambda}}={self.lhat[li]},\, \hat{{\delta}}={self.doL[dj]}$', bbox={'facecolor':'white'},
                                 transform = ax[ind].transAxes)

                    if dj == len(self.doL)-1:
                        ax[ind].set_xlabel(r'$\nu_{\mathrm{ref}}$')

                    ax[ind].set_xlim(self.coll[0],self.coll[-1])
                    ax[ind].set_ylim(1e-3,1e3)
                ax[ind].legend()
        fig.savefig(saveloc+'near'+fname+'.eps')

    def plot_far_SOL(self, what_uF=False):
        fig, ax = cosmoplots.figure_multiple_rows_columns(len(self.doL),len(self.lhat))
        C, good, plot, fname = self._get_plot_params(what_uF)
        for li in range(len(self.lhat)):
            for dj in range(len(self.doL)):
                ind = li + len(self.lhat)*dj
                cosmoplots.change_log_axis_base(ax[ind],'y')
                for styleind, wk in enumerate(self.what_ind):
                    plot(ax[ind], C(wk,li,dj), (good(self.ndF/self.nuF,wk,li,dj)), 'C0',styleind)
                    plot(ax[ind], C(wk,li,dj), (good(self.TdF/self.TuF,wk,li,dj)), 'C1',styleind)
                    #plt.plot(Data['coll'], Data['TuF']/Data['TuN'], 'C2'+ls)
                    #plt.plot(Data['coll'], Data['nuF'], 'C3'+ls)

                    ax[ind].text(0.62, 0.1, rf'$\hat{{\lambda}}={self.lhat[li]},\, \hat{{\delta}}={self.doL[dj]}$', bbox={'facecolor':'white'},
                                 transform = ax[ind].transAxes)

                    if dj == len(self.doL)-1:
                        ax[ind].set_xlabel(r'$\nu_{\mathrm{ref}}$')

                    ax[ind].set_xlim(self.coll[0],self.coll[-1])
                    #ax[ind].set_ylim(0,10)
                lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$n_\mathrm{d,F}/n_\mathrm{u,F}$'),
                         mlines.Line2D([], [], color='C1', ls = '-', label=r'$T_\mathrm{d,F}/T_\mathrm{u,F}$')]
                         #mlines.Line2D([], [], color='C2', ls = '-', label=r'$T_\mathrm{u,F}/T_\mathrm{u,N}$'),
                         #mlines.Line2D([], [], color='C3', ls = '-', label=r'$n_\mathrm{u,F}/n_\mathrm{u,N}$')]
                ax[ind].legend(handles=lines)

        fig.savefig(saveloc+'far'+fname+'.eps')

    def plot_pressure(self, what_uF=False):
        fig, ax = cosmoplots.figure_multiple_rows_columns(len(self.doL),len(self.lhat))
        C, good, plot, fname = self._get_plot_params(what_uF)
        for li in range(len(self.lhat)):
            for dj in range(len(self.doL)):
                ind = li + len(self.lhat)*dj
                cosmoplots.change_log_axis_base(ax[ind],'y')
                for styleind, wk in enumerate(self.what_ind):
                    plot(ax[ind], C(wk,li,dj), (good(self.TuN,wk,li,dj)), 'C0',styleind)
                    plot(ax[ind], C(wk,li,dj), (good(self.nuF*self.TuF,wk,li,dj)), 'C1',styleind)
                    #plt.plot(Data['coll'], Data['TuF']/Data['TuN'], 'C2'+ls)
                    #plt.plot(Data['coll'], Data['nuF'], 'C3'+ls)

                    ax[ind].text(0.62, 0.1, rf'$\hat{{\lambda}}={self.lhat[li]},\, \hat{{\delta}}={self.doL[dj]}$', bbox={'facecolor':'white'},
                                 transform = ax[ind].transAxes)

                    if dj == len(self.doL)-1:
                        ax[ind].set_xlabel(r'$\nu_{\mathrm{ref}}$')

                    ax[ind].set_xlim(self.coll[0],self.coll[-1])
                    #ax[ind].set_ylim(0,10)
                lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$n_\mathrm{u,N} T_\mathrm{u,N}$'),
                         mlines.Line2D([], [], color='C1', ls = '-', label=r'$n_\mathrm{u,F} T_\mathrm{u,F}$')]
                         #mlines.Line2D([], [], color='C2', ls = '-', label=r'$T_\mathrm{u,F}/T_\mathrm{u,N}$'),
                         #mlines.Line2D([], [], color='C3', ls = '-', label=r'$n_\mathrm{u,F}/n_\mathrm{u,N}$')]
                ax[ind].legend(handles=lines)

        fig.savefig(saveloc+'pressure'+fname+'.eps')

    def plot_coll(self,what_uF=False):
        fig, ax = cosmoplots.figure_multiple_rows_columns(len(self.doL),len(self.lhat))
        lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$\mathrm{u,N}$'),
                 mlines.Line2D([], [], color='C1', ls = '-', label=r'$\mathrm{d,N}$'),
                 mlines.Line2D([], [], color='C2', ls = '-', label=r'$\mathrm{u,F}$'),
                 mlines.Line2D([], [], color='C3', ls = '-', label=r'$\mathrm{d,F}$'),
                 mlines.Line2D([], [], color='k', ls = '-.', label=r'$\mathrm{ref}$')]
        C, good, plot, fname = self._get_plot_params(what_uF)
        for li in range(len(self.lhat)):
            for dj in range(len(self.doL)):
                ind = li + len(self.lhat)*dj
                cosmoplots.change_log_axis_base(ax[ind],'y')
                ax[ind].plot(self.coll, self.coll, 'k')
                ax[ind].plot(self.coll, self.coll/self.standard_twopoint['TuN']**2, 'C0')
                ax[ind].plot(self.coll, self.coll*self.standard_twopoint['ndN']/self.standard_twopoint['TdN']**2, 'C1')
                for styleind, wk in enumerate(self.what_ind):
                    plot(ax[ind], C(wk,li,dj), (C(wk,li,dj)*good(1./self.TuN**2,wk,li,dj)), 'C0',styleind)
                    plot(ax[ind], C(wk,li,dj), (C(wk,li,dj)*good(self.ndN/self.TdN**2,wk,li,dj)), 'C1',styleind)
                    plot(ax[ind], C(wk,li,dj), (C(wk,li,dj)*good(self.nuF/self.TuF**2,wk,li,dj)), 'C2',styleind)
                    plot(ax[ind], C(wk,li,dj), (C(wk,li,dj)*good(self.ndF/self.TdF**2,wk,li,dj)), 'C3',styleind)

                ax[ind].text(0.62, 0.1, rf'$\hat{{\lambda}}={self.lhat[li]},\, \hat{{\delta}}={self.doL[dj]}$', bbox={'facecolor':'white'},
                             transform = ax[ind].transAxes)

                ax[ind].set_xlabel(r'$\nu_{\mathrm{ref}}$')
                ax[ind].set_ylabel(r'$\nu$')

                ax[ind].set_xlim(self.coll[0],self.coll[-1])
                ax[ind].set_ylim(1e-3,1e3)
                ax[ind].legend(handles=lines)

        fig.savefig(saveloc+'coll'+fname+'.eps')

    def plot_fluxes(self, what_uF=False):
        fig, ax = cosmoplots.figure_multiple_rows_columns(len(self.doL),len(self.lhat))
        C, good, plot, fname = self._get_plot_params(what_uF)
        for li in range(len(self.lhat)):
            for dj in range(len(self.doL)):
                ind = li + len(self.lhat)*dj
                cosmoplots.change_log_axis_base(ax[ind],'y')
                for styleind, wk in enumerate(self.what_ind):
                    plot(ax[ind], C(wk,li,dj), (1+self.lhat[li]**(-1))*self.W(wk,li,dj)*(good(self.nuF,wk,li,dj)), 'C0',styleind)
                    plot(ax[ind], C(wk,li,dj), 2.5*(1+self.lhat[li]**(-1))*self.W(wk,li,dj)*(self.xi_factor*C(wk,li,dj)*good(self.nuF*self.TuF,wk,li,dj)), 'C1',styleind)
                    plot(ax[ind], C(wk,li,dj), self.ghat*self.xi_factor*C(wk,li,dj)*good(self.ndF*self.TdF**1.5,wk,li,dj), 'C2',styleind)
                    plot(ax[ind], C(wk,li,dj), self.ghat*self.xi_factor*C(wk,li,dj)*good(self.ndN*self.TdN**1.5,wk,li,dj), 'C3',styleind)

                    ax[ind].text(0.62, 0.1, rf'$\hat{{\lambda}}={self.lhat[li]},\, \hat{{\delta}}={self.doL[dj]}$', bbox={'facecolor':'white'},
                                 transform = ax[ind].transAxes)

                    if dj == len(self.doL)-1:
                        ax[ind].set_xlabel(r'$\nu_{\mathrm{ref}}$')

                    ax[ind].set_xlim(self.coll[0],self.coll[-1])
                    ax[ind].set_ylim(1e-2,10)
                lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$\Gamma_{\perp,F}(0)$'),
                         mlines.Line2D([], [], color='C1', ls = '-', label=r'$q_{\perp,F}(0)$'),
                         mlines.Line2D([], [], color='C2', ls = '-', label=r'$q_{\parallel,F}(1)$'),
                         mlines.Line2D([], [], color='C3', ls = '-', label=r'$q_{\parallel,N}(1)$')]
                ax[ind].legend(handles=lines)

        fig.savefig(saveloc+'fluxes'+fname+'.eps')

    def plot_TdF_vs_TuN(self, what_uF=False):
        fig, ax = cosmoplots.figure_multiple_rows_columns(len(self.doL),len(self.lhat))
        C, good, plot, fname = self._get_plot_params(what_uF)
        nuref = 10**np.linspace(-1,2,20)
        for li in range(len(self.lhat)):
            for dj in range(len(self.doL)):
                ind = li + len(self.lhat)*dj
                cosmoplots.change_log_axis_base(ax[ind],'both')
                for styleind, wk in enumerate(self.what_ind):
                    plot(ax[ind], C(wk,li,dj),(good(self.TdF/self.TuN,wk,li,dj)), 'C0',styleind)
                    plot(ax[ind], C(wk,li,dj),(good(self.TdF,wk,li,dj)), 'C1',styleind)
                plot(ax[ind], nuref[nuref<5], 0.2*nuref[nuref<5]**(0.6), 'k',styleind)
                plot(ax[ind], nuref[nuref>10], 0.4*nuref[nuref>10]**(0.2), 'k',styleind)
                plot(ax[ind], nuref[nuref>10], 0.3*nuref[nuref>10]**(0.4), 'k',styleind)
                plot(ax[ind], nuref, 0.5*nuref**(2/21), 'grey',styleind)
                    #plot(ax[ind], C(wk,li,dj), (good(self.TdF,wk,li,dj)), 'C1',styleind)
                    #plt.axhline(2.5/self.ghat, c='k', ls='-.')
                ax[ind].text(0.2, 0.4, r'$\nu_\mathrm{ref}^{0.6}$', transform = ax[ind].transAxes)
                ax[ind].text(0.8, 0.6, r'$\nu_\mathrm{ref}^{0.2}$', transform = ax[ind].transAxes)
                ax[ind].text(0.6, 0.8, r'$\nu_\mathrm{ref}^{0.4}$', transform = ax[ind].transAxes)

                ax[ind].set_xlabel(r'$\nu_{\mathrm{ref}}$')
                ax[ind].set_ylabel(r'$T_\mathrm{d,F}/T_\mathrm{u,N}$')

                    #ax[ind].set_xlim(self.coll[0],self.coll[-1])
                    #ax[ind].set_ylim()
                lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$T_\mathrm{d,F}/T_\mathrm{u,N}$'),
                         mlines.Line2D([], [], color='C1', ls = '-', label=r'$T_\mathrm{d,F}/T_\mathrm{ref}$'),
                         mlines.Line2D([], [], color='k', ls = '-.', label=r'$5 /(2 \hat{\gamma}\xi)$')]
                #ax[ind].legend(handles=lines)

        fig.savefig(saveloc+'TdF_v_TuN'+fname+'.eps')

    def plot_nvt(self, what_uF=False):
        fig, ax = plt.subplots()
        C, good, plot, fname = self._get_plot_params(what_uF)
        n = 10**np.linspace(-1,5,100)
        ax.loglog(n, n**(-0.68),'k-.')
        for li in range(len(self.lhat)):
            for dj in range(len(self.doL)):
                ind = li + len(self.lhat)*dj
                for styleind, wk in enumerate(self.what_ind):
                    plot(ax, (C(wk,li,dj)*good(self.ndN,wk,li,dj)), (good(self.TdN,wk,li,dj)), f'C{ind}',styleind)
                    plot(ax, (C(wk,li,dj)*good(self.ndN,wk,li,dj)), (good(self.TdN,wk,li,dj)), f'C{ind}',styleind)
                    #plot(ax[ind], C(wk,li,dj), (good(self.TdF,wk,li,dj)), 'C1',styleind)
                    #plt.axhline(2.5/self.ghat, c='k', ls='-.')
        ax.set_xlabel(r'$\nu_\mathrm{ref} n_\mathrm{d,N}$')
        ax.set_ylabel(r'$T_\mathrm{d,N}$')

                    #ax[ind].set_xlim(self.coll[0],self.coll[-1])
                    #ax[ind].set_ylim()

        fig.savefig(saveloc+'nvt'+fname+'.eps')



def plot_comparison_all(what_vals, lhat_vals, doL_vals):
    P = Plot(lhat_vals=lhat_vals, doL_vals=doL_vals)
    P.find_what_uF_indices(what_vals=what_vals)
    #P.plot_good_params_what_uF()
    #P.plot_good_params(truncated=False)
    #P.plot_good_params(truncated=True)
    #P.plot_coll(what_uF=False)
    #P.plot_coll(what_uF=True)
    #P.compare_what_ref_uF_near()
    #P.plot_far_SOL(what_uF=True)
    P.plot_near_SOL(what_uF=True)

def plot_full_figs(what_vals, lhat_vals, doL_vals):
    P = Plot(lhat_vals=lhat_vals, doL_vals=doL_vals)
    P.find_what_uF_indices(what_vals=what_vals)
    if False:
        P.plot_near_SOL(what_uF=True)
        P.plot_far_SOL(what_uF=True)
        P.plot_coll(what_uF=True)
        P.plot_pressure(what_uF=True)
        P.plot_fluxes(what_uF=True)
        P.plot_nvt(what_uF=True)
    P.plot_TdF_vs_TuN(what_uF=True)

if __name__=="__main__":
    what_vals = [0.2, 1.]
    lhat_vals = [1., 2.]
    doL_vals = [0.1,0.2]
    #plot_comparison_all(what_vals,lhat_vals,doL_vals)
    plot_full_figs(what_vals,lhat_vals[:1],doL_vals[-1:])
    #plot_full_figs(what_vals,lhat_vals,doL_vals)
