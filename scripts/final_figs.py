import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import cosmoplots
import cmasher as cmr
plt.style.use(["cosmoplots.default"])
from twopointfilaments import xi_factor

dataloc='/home/ath019/Documents/manuscripts/two-point/code/data/'
saveloc='/home/ath019/Documents/manuscripts/two-point/code/figures/'

# Linestyle for plotting wrt what
linestyle = ['--', ':','-']
symbstyle = ['v','^','<','>']
lhat_name = {1.0: '1', 2.0: '2'}
dhat_name = {0.1: '1/10', 0.2: '1/5'}

def cmap(n,M=4):
    # set the default M to the number of lines to plot
    tmp = cmr.tropical.resampled(M)
    return tmp(n/M)

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

    def plot_good_params(self):   
        X, Y = np.meshgrid(np.log10(self.scan['coll']), 
                           np.log10(self.scan['what']), indexing='ij')
        
        lhat = self.lhat
        doL = self.doL
        P = self.P

        fig, ax = cosmoplots.figure_multiple_rows_columns(len(doL),len(lhat))
        for li in range(len(lhat)):
            for dj in range(len(doL)):
                ind = li + len(lhat)*dj
                ax[ind].contourf(X, Y, P[:,:,li,dj], cmap='cividis', vmin=-7,vmax=0, levels=8)
                ldlab = rf'$\hat{{\lambda}}= {lhat_name[self.lhat[li]]},\, \hat{{\delta}}= {dhat_name[self.doL[dj]]}$'
                ax[ind].text(0.62, 0.1, ldlab, bbox={'facecolor':'white'},
                             transform = ax[ind].transAxes)
                if li == 0:
                    ax[ind].set_ylabel(r'$\log_{10} \hat{\omega}$')
                if dj == len(doL)-1:
                    ax[ind].set_xlabel(r'$\log_{10} \nu_{\mathrm{ref}}$')

                ax[ind].text(0.1, 0.1, r'$\mathrm{I}$', c='white', transform = ax[ind].transAxes)
                ax[ind].text(0.9, 0.65, r'$\mathrm{IV}$', transform = ax[ind].transAxes)
                ax[ind].text(0.9, 0.9, r'$\mathrm{V}$', transform = ax[ind].transAxes)
                if li==0:
                    ax[ind].text(0.6, 0.6, r'$\mathrm{II}$',transform = ax[ind].transAxes)
                if ind==1:
                    ax[ind].text(0.65, 0.35, r'$\mathrm{II}$',transform = ax[ind].transAxes)
                    ax[ind].text(0.65, 0.55, r'$\mathrm{III}$', c = 'white',transform = ax[ind].transAxes)
                if ind==3:
                    ax[ind].text(0.7, 0.5, r'$\mathrm{II}$',transform = ax[ind].transAxes)
                    ax[ind].text(0.05, 0.9, r'$\mathrm{III}$',c='white',transform = ax[ind].transAxes)

        fig.savefig(saveloc+'good_params.eps')

    def plot_near_SOL_compare(self, what_uF=True):
        fig, ax = cosmoplots.figure_multiple_rows_columns(len(self.doL),len(self.lhat))
        C, good, plot, fname = self._get_plot_params(what_uF)
        for li in range(len(self.lhat)):
            for dj in range(len(self.doL)):
                ind = li + len(self.lhat)*dj
                cosmoplots.change_log_axis_base(ax[ind],'y')
                ax[ind].plot(self.coll, self.standard_twopoint['ndN'], c=cmap(0), label = r'$n_\mathrm{d,N}/n_\mathrm{u,N}$')
                ax[ind].plot(self.coll, self.standard_twopoint['TdN']/self.standard_twopoint['TuN'], c=cmap(1), label = r'$T_\mathrm{d,N}/T_\mathrm{u,N}$')
                ax[ind].plot(self.coll, self.standard_twopoint['TuN'], c=cmap(2), label = r'$T_\mathrm{u,N}/T_\mathrm{ref}$')

                for styleind, wk in enumerate(self.what_ind):
                    plot(ax[ind], C(wk,li,dj), good(self.ndN,wk,li,dj), cmap(0), styleind)
                    plot(ax[ind], C(wk,li,dj), good(self.TdN/self.TuN,wk,li,dj), cmap(1), styleind)
                    plot(ax[ind], C(wk,li,dj), good(self.TuN,wk,li,dj), cmap(2), styleind)
                    
                ldlab = rf'$\hat{{\lambda}}= {lhat_name[self.lhat[li]]},\, \hat{{\delta}}= {dhat_name[self.doL[dj]]}$'
                ax[ind].text(0.62, 0.1, ldlab, bbox={'facecolor':'white'},
                             transform = ax[ind].transAxes)

                if dj == len(self.doL)-1:
                    ax[ind].set_xlabel(r'$\nu_{\mathrm{ref}}$')

                ax[ind].set_xlim(self.coll[0],self.coll[-1])
                ax[ind].set_ylim(1e-3,1e3)
                ax[ind].legend()
        fig.savefig(saveloc+'near'+fname+'_compare.eps')

    def plot_far_SOL_compare(self, what_uF=True):
        fig, ax = cosmoplots.figure_multiple_rows_columns(len(self.doL),len(self.lhat))
        C, good, plot, fname = self._get_plot_params(what_uF)
        for li in range(len(self.lhat)):
            for dj in range(len(self.doL)):
                ind = li + len(self.lhat)*dj
                cosmoplots.change_log_axis_base(ax[ind],'y')
                for styleind, wk in enumerate(self.what_ind):
                    plot(ax[ind], C(wk,li,dj), (good(self.ndF/self.nuF,wk,li,dj)), cmap(0),styleind)
                    plot(ax[ind], C(wk,li,dj), (good(self.TdF/self.TuF,wk,li,dj)), cmap(1),styleind)
                    #plt.plot(Data['coll'], Data['TuF']/Data['TuN'], cmap(2)+ls)
                    #plt.plot(Data['coll'], Data['nuF'], cmap(3)+ls)

                ldlab = rf'$\hat{{\lambda}}= {lhat_name[self.lhat[li]]},\, \hat{{\delta}}= {dhat_name[self.doL[dj]]}$'
                ax[ind].text(0.65, 0.1, ldlab, bbox={'facecolor':'white'},
                             transform = ax[ind].transAxes)

                if dj == len(self.doL)-1:
                    ax[ind].set_xlabel(r'$\nu_{\mathrm{ref}}$')

                ax[ind].set_xlim(self.coll[0],self.coll[-1])
                #ax[ind].set_ylim(0,10)
                lines = [mlines.Line2D([], [], color=cmap(0), ls = '-', label=r'$n_\mathrm{d,F}/n_\mathrm{u,F}$'),
                         mlines.Line2D([], [], color=cmap(1), ls = '-', label=r'$T_\mathrm{d,F}/T_\mathrm{u,F}$')]
                         #mlines.Line2D([], [], color=cmap(2), ls = '-', label=r'$T_\mathrm{u,F}/T_\mathrm{u,N}$'),
                         #mlines.Line2D([], [], color=cmap(3), ls = '-', label=r'$n_\mathrm{u,F}/n_\mathrm{u,N}$')]
                ax[ind].legend(handles=lines)

        fig.savefig(saveloc+'far'+fname+'_compare.eps')

    def plot_near_far_pressure(self, what_uF=True):
        fig, ax = cosmoplots.figure_multiple_rows_columns(3,1)
        C, good, plot, fname = self._get_plot_params(what_uF)

        for i,lab in enumerate([r'$\mathrm{Near-SOL}$',r'$\mathrm{Far-SOL}$',r'$\mathrm{Pressure}$']):
            cosmoplots.change_log_axis_base(ax[i],'y')
            ax[i].set_xlim(self.coll[0],60)
            ax[i].set_xlabel(r'$\nu_{\mathrm{ref}}$')
            ax[i].text(0.1, 0.85, lab, bbox={'facecolor':'white'},
                         transform = ax[i].transAxes)

        # Near-SOL
        ax[0].plot(self.coll, self.standard_twopoint['ndN'], c=cmap(0), label = r'$n_\mathrm{d,N}/n_\mathrm{u,N}$')
        ax[0].plot(self.coll, self.standard_twopoint['TdN']/self.standard_twopoint['TuN'], c=cmap(1), label = r'$T_\mathrm{d,N}/T_\mathrm{u,N}$')
        ax[0].plot(self.coll, self.standard_twopoint['TuN'], c=cmap(2), label = r'$T_\mathrm{u,N}/T_\mathrm{ref}$')

        for styleind, wk in enumerate(self.what_ind):
            plot(ax[0], C(wk,0,0), good(self.ndN,wk,0,0), cmap(0), styleind)
            plot(ax[0], C(wk,0,0), good(self.TdN/self.TuN,wk,0,0), cmap(1), styleind)
            plot(ax[0], C(wk,0,0), good(self.TuN,wk,0,0), cmap(2), styleind)
            
        ax[0].set_ylim(1e-3,1e3)
        ax[0].legend(loc=3)

        # Far-SOL
        for styleind, wk in enumerate(self.what_ind):
            plot(ax[1], C(wk,0,0), (good(self.ndF/self.nuF,wk,0,0)), cmap(0),styleind)
            plot(ax[1], C(wk,0,0), (good(self.TdF/self.TuF,wk,0,0)), cmap(1),styleind)
            #plot(ax[1], C(wk,0,0), (good(self.nuF,wk,0,0)), cmap(2),styleind)
            #plot(ax[1], C(wk,0,0), (good(self.TuF,wk,0,0)), cmap(3),styleind)

        lines = [mlines.Line2D([], [], color=cmap(0), ls = '-', label=r'$n_\mathrm{d,F}/n_\mathrm{u,F}$'),
                 mlines.Line2D([], [], color=cmap(1), ls = '-', label=r'$T_\mathrm{d,F}/T_\mathrm{u,F}$')]
                 #mlines.Line2D([], [], color=cmap(2), ls = '-', label=r'$n_\mathrm{u,F}/n_\mathrm{u,N}$'),
                 #mlines.Line2D([], [], color=cmap(3), ls = '-', label=r'$T_\mathrm{d,F}/T_\mathrm{ref}$')]
        ax[1].legend(handles=lines)
        ax[1].set_ylim(1e-3,1e3)

        # Pressure
        for styleind, wk in enumerate(self.what_ind):
            plot(ax[2], C(wk,0,0), (good(self.TuN,wk,0,0)), cmap(0),styleind)
            plot(ax[2], C(wk,0,0), (good(self.nuF*self.TuF,wk,0,0)), cmap(1),styleind)

        ax[2].set_xlabel(r'$\nu_{\mathrm{ref}}$')
        ax[2].set_ylabel(r'$n T / n_\mathrm{u,N} T_\mathrm{ref}$')

        lines = [mlines.Line2D([], [], color=cmap(0), ls = '-', label=r'$n_\mathrm{u,N} T_\mathrm{u,N}$'),
                 mlines.Line2D([], [], color=cmap(1), ls = '-', label=r'$n_\mathrm{u,F} T_\mathrm{u,F}$')]
                 #mlines.Line2D([], [], color=cmap(2), ls = '-', label=r'$T_\mathrm{u,F}/T_\mathrm{u,N}$'),
                 #mlines.Line2D([], [], color=cmap(3), ls = '-', label=r'$n_\mathrm{u,F}/n_\mathrm{u,N}$')]
        ax[2].legend(handles=lines)

        fig.savefig(saveloc+'near_far_p'+fname+'.eps')

    def plot_coll(self,what_uF=True):
        fig, ax = plt.subplots()
        lines = [mlines.Line2D([], [], color=cmap(0), ls = '-', label=r'$\mathrm{u,N}$'),
                 mlines.Line2D([], [], color=cmap(1), ls = '-', label=r'$\mathrm{d,N}$'),
                 mlines.Line2D([], [], color=cmap(2), ls = '-', label=r'$\mathrm{u,F}$'),
                 mlines.Line2D([], [], color=cmap(3), ls = '-', label=r'$\mathrm{d,F}$'),
                 mlines.Line2D([], [], color='k', ls = '-', label=r'$\mathrm{ref}$')]
        C, good, plot, fname = self._get_plot_params(what_uF)
        cosmoplots.change_log_axis_base(ax,'y')
        ax.plot(self.coll, self.coll, 'k')
        #ax.plot(self.coll, self.coll/self.standard_twopoint['TuN']**2, cmap(0))
        #ax.plot(self.coll, self.coll*self.standard_twopoint['ndN']/self.standard_twopoint['TdN']**2, cmap(1))
        for styleind, wk in enumerate(self.what_ind):
            plot(ax, C(wk,0,0), (C(wk,0,0)*good(1./self.TuN**2,wk,0,0)), cmap(0),styleind)
            plot(ax, C(wk,0,0), (C(wk,0,0)*good(self.ndN/self.TdN**2,wk,0,0)), cmap(1),styleind)
            plot(ax, C(wk,0,0), (C(wk,0,0)*good(self.nuF/self.TuF**2,wk,0,0)), cmap(2),styleind)
            plot(ax, C(wk,0,0), (C(wk,0,0)*good(self.ndF/self.TdF**2,wk,0,0)), cmap(3),styleind)

        ax.set_xlabel(r'$\nu_{\mathrm{ref}}$')
        ax.set_ylabel(r'$\nu_*$')

        ax.set_xlim(self.coll[0],60)
        ax.set_ylim(1e-3,1e3)
        ax.legend(handles=lines)

        fig.savefig(saveloc+'coll'+fname+'.eps')

    def plot_fluxes(self, what_uF=True):
        fig, ax = cosmoplots.figure_multiple_rows_columns(1,2)
        C, good, plot, fname = self._get_plot_params(what_uF)
        cosmoplots.change_log_axis_base(ax[0],'y')
        for styleind, wk in enumerate(self.what_ind):
            SF_int = self.doL[0]*(1+self.lhat[0]**(-1))*self.W(wk,0,0)*(1-np.exp(-1./self.doL[0]))*(good(self.nuF,wk,0,0))
            #SN_int = self.doL[0]*(1+self.lhat[0])*self.W(wk,0,0)*(1-np.exp(-1./self.doL[0]))
            G1 = good(self.ndF*np.sqrt(self.TdF),wk,0,0)

            plot(ax[0], C(wk,0,0), SF_int, cmap(0),styleind)
            #plot(ax[0], C(wk,0,0), SN_int, cmap(1),styleind)
            plot(ax[0], C(wk,0,0), G1, cmap(1),styleind)

        ax[0].set_xlabel(r'$\nu_{\mathrm{ref}}$')
        ax[0].set_ylabel(r'$\Gamma_\mathrm{tot}$')
        ax[0].set_xlim(self.coll[0],60)
        #ax[0].set_ylim(1e-2,20)
        lines = [mlines.Line2D([], [], color=cmap(0), ls = '-', label=r'$\int_0^1 S_{F}(x)\mathrm{d}x$'),
                 #mlines.Line2D([], [], color=cmap(1), ls = '-', label=r'$\int_0^1 S_{N}(x)\mathrm{d}x$'),
                 mlines.Line2D([], [], color=cmap(1), ls = '-', label=r'$n_\mathrm{d,F} \sqrt{T_\mathrm{d,F}}$')]
        ax[0].legend(handles=lines)

        for styleind, wk in enumerate(self.what_ind):
            QF_int = self.doL[0]*(1-np.exp(-1./self.doL[0]))*2.5*(1+self.lhat[0]**(-1))*self.W(wk,0,0)*(self.xi_factor*C(wk,0,0)*good(self.nuF*self.TuF,wk,0,0))
            #QN_int = self.doL[0]*(1-np.exp(-1./self.doL[0]))*2.5*(1+self.lhat[0])*self.W(wk,0,0)*(self.xi_factor*C(wk,0,0)*good(self.TuN,wk,0,0))
            qparF = self.ghat*self.xi_factor*C(wk,0,0)*good(self.ndF*self.TdF**1.5,wk,0,0)
            qparN = self.ghat*self.xi_factor*C(wk,0,0)*good(self.ndN*self.TdN**1.5,wk,0,0)
            plot(ax[1], C(wk,0,0), QF_int, cmap(0),styleind)
            #plot(ax, C(wk,0,0), QN_int, cmap(1),styleind)
            plot(ax[1], C(wk,0,0), qparF, cmap(1),styleind)
            plot(ax[1], C(wk,0,0), qparN, cmap(2),styleind)

        ax[1].set_xlabel(r'$\nu_{\mathrm{ref}}$')
        ax[1].set_ylabel(r'$q_{\mathrm{tot}}$')

        ax[1].set_xlim(self.coll[0],60)
        ax[1].set_ylim(0,1)
        lines = [mlines.Line2D([], [], color=cmap(0), ls = '-', label=r'$\int_0^1 Q_{F}(x)\mathrm{d}x$'),
                 #mlines.Line2D([], [], color=cmap(1), ls = '-', label=r'$\int_0^1 Q_{N}(x)\mathrm{d}x$'),
                 mlines.Line2D([], [], color=cmap(1), ls = '-', label=r'$q_{\parallel,\mathrm{F}}(1)$'),
                 mlines.Line2D([], [], color=cmap(2), ls = '-', label=r'$q_{\parallel,\mathrm{N}}(1)$')]
        ax[1].legend(handles=lines,ncols=3,handlelength=1,handletextpad=0.5)

        fig.savefig(saveloc+'fluxes'+fname+'.eps')

    def plot_nvt(self, what_uF=True):
        fig, ax = plt.subplots()
        C, good, plot, fname = self._get_plot_params(what_uF)
        n = 10**np.linspace(-1,6,100)
        cosmoplots.change_log_axis_base(ax,'both')
        ax.plot(n, n**(-0.68),'k-')
        ax.text(0.66, 0.45, r'$(\nu_\mathrm{ref} n_\mathrm{d,N})^{-0.68}$', transform = ax.transAxes)
        lines = []
        for li in range(len(self.lhat)):
            for dj in range(len(self.doL)):
                ind = li + len(self.lhat)*dj
                lines.append(mlines.Line2D([], [], color=cmap(ind), ls = '-', label=rf'$\hat{{\lambda}}={lhat_name[self.lhat[li]]},\, \hat{{\delta}}={dhat_name[self.doL[dj]]}$'))
                for styleind, wk in enumerate(self.what_ind):
                    plot(ax, (C(wk,li,dj)*good(self.ndN,wk,li,dj)), (good(self.TdN,wk,li,dj)), cmap(ind),styleind)
                    plot(ax, (C(wk,li,dj)*good(self.ndN,wk,li,dj)), (good(self.TdN,wk,li,dj)), cmap(ind),styleind)
        ax.set_xlabel(r'$\nu_\mathrm{ref} n_\mathrm{d,N}$')
        ax.set_ylabel(r'$T_\mathrm{d,N}$')
        ax.legend(handles=lines)

        #ax.set_xlim(self.coll[0],60)

        fig.savefig(saveloc+'nvt'+fname+'.eps')

    def plot_coll_lhat_doL_scan(self):
        # Only for what_uF = 1
        fig, ax = plt.subplots()
        good = lambda data, li, dj: data[:,:,li,dj][self.what_uF_good_ind[self.what_ind[0]][:,:,li,dj]]
        C = lambda li, dj: self.C(self.what_ind[0],li,dj)

        cosmoplots.change_log_axis_base(ax,'y')
        plt.axhline(1,c='grey',ls=':')
        lines = []
        for li in range(len(self.lhat)):
            for dj in range(len(self.doL)):
                ind = li + len(self.lhat)*dj
                lines.append(mlines.Line2D([], [], color=cmap(ind), ls = '-', label=rf'$\hat{{\lambda}}={lhat_name[self.lhat[li]]},\, \hat{{\delta}}={dhat_name[self.doL[dj]]}$'))
                ax.plot(C(li,dj), good(self.ndN*self.TuN**2/self.TdN**2,li,dj),color=cmap(ind), ls='-')
                ax.plot(C(li,dj), good(self.ndF*self.TuF**2/(self.nuF*self.TdF**2),li,dj), color=cmap(ind),ls='--')
        ax.set_xlabel(r'$\nu_\mathrm{ref}$')
        ax.set_ylabel(r'$\nu_\mathrm{d}/\nu_\mathrm{u}$')
        ax.legend(handles=lines,ncols=2)

        ax.set_xlim(self.coll[0],30)
        ax.set_ylim(1e-5,1e9)

        fig.savefig(saveloc+'coll_lhat_doL_scan_what_uF.eps')

    def plot_fluxes_doL_scan(self):
        # Only for what_uF = 1, lhat= 1
        fig, ax = cosmoplots.figure_multiple_rows_columns(1,2)
        good = lambda data, li, dj: data[:,:,li,dj][self.what_uF_good_ind[self.what_ind[0]][:,:,li,dj]]
        C = lambda li, dj: self.C(self.what_ind[0],li,dj)

        cosmoplots.change_log_axis_base(ax[0],'y')
        cosmoplots.change_log_axis_base(ax[1],'y')
        #plt.axhline(1,c='grey',ls=':')
        for li in range(0,len(self.lhat)-1):
            for dj in range(len(self.doL)):
                ind = li + len(self.lhat)*dj
                SF_int = self.doL[dj]*(1+self.lhat[li]**(-1))*self.W(self.what_ind[0],li,dj)*(1-np.exp(-1./self.doL[dj]))*(good(self.nuF,li,dj))
                G1 = good(self.ndF*np.sqrt(self.TdF),li,dj)
                QF_int = self.doL[dj]*(1-np.exp(-1./self.doL[dj]))*2.5*(1+self.lhat[li]**(-1))*self.W(self.what_ind[0],li,dj)*(self.xi_factor*C(li,dj)*good(self.nuF*self.TuF,li,dj))
                qparF = self.ghat*self.xi_factor*C(li,dj)*good(self.ndF*self.TdF**1.5,li,dj)
                qparN = self.ghat*self.xi_factor*C(li,dj)*good(self.ndN*self.TdN**1.5,li,dj)

                ax[0].plot(C(li,dj), SF_int,color=cmap(ind), ls='-')
                ax[0].plot(C(li,dj), G1,color=cmap(ind), ls='--')
                ax[1].plot(C(li,dj), QF_int,color=cmap(ind), ls='-')
                ax[1].plot(C(li,dj), qparF,color=cmap(ind), ls='--')
                ax[1].plot(C(li,dj), qparN,color=cmap(ind), ls=':')

        ax[0].set_xlabel(r'$\nu_{\mathrm{ref}}$')
        ax[0].set_ylabel(r'$\Gamma_\mathrm{tot}$')
        ax[0].set_xlim(self.coll[0],60)
        lines = [mlines.Line2D([], [], color='k', ls = '-', label=r'$\int_0^1 S_{F}(x)\mathrm{d}x$'),
                 #mlines.Line2D([], [], color=cmap(1), ls = '-', label=r'$\int_0^1 S_{N}(x)\mathrm{d}x$'),
                 mlines.Line2D([], [], color='k', ls = '--', label=r'$n_\mathrm{d,F} \sqrt{T_\mathrm{d,F}}$')]
        ax[0].legend(handles=lines)

        ax[1].set_xlabel(r'$\nu_{\mathrm{ref}}$')
        ax[1].set_ylabel(r'$q_{\mathrm{tot}}$')
        ax[1].set_xlim(self.coll[0],60)
        lines = [mlines.Line2D([], [], color='k', ls = '-', label=r'$\int_0^1 Q_{F}(x)\mathrm{d}x$'),
                 mlines.Line2D([], [], color='k', ls = '--', label=r'$q_{\parallel,\mathrm{F}}(1)$'),
                 mlines.Line2D([], [], color='k', ls = ':', label=r'$q_{\parallel,\mathrm{N}}(1)$')]
        ax[1].legend(handles=lines,ncols=3,handlelength=1,handletextpad=0.5)
        fig.savefig(saveloc+'fluxes_doL_scan_what_uF.eps')

if __name__=="__main__":
    what_vals = [0.2, 1.]
    lhat_vals = [1., 2.]
    doL_vals = [0.1,0.2]

    R = Plot(lhat_vals=lhat_vals, doL_vals=doL_vals)
    R.find_what_uF_indices(what_vals=what_vals[-1:])
    R.plot_coll_lhat_doL_scan()
    R.plot_fluxes_doL_scan()

    if False:
        # Plot comparison figures
        P = Plot(lhat_vals=lhat_vals, doL_vals=doL_vals)
        P.find_what_uF_indices(what_vals=what_vals)
        P.plot_good_params()
        P.plot_near_SOL_compare()
        P.plot_far_SOL_compare()
        P.plot_nvt()

        # Plot only lhat=1, dhat=0.2
        Q = Plot(lhat_vals=lhat_vals[:1], doL_vals=doL_vals[-1:])
        Q.find_what_uF_indices(what_vals=what_vals)
        Q.plot_near_far_pressure()
        Q.plot_coll()
        Q.plot_fluxes()
