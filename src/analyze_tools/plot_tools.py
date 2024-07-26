import numpy as np
import matplotlib.pyplot as plt

mcolors_list = ["b","g","r","c","m","y","k"]
annotate = ["(a)","(b)","(c)","(d)"]

def moving_average(x, y, w):
    #x = np.convolve(x, np.ones(w), 'valid') / w
    y_new = np.convolve(y, np.ones(w), 'valid') / w
    halfw = int(w/2)
    y_new = np.hstack(
        [y[:halfw], y_new, y[len(x) + 1 - halfw:]]
        )
    return x,y_new

class Plot:
    """
    Class for constructing plots for dipole vs time, energy vs modes' wavenumebers
    histogram distrubtion of velocities,
    """
    def __init__(
        self, n_spec_plots, n_dipole_plots = 0
        ):
    
        self.n_spec_plots = n_spec_plots
        self.n_dipole_plots = n_dipole_plots

        if n_dipole_plots > 0:
            self.fig_dip, self.ax_dip = plt.subplots(
                n_dipole_plots, figsize = (6,4 * n_dipole_plots)
                )

        if n_spec_plots > 0:
            self.fig_spec, self.ax_spec = plt.subplots(
                n_spec_plots, figsize = (6,4 * n_spec_plots)
                )

    def add_dipole_plot(self, i, t, dipole, index):
        ax = self.ax_dip[i] if self.n_dipole_plots > 1 else self.ax_dip
        color_index = index % len(mcolors_list)
        self.ax_dip[i].plot(t, dipole, mcolors_list[i], c = mcolors_list[color_index]) 

    def add_spec_plot(
        self, i, wavenumber, energy, ma_w = 10, scatter = True,
        line_label = None
        ):
        
        ax = self.ax_spec[i] if self.n_spec_plots > 1 else self.ax_spec
        if scatter:
            ax.scatter(wavenumber, energy, s = 1)

        if ma_w:
            w, e = moving_average(wavenumber, energy, w = ma_w)
            ax.plot(w, e, label = line_label)

    def add_label(self, i, spec_label = None, dip_label = None):
        if self.n_spec_plots < 1: 
            pass
        else:
            if spec_label is not None:
                xlabel , ylabel = spec_label
            elif spec_label == None:
                xlabel, ylabel = "Wavenumber (1/cm)", "Radiation mode energy (eV)"
            ax = self.ax_spec[i] if self.n_spec_plots > 1 else self.ax_spec
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        if dip_label:
            ax = self.ax_dip[i] if self.n_dip_plots > 1 else self.ax_dip
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

    def add_legend(self, i):
        ax = self.ax_spec[i] if self.n_spec_plots > 1 else self.ax_spec
        ax.legend()

    def annotate(self):
        if self.n_spec_plots == 1: 
            print("No need annotation for 1 plot!")
            return 0
        for i in range(self.n_spec_plots):
            self.ax_spec[i].annotate(
                annotate[i], xy = (0.05,0.9), xycoords = 'axes fraction')

    def savefig(self, root, spec_append = '', dip_append = ''):
        
        if self.n_spec_plots > 0:
            self.fig_spec.savefig(path + "spectrum.jpeg",dpi=600,bbox_inches = "tight")
        if self.n_dipole_plots > 0:
            self.fig_dip.savefig(path + "total_dipole.jpeg",dpi=600,bbox_inches = "tight")



