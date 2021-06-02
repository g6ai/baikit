import matplotlib.pyplot as plt
import numpy as np
import palettable
from cycler import cycler
from matplotlib.ticker import AutoMinorLocator


class Data:
    """
    Base class of BasKit

    Attributes
    ----------
    manifest_fn : str
        Manifest filename.
    line_dir : str, default: ""
    line_ext : str, default: ".txt"
    line_loadtxt: dict, default: {"comments": "#", "delimiter": None, "skiprows": 0, "unpack": False, "encoding": "latin1"}
    """

    def __init__(self, manifest_fn):
        """
        Parameters
        ----------
        manifest_fn : str
            Manifest filename.
        """
        self.manifest_fn = manifest_fn

        # Var name from PlotData class
        self.line_dir = ""
        self.line_ext = ".txt"  # ".txt", ".csv", or other
        self.line_loadtxt = {
            "comments": "#",
            "delimiter": None,
            "skiprows": 0,
            "unpack": False,
            "encoding": "latin1",
        }

        # Read manifest file
        with open("input/manifest/" + self.manifest_fn + ".txt") as f:
            self.manifest_lines = [line.rstrip("\n") for line in f]
            print(self.manifest_lines)

    def load_manifest(self) -> np.ndarray:
        """
        Load manifest

        Load manifest file and return manifest lines in list.

        Returns
        -------
        ndarray:
            List of manifest lines.
        """
        manifest_lines_fields = np.array([["fn", "tag"]])

        for manifest_line_index in range(1, len(self.manifest_lines)):
            manifest_lines_fields = np.append(
                manifest_lines_fields,
                [self.manifest_lines[manifest_line_index].split(", ")],
                axis=0,
            )
        return manifest_lines_fields


class WrangleData(Data):
    """
    Wrangling data

    Wrangle data for later use.
    """

    def __init__(self, manifest_fn):
        """
        Parameters
        ----------
        manifest_fn : str
            Manifest filename.
        """
        super().__init__(manifest_fn)

    def manifest_line_2_data(self, manifest_line_index) -> tuple[np.ndarray, str, str]:
        """
        Convert manifest line to data

        Load data from manifest line.

        Parameters
        ----------
        manifest_line_index : int
            Line number of the manifest line.

        Returns
        -------
        tuple[ndarray, str, str]:
            Data, line filename, and line tag.
        """
        line_fn, line_tag = self.load_manifest()[manifest_line_index]
        data = np.loadtxt(self.line_dir + line_fn + self.line_ext, **self.line_loadtxt)
        print("Data {} shape: {}".format(line_tag, data.shape))
        return data, line_fn, line_tag

    def data_2_manifest_line(self, data, line_fn, line_tag):
        """
        Convert data to manifest line

        Save data as specified in manifest line, and print manifest line.

        Parameters
        ----------
        data : np.ndarray
            Data to save.
        line_fn : str
            Line filename.
        line_tag : str
            Line tag.
        """
        line_path = self.line_dir + line_fn + self.line_ext
        np.savetxt(line_path, data)
        print("Data {} saved as: {}".format(line_tag, line_path))
        print("fn, tag\n{}, {}".format(line_fn, line_tag))

    def unique_col0(self, data) -> np.ndarray:
        """
        Get unique values in 1st column

        Save rows where the value in 1st column is unique.

        Parameters
        ----------
        data : ndarray
            Input ndarray.

        Returns
        -------
        ndarray:
            Output ndarray.
        """
        _, index = np.unique(data[:, 0], return_index=True)
        return data[index, :]

    def find_peak(
        self, manifest_line_index, peakregion_boundaries, row_index, k_US=4, s_US=3000
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find single peak

        Find a single peak within peak region boundaries.

        Parameters
        ----------
        manifest_line_index : int
            Line number of the manifest line.
        peakregion_boundaries : ndarray
            ndarray of the peak region boundaries of the peak.
        row_index : int
            Row number.
        k_US : int, default: 4
            Degree of the smoothing spline.
        s_US : int, default: 3000
            Positive smoothing factor used to choose the number of knots.

        Returns
        -------
        tuple[ndarray, ndarray]:
            Data within peak region and peak value.
        """
        # row_index is just for print()

        # r is from Raman
        r, line_fn, line_tag = self.manifest_line_2_data(manifest_line_index)

        r_peakregion_row_index = np.where(
            np.logical_and(
                r[:, 0] >= peakregion_boundaries[0], r[:, 0] <= peakregion_boundaries[1]
            )
        )
        r_peakregion = r[r_peakregion_row_index[0], :]
        #     print('r.shape:\n{}'.format(r.shape))
        #     print('r_peakregion_row_index:\n{}'.format(r_peakregion_row_index))
        #     print('r_peakregion.shape:\n{}'.format(r_peakregion.shape))

        from scipy.interpolate import UnivariateSpline

        sp = UnivariateSpline(r_peakregion[:, 0], r_peakregion[:, 1], k=k_US, s=s_US)
        sp_d1 = sp.derivative()
        d1 = sp_d1(r_peakregion[0])

        minmax = sp_d1.roots()

        #     print(minmax.shape[0])
        #     print(minmax)

        if minmax.shape[0] == 0:
            peak_fallback_pos = np.mean(peakregion_boundaries)
            peak = np.array([peak_fallback_pos, -65536])
            #         print('peak {} cannot be determined, default value used'.format(row_index+1))
            for minmax_enum in minmax:
                if sp(minmax_enum) > peak[1]:
                    peak = np.array([minmax_enum, sp(minmax_enum)])
        else:
            peak = np.array([minmax[0], sp(minmax[0])])

        return r_peakregion, peak

    def find_peaks(self, manifest_line_index, peakregion_boundaries) -> np.ndarray:
        """
        Wrapper of find_peak()

        Find multiple peaks within a series of peak region boundaries.

        Parameters
        ----------
        manifest_line_index : int
            Line number of the manifest line.
        peakregion_boundaries : ndarray
            A 2-D ndarray of peak region boundaries.

        Returns
        -------
        ndarray:
            Peaks values.
        """
        peaks = np.zeros(peakregion_boundaries.shape)

        for row_index in range(peakregion_boundaries.shape[0]):
            _, peaks[row_index] = self.find_peak(
                manifest_line_index, peakregion_boundaries[row_index], row_index
            )
            rounded_peak = np.round(peaks[row_index])
            if rounded_peak[1] < 0:
                print(
                    "peak {}: position {} (default), height cannot be determined".format(
                        row_index + 1, rounded_peak[0]
                    )
                )
            else:
                print(
                    "peak {}: position {}, height {}".format(
                        row_index + 1, rounded_peak[0], rounded_peak[1]
                    )
                )

        print("\n")

        return peaks

    def raman_calib(self, manifest_line_index) -> np.ndarray:
        """
        Calibrate Raman spectrum

        Calibrate shift with Si's 1st order peak position (520 cm^-1). Calibrate intensity with Si's 2nd order peak average height.

        Parameters
        ----------
        manifest_line_index : int
            Line number of the manifest line.

        Returns
        -------
        ndarray:
            Calibrated Raman spectrum.
        """
        si_peakregion_boundaries = np.array([515, 525])
        # si_peakregion_boundaries = np.array([520, 540])
        si_2ndorder_peaksregion_boundaries = np.array([944, 975])

        r, line_fn, line_tag = self.manifest_line_2_data(manifest_line_index)

        # Calibrate Raman shift

        # Random fallback negative value, as row_index attribute has no use here
        row_index = -1

        r_peakregion, peak = self.find_peak(
            manifest_line_index, si_peakregion_boundaries, row_index, k_US=4, s_US=2000
        )

        from scipy.interpolate import UnivariateSpline

        sp = UnivariateSpline(r_peakregion[:, 0], r_peakregion[:, 1], k=4, s=2000)
        sp_d1 = sp.derivative()
        d1 = sp_d1(r_peakregion[:, 0])

        minmax = sp_d1.roots()

        peak = np.array([minmax[0], sp(minmax[0])])

        for minmax_enum in minmax:
            if sp(minmax_enum) > peak[1]:
                peak = [minmax_enum, sp(minmax_enum)]

        # Now calibrate
        diff = round(peak[0] - 520, 4)
        diff_w = np.multiply(np.ones(r.shape[0]), diff)

        print("Raman shift calibrated, with diff: {}".format(diff))

        r[:, 0] -= diff_w

        # Calibrate Raman intensity

        r_si_2ndorder_peaksregion_row_index = np.where(
            np.logical_and(
                r[:, 0] >= si_2ndorder_peaksregion_boundaries[0],
                r[:, 0] <= si_2ndorder_peaksregion_boundaries[1],
            )
        )
        r_si_2ndorder_peaksregion = r[r_si_2ndorder_peaksregion_row_index[0], :]
        # print('r.shape:\n{}'.format(r.shape))
        # print('r_si_2ndorder_peaksregion_row_index:\n{}'.format(r_si_2ndorder_peaksregion_row_index))
        # print('r_si_2ndorder_peaksregion.shape:\n{}'.format(r_si_2ndorder_peaksregion.shape))

        mean_2ndorder_peaksregion = np.mean(r_si_2ndorder_peaksregion[:, 1])

        coeff = 200 / mean_2ndorder_peaksregion

        r[:, 1] *= coeff

        print("Raman intensity calibrated, with coeff: {}".format(coeff))

        # ax.plot(r_peakregion[:, 0], sp(r_peakregion[:, 0])*coeff, linewidth=1, zorder=20)

        si_1storder_peakheight = peak[1] * coeff
        print(
            "Si 1st order peak height after calibration: {}\n".format(
                si_1storder_peakheight
            )
        )

        return r


class PlotData(Data):
    """
    Plot figure

    Plot figure of 2-D data.

    Attributes
    ----------
    plot_figsize : tuple, default: (6.4, 4.8)
    plot_title : str
    plot_title_flag : bool, default: False
    plot_title_fontsize : float, default: 14
    plot_ylabel : str, default: ""
    plot_axes_label_fontsize : float, default: 18
    plot_savefig_flag : bool, default: False

    subplots_layout : ndarray, default: numpy.array([[[7, 8]], [[5, 6]], [[3, 4]], [[1, 2]]])
    subplots_tag : ndarray, default: numpy.array([["example_4"], ["example_3"], ["example_2"], ["example_1"]])
    subplots_wspace : float, default: 0
    subplots_hspace : float, default: 0

    subplot_xlim : list, default: []
    subplot_ylim : list, default: []
    subplot_ylim_offset : float, default: 0
    subplot_xlabel : list, default: ""
    subplot_xscale : list, default:  "linear"
    subplot_yscale : list, default:  "linear"
    subplot_x_tick_params : dict, default: {"which": "both", "direction": "in", "width": 1.5, "labelsize": 15, "bottom": True, "top": False, "labelbottom": True}
    subplot_y_tick_params : dict, default: {"which": "both", "direction": "in", "width": 1.5, "labelsize": 15, "left": False, "right": False, "labelleft": False}
    subplot_legend_flag : bool, default: False
    subplot_legend_loc : str, default: "best"
    subplot_legend_size : float, default: 10

    line_ystep : float, default: 0
    line_annotate_flag : bool, default: False
    line_annotate_x : float, default: 1200
        The x position of the annotation.
    line_annotate_interval : float, default: 2
        The x interval of the data.
    line_annotate_fontsize : float, default: 12
    """

    def __init__(self, plot_title):
        """
        Parameters
        ----------
        plot_title : str
            Plot title which is also manifest filename.
        """
        super().__init__(plot_title)

        self.plot_figsize = (6.4, 4.8)
        self.plot_title = plot_title
        self.plot_title_flag = False
        self.plot_title_fontsize = 14
        self.plot_ylabel = ""
        self.plot_axes_label_fontsize = 18
        self.plot_savefig_flag = False

        self.subplots_layout = np.array([[[7, 8]], [[5, 6]], [[3, 4]], [[1, 2]]])
        self.subplots_tag = np.array(
            [["example_4"], ["example_3"], ["example_2"], ["example_1"]]
        )
        self.subplots_wspace = 0
        self.subplots_hspace = 0

        self.subplot_xlim = []
        self.subplot_ylim = []
        self.subplot_ylim_offset = 0
        self.subplot_xlabel = ""
        self.subplot_xscale = "linear"
        self.subplot_yscale = "linear"
        self.subplot_x_tick_params = {
            "which": "both",
            "direction": "in",
            "width": 1.5,
            "labelsize": 15,
            "bottom": True,
            "top": False,
            "labelbottom": True,
        }
        self.subplot_y_tick_params = {
            "which": "both",
            "direction": "in",
            "width": 1.5,
            "labelsize": 15,
            "left": False,
            "right": False,
            "labelleft": False,
        }
        self.subplot_legend_flag = False
        self.subplot_legend_loc = "best"
        self.subplot_legend_size = 10

        self.line_ystep = 0
        self.line_annotate_flag = False
        self.line_annotate_x = 1200  # annotation x pos
        self.line_annotate_interval = 2  # Data x interval
        self.line_annotate_fontsize = 12

    def init_plot(self):
        """
        Initialise plot

        Create subplots with axes.

        Attributes
        ----------
        subplots_shape : ndarray
        fig : Figure
        axs : Axes
        subplot_cycler : Cycler
        """
        self.subplots_shape = np.asarray(self.subplots_layout.shape)[:-1]

        self.fig = plt.figure(
            figsize=self.plot_figsize, dpi=256, facecolor="w", edgecolor="k"
        )
        gs = self.fig.add_gridspec(
            *self.subplots_shape,
            wspace=self.subplots_wspace,
            hspace=self.subplots_hspace
        )
        axs = gs.subplots(sharex=True)
        if np.prod(self.subplots_shape) == 1:
            self.axs = np.asarray([[axs]])
        elif self.subplots_shape[0] == 1:
            self.axs = np.asarray([axs])
        elif self.subplots_shape[1] == 1:
            axs_temp = [[axs[0]]]
            for axs_index in range(1, len(axs)):
                axs_temp = np.vstack([axs_temp, [axs[axs_index]]])
            self.axs = axs_temp
        else:
            self.axs = axs

        self.subplot_cycler = (
            cycler(color=palettable.tableau.Tableau_10.mpl_colors)
            * cycler(linewidth=[1.5])
            * cycler(markersize=[5])
        )

    def plot_subplots(self):
        """
        Wrapper of subplot_lines()

        Plot subplots.
        """
        # Init iteration number (array)
        its = np.zeros(self.subplots_shape, dtype=int)

        manifest_lines_fields = self.load_manifest()
        for i in range(0, self.subplots_shape[0]):
            for j in range(0, self.subplots_shape[1]):
                self.subplot_lines(i, j, its, manifest_lines_fields)

        # Plot y label
        self.fig.text(
            -0.01,
            0.5,
            self.plot_ylabel,
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=self.plot_axes_label_fontsize,
        )

        # Plot title
        if self.plot_title_flag:
            self.fig.text(
                0.5,
                1.01,
                self.plot_title,
                ha="center",
                va="center",
                fontsize=self.plot_title_fontsize,
            )

        plt.tight_layout()

        if self.plot_savefig_flag:
            plt.savefig(
                "output/" + self.plot_title + ".pdf",
                bbox_inches="tight",
                transparent=True,
            )

        plt.show()
        plt.gcf().clear()

    def subplot_lines(self, i, j, its, manifest_lines_fields):
        """
        Wrapper of the line()

        Wrap lines into a subplot.

        Parameters
        ----------
        i : int
            Row number of subplot grid.
        j : int
            Column number of subplot grid.
        its : ndarray
            ndarray of iteration numbers.
        manifest_lines_fields : ndarray
            ndarray of fields of manifest lines.
        """
        self.axs[i, j].set_prop_cycle(self.subplot_cycler)  # type: ignore

        for manifest_line_index in self.subplots_layout[i, j, :]:
            its[i, j] = self.line(
                manifest_lines_fields[manifest_line_index, 0],
                manifest_lines_fields[manifest_line_index, 1],
                its[i, j],
                self.axs[i, j],  # type: ignore
            )

        # Axis view limit
        self.axs[i, j].set_xlim(*self.subplot_xlim)  # type: ignore
        if len(self.subplot_ylim) == 2:
            self.axs[i, j].set_ylim(  # type: ignore
                self.subplot_ylim[0],
                (its[i, j] + 0) * self.line_ystep
                + self.subplot_ylim[1]
                + self.subplot_ylim_offset,
            )

        # Axis label
        if i == self.subplots_shape[0] - 1:
            self.axs[i, j].set_xlabel(  # type: ignore
                self.subplot_xlabel, fontsize=self.plot_axes_label_fontsize
            )
        self.axs[i, j].set_ylabel(  # type: ignore
            self.subplots_tag[i, j], fontsize=self.plot_axes_label_fontsize
        )

        # Axis scale
        self.axs[i, j].set_xscale(self.subplot_xscale)  # type: ignore
        self.axs[i, j].set_yscale(self.subplot_yscale)  # type: ignore

        # Axis tick
        self.axs[i, j].xaxis.set_minor_locator(AutoMinorLocator(2))  # type: ignore

        if i == self.subplots_shape[0] - 1:
            self.subplot_x_tick_params["labelbottom"] = True
        else:
            self.subplot_x_tick_params["labelbottom"] = False
        self.axs[i, j].tick_params(axis="x", **self.subplot_x_tick_params)  # type: ignore

        self.axs[i, j].tick_params(axis="y", **self.subplot_y_tick_params)  # type: ignore

        # Axis line width
        for axis in ["top", "bottom", "left", "right"]:
            self.axs[i, j].spines[axis].set_linewidth(1.5)  # type: ignore

        # Legend
        if self.subplot_legend_flag is True:
            self.axs[i, j].legend(  # type: ignore
                loc=self.subplot_legend_loc, prop={"size": self.subplot_legend_size}
            )

    def line(
        self,
        line_fn,
        line_tag,
        it,
        ax,
    ) -> int:
        """
        Plot line

        Plot a ndarray of 2-D data.

        Parameters
        ----------
        line_fn : str
            Line filename.
        line_tag : str
            Line tag.
        it : int
            Iteration number.
        ax : axes
            The axes of the subplot.

        Returns
        -------
        int:
            Iteration number.
        """
        data = np.loadtxt(self.line_dir + line_fn + self.line_ext, **self.line_loadtxt)
        print("Data {} shape: {}".format(line_tag, data.shape))

        y_cascaded = np.array(data[:, 1] + self.line_ystep * it)

        ax.plot(
            data[:, 0],
            y_cascaded,
            label=line_tag,
        )

        if self.line_annotate_flag is True:
            x_annotate_row_index = np.abs(data[:, 0] - self.line_annotate_x).argmin()
            ax.annotate(
                line_tag,
                xy=(self.line_annotate_x, y_cascaded[x_annotate_row_index, 1]),
                xytext=(-5, -self.line_annotate_fontsize),
                textcoords="offset points",
                horizontalalignment="right",
                fontsize=self.line_annotate_fontsize,
            )

        it += 1

        return it
