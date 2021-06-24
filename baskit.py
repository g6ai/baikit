import matplotlib.pyplot as plt
import numpy as np
import palettable
from cycler import cycler
from lmfit import models
from matplotlib.ticker import AutoMinorLocator


class Data:
    """
    Base class of BasKit

    Attributes
    ----------
    manifest_fn : str
        Manifest filename.
    line_dir : str
        Default: `""`
    line_ext : str
        Default: `".txt"`
    line_loadtxt: dict
        Default: `{"comments": "#", "delimiter": None, "skiprows": 0, "unpack": False, "encoding": "latin1"}`
    """

    def __init__(self, manifest_fn, manifest_dir="input/manifest/"):
        """
        Parameters
        ----------
        manifest_fn : str
            Manifest filename.
        manifest_dir : str, optional
            Manifest directory. Default: `"input/manifest/"`
        """
        self.manifest_fn = manifest_fn
        self.manifest_dir = manifest_dir

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
        with open(self.manifest_dir + self.manifest_fn + ".txt") as f:
            self.manifest_lines = [line.rstrip("\n") for line in f]
            print(self.manifest_lines)

    def load_manifest(self) -> np.ndarray:
        """
        Load manifest

        Loads manifest file and return manifest lines in list.

        Returns
        -------
        ndarray:
            List of manifest lines.
        """
        manifest_lines_fields = np.array([["fn", "tag"]], dtype=object)

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

    Wrangles data for later use.
    """

    def __init__(self, manifest_fn, manifest_dir="input/manifest/"):
        """
        Parameters
        ----------
        manifest_fn : str
            Manifest filename.
        manifest_dir : str, optional
            Manifest directory. Default: `"input/manifest/"`
        """
        super().__init__(manifest_fn, manifest_dir)

    def load_data(self, manifest_line_index) -> tuple[np.ndarray, str, str]:
        """
        Convert manifest line to data

        Loads data from manifest line.

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
        print("\nData {} shape: {}".format(line_tag, data.shape))
        return data, line_fn, line_tag

    def save_data(self, data, line_fn, line_tag):
        """
        Convert data to manifest line

        Saves data as specified in manifest line, and print manifest line.

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
        Find the unique values in 1st column

        Returns data where the values in 1st column are unique (duplicates removed) and sorted.

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
        self, data, peakregion_boundaries
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find single peak

        Finds a single peak within peak region boundaries.

        Parameters
        ----------
        data : ndarray
            Data.
        peakregion_boundaries : ndarray
            ndarray of the peak region boundaries of the peak.

        Returns
        -------
        tuple[ndarray, ndarray, ndarray]:
            Data within peak region, peak value, and data generated from fitted model.
        """
        data_pr_row_index = np.where(
            np.logical_and(
                data[:, 0] >= peakregion_boundaries[0],
                data[:, 0] <= peakregion_boundaries[1],
            )
        )
        data_pr = data[data_pr_row_index[0], :]

        mod_constant = models.ConstantModel()
        mod_gaussian = models.GaussianModel()
        mod = mod_constant + mod_gaussian

        pars = mod_constant.make_params(c=np.mean(data_pr[:, 1]))
        pars += mod_gaussian.guess(data_pr[:, 1], data_pr[:, 0])
        out = mod.fit(data_pr[:, 1], pars, x=data_pr[:, 0])

        print(out.fit_report)

        x_fit = np.arange(*peakregion_boundaries, 0.01)
        y_fit = out.eval(x=x_fit)

        peak = np.array(
            [out.best_values["center"], out.eval(x=out.best_values["center"])]
        )

        fit = np.transpose(np.vstack((x_fit, y_fit)))

        return data_pr, peak, fit

    def find_peaks(self, data, peakregion_boundaries) -> np.ndarray:
        """
        Wrapper of find_peak()

        Finds multiple peaks within a series of peak region boundaries.

        Parameters
        ----------
        data : ndarray
            Data.
        peakregion_boundaries : ndarray
            A 2-D ndarray of peak region boundaries.

        Returns
        -------
        ndarray:
            Peaks values.
        """
        peaks = np.zeros(peakregion_boundaries.shape)

        for row_index in range(peakregion_boundaries.shape[0]):
            _, peaks[row_index], _ = self.find_peak(
                data, peakregion_boundaries[row_index]
            )
            rounded_peak = np.round(peaks[row_index])
            print(
                "peak {}: position {}, height {}".format(
                    row_index + 1, rounded_peak[0], rounded_peak[1]
                )
            )

        print("\n")

        return peaks

    def shift_col0(
        self, data, peakregion_par, col0_precision=4
    ) -> tuple[np.ndarray, np.ndarray, float]:
        peakregion_boundaries = np.array(
            [
                peakregion_par[0] - peakregion_par[1],
                peakregion_par[0] + peakregion_par[1],
            ]
        )

        _, peak, _ = self.find_peak(data, peakregion_boundaries)

        diff = round(peak[0] - peakregion_par[0], col0_precision)
        diff_col0 = np.multiply(np.ones(data.shape[0]), diff)

        data[:, 0] -= diff_col0

        print("col0 calibrated with diff: {}".format(diff))

        return data, peak, diff

    def stretch_col1(
        self, data, peaksregion_boundaries, height, col1_precision=4
    ) -> tuple[np.ndarray, float]:
        data_pr_row_index = np.where(
            np.logical_and(
                data[:, 0] >= peaksregion_boundaries[0],
                data[:, 0] <= peaksregion_boundaries[1],
            )
        )
        data_pr = data[data_pr_row_index[0], :]

        mean_data_pr = np.mean(data_pr[:, 1])

        coeff = height / mean_data_pr

        data[:, 1] *= coeff
        data[:, 1] = data[:, 1].round(col1_precision)

        print("col1 streched with coeff: {}".format(coeff))

        return data, coeff

    def raman_calib(self, data) -> np.ndarray:
        """
        Calibrate Raman spectrum

        Calibrates shift with Si's 1st order peak position (520 cm^-1). Calibrates intensity with Si's 2nd order peak average height.

        Parameters
        ----------
        manifest_line_index : int
            Line number of the manifest line.

        Returns
        -------
        ndarray:
            Calibrated Raman spectrum.
        """
        si_1st_peakregion_par = np.array([520, 5])
        # si_peakregion_boundaries = np.array([520, 540])
        si_2nd_peaksregion_boundaries = np.array([944, 975])

        # Calibrate Raman shift

        data, peak, _ = self.shift_col0(data, si_1st_peakregion_par)

        # Calibrate Raman intensity

        data, coeff = self.stretch_col1(data, si_2nd_peaksregion_boundaries, 200)

        si_1st_peakheight = peak[1] * coeff
        print(
            "Si 1st order peak height after calibration: {}".format(si_1st_peakheight)
        )

        return data


class PlotData(Data):
    """
    Plot figure

    Plots figure of 2-D data.

    Attributes
    ----------
    plot_figsize : tuple
        Default: `(6.4, 4.8)`
    plot_title : str
    plot_title_flag : bool
        Default: `False`
    plot_title_fontsize : float
        Default: `14`
    plot_ylabel : str
        Default: `""`
    plot_axes_label_fontsize : float
        Default: `18`
    plot_savefig_flag : bool
        Default: `False`

    subplots_layout : ndarray
        Default: `numpy.array([[2 * x - 1, 2 * x] for x in range(4, 0, -1)]).reshape(4, 1, 2)`
    subplots_tag : ndarray
        Default: `numpy.array([f"example_{x}" for x in range(4, 0, -1)]).reshape(4, 1)`
    subplots_annotate_xyoffset : ndarray
        Default: `numpy.tile([0, -12], (4, 1, 2, 1))`
    subplots_wspace : float
        Default: `0`
    subplots_hspace : float
        Default: `0`

    subplot_xlim : list
        Default: `[]`
    subplot_ylim : list
        Default: `[]`
    subplot_ylim_offset : float
        Default: `0`
    subplot_xlabel : list
        Default: `""`
    subplot_xscale : list
        Default: `"linear"`
    subplot_yscale : list
        Default: `"linear"`
    subplot_x_tick_params : dict
        Default: `{"which": "both", "direction": "in", "width": 1.5, "labelsize": 15, "bottom": True, "top": False, "labelbottom": True}`
    subplot_y_tick_params : dict
        Default: `{"which": "both", "direction": "in", "width": 1.5, "labelsize": 15, "left": False, "right": False, "labelleft": False}`
    subplot_legend_flag : bool
        Default: `False`
    subplot_legend_loc : str
        Default: `"best"`
    subplot_legend_size : float
        Default: `10`

    line_ystep : float
        Default: `0`
    line_annotate_flag : bool
        Default: `False`
    line_annotate_x : float
        The x position of the annotation. Default: `1200`
    line_annotate_interval : float
        The x interval of the data. Default: `2`
    line_annotate_fontsize : float
        Default: `12`
    line_print_flag : bool
        Default: `True`
    """

    def __init__(self, plot_title, manifest_dir="input/manifest/"):
        """
        Parameters
        ----------
        plot_title : str
            Plot title which is also manifest filename.
        manifest_dir : str, optional
            Manifest directory. Default: `"input/manifest/"`
        """
        super().__init__(plot_title, manifest_dir)

        self.plot_figsize = (6.4, 4.8)
        self.plot_title = plot_title
        self.plot_title_flag = False
        self.plot_title_fontsize = 14
        self.plot_ylabel = ""
        self.plot_axes_label_fontsize = 18
        self.plot_savefig_flag = False

        self.subplots_layout = np.array(
            [[2 * x - 1, 2 * x] for x in range(4, 0, -1)]
        ).reshape(4, 1, 2)
        self.subplots_tag = np.array([f"example_{x}" for x in range(4, 0, -1)]).reshape(
            4, 1
        )
        self.subplots_annotate_xyoffset = np.tile([0, -12], (4, 1, 2, 1))
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
        self.line_print_flag = True

    def init_plot(self):
        """
        Initialise plot

        Creates subplots with axes.

        Parameters
        ----------
        subplots_shape : ndarray
        manifest_lines_fields : ndarray
        fig : Figure
        axs : Axes
        subplot_cycler : Cycler
        """
        self.subplots_shape = np.asarray(self.subplots_layout.shape)[:-1]
        self.manifest_lines_fields = self.load_manifest()

        self.fig = plt.figure(
            figsize=self.plot_figsize, dpi=256, facecolor="w", edgecolor="k"
        )
        gs = self.fig.add_gridspec(
            *self.subplots_shape,
            wspace=self.subplots_wspace,
            hspace=self.subplots_hspace,
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

        Plots subplots.
        """
        # Init iteration numbers (array)
        its = np.zeros(self.subplots_shape, dtype=int)

        for i in range(0, self.subplots_shape[0]):
            for j in range(0, self.subplots_shape[1]):
                self.subplot_lines(i, j, its)

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

        # plt.show()
        # plt.gcf().clear()

    def subplot_lines(self, i, j, its):
        """
        Wrapper of the line()

        Wraps lines into a subplot.

        Parameters
        ----------
        i : int
            Row number of subplot grid.
        j : int
            Column number of subplot grid.
        its : ndarray
            Iteration numbers.
        """
        self.axs[i, j].set_prop_cycle(self.subplot_cycler)  # type: ignore

        for manifest_line_index in self.subplots_layout[i, j, :]:
            its = self.line(manifest_line_index, i, j, its)

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

    def line(self, manifest_line_index, i, j, its) -> np.ndarray:
        """
        Plot line

        Plots a ndarray of 2-D data.

        Parameters
        ----------
        manifest_line_index : int
            Row number of manifest line.
        i : int
            Row number of subplot grid.
        j : int
            Column number of subplot grid.
        its : numpy.ndarray
            Iteration numbers.

        Returns
        -------
        numpy.ndarray:
            Iteration numbers.
        """
        data = np.loadtxt(
            self.line_dir
            + self.manifest_lines_fields[manifest_line_index, 0]
            + self.line_ext,
            **self.line_loadtxt,
        )
        if self.line_print_flag:
            print(
                "Data {} shape: {}".format(
                    self.manifest_lines_fields[manifest_line_index, 1], data.shape
                )
            )

        y_cascaded = np.array(data[:, 1] + self.line_ystep * its[i, j])

        (line,) = self.axs[i, j].plot(  # type: ignore
            data[:, 0],
            y_cascaded,
            label=self.manifest_lines_fields[manifest_line_index, 1],
        )

        if self.line_annotate_flag is True:
            x_annotate_row_index = np.abs(data[:, 0] - self.line_annotate_x).argmin()
            self.axs[i, j].annotate(  # type: ignore
                self.manifest_lines_fields[manifest_line_index, 1],
                xy=(self.line_annotate_x, y_cascaded[x_annotate_row_index]),
                xytext=tuple(self.subplots_annotate_xyoffset[i, j, its[i, j]]),
                textcoords="offset points",
                horizontalalignment="right",
                fontsize=self.line_annotate_fontsize,
                color=line.get_color(),
                bbox=dict(boxstyle="square", alpha=0.8, ec="w", fc="w"),
            )

        its[i, j] += 1

        return its
