from html.parser import HTMLParser
import re

import Orange
import numpy as np
import pandas as pd
from Orange.data import FileFormat, Table
from scipy.interpolate import interp1d

from pySNOM import readers

from orangecontrib.spectroscopy.io.gsf import reader_gsf
from orangecontrib.spectroscopy.io.util import SpectralFileFormat, _spectra_from_image
from orangecontrib.spectroscopy.utils import MAP_X_VAR, MAP_Y_VAR


class NeaReader(FileFormat, SpectralFileFormat):
    EXTENSIONS = (
        ".nea",
        ".txt",
        ".gsf",
    )
    DESCRIPTION = "NeaSPEC"

    @property
    def sheets(self):
        if self.filename.endswith(".nea"):
            # For old neaspec filenformat
            data_reader = readers.NeaFileLegacyReader(self.filename)
            data, _ = data_reader.read()
            channels = list(data.keys())[3:]
            channels.insert(0, "All")
        elif self.filename.endswith(".txt"):
            data_reader = readers.NeaHeaderReader(self.filename)
            channels, _ = data_reader.read()
            channels.insert(0, "All")
            channels = [
                c
                for c in channels
                if c not in ("Row", "Column", "Run", "Omega", "Wavenumber", "Depth")
            ]
        else:
            channels = []

        return channels

    @staticmethod
    def detect_image_signaltype(filename):
        # Copied from NeaImageGSF
        # We will need to change this when neaspec comes up with new channel names
        channel_strings = ["M(.?)A", "M(.?)P", "O(.?)A", "O(.?)P", "Z C", "Z raw"]
        channel_name = None

        for pattern in channel_strings:
            if re.search(pattern, filename) is not None:
                channel_name = re.search(pattern, filename)[0]

        if channel_name is None:
            signal_type = "Topography"
        elif "P" in channel_name:
            signal_type = "Phase"
        elif "A" in channel_name:
            signal_type = "Amplitude"
        else:
            signal_type = "Topography"

        return signal_type

    def read_spectra(self):
        if self.filename.endswith(".gsf"):
            # In case of GSF images the wavelength is
            # in a separate file or in the filenames
            wn = readers.get_wl_from_filename(self.filename)
            if wn is None:
                wn = readers.get_wl_from_infofile(self.filename)
            if wn is None:
                wn = 1.0

            X, XRr, YRr = reader_gsf(self.filename)
            # Flip YRr to match the coordinates
            # TODO: this should be correccted in the reader_gsf function
            # as it results in incorrect coordinates
            YRr = np.flip(YRr, axis=0)
            features, final_data, meta_data = _spectra_from_image(
                X, np.array([wn]), XRr, YRr
            )
            signal_type = self.detect_image_signaltype(self.filename)
            meta_data.attributes["measurement.signaltype"] = signal_type

            # TODO: add all the meta info here from the Gwyddion header
            return features, final_data, meta_data

        if self.filename.endswith(".nea"):
            # For old neaspec format
            data_reader = readers.NeaFileLegacyReader(self.filename)
            data, measparams = data_reader.read()
        else:
            # Current fileformat
            data_reader = readers.NeaSpectralReader(self.filename)
            data, measparams = data_reader.read()

        if self.sheet:
            chn = self.sheet
        else:
            chn = self.sheets[0]
            self.sheet = chn

        if chn == "All":
            N_chn = len(self.sheets) - 1
            channelnames = [c for c in self.sheets if c != "All"]
        else:
            N_chn = 1
            channelnames = [chn]

        Max_row = len(np.unique(data["Row"]))
        Max_col = len(np.unique(data["Column"]))

        if self.filename.endswith(".nea"):
            Max_omega = measparams["PixelArea"][2]
        else:
            # Neaspec have several the naming
            # for the indepedent variable index
            if "Depth" in data:
                Max_omega = len(np.unique(data["Depth"]))
            elif "Index" in data:
                Max_omega = len(np.unique(data["Index"]))
            elif "Omega" in data:
                Max_omega = len(np.unique(data["Omega"]))
            else:
                raise ValueError("Variable index not found")

        # Run is only there for ifg files
        Max_run = 1
        meta_cols = 3  # number of meta columns (used later)
        is_ifg = False
        N_single_chn = Max_row * Max_col * Max_run

        if "Run" in data:
            is_ifg = True
            Max_run = len(np.unique(data["Run"]))
            meta_cols = 4
            N_single_chn = Max_row * Max_col * Max_run
            # need to get the maximum of each run's minimum and minimum of each run's maximum
            # to create a common axis without extrapolation (just interpolation)
            min_newaxis = np.max(
                np.min(np.reshape(data["M"], (N_single_chn, Max_omega)), axis=1)
            )
            max_newaxis = np.min(
                np.max(np.reshape(data["M"], (N_single_chn, Max_omega)), axis=1)
            )
            new_maxis = np.linspace(min_newaxis, max_newaxis, Max_omega)

        # Number of rows and cols for X
        N_rows = N_single_chn * N_chn
        N_cols = Max_omega

        # Calculate coordinates for each point if parameters are given
        if "Rotation" in measparams:
            angle = np.radians(measparams["Rotation"])
        else:
            angle = 0
        if "ScanArea" in measparams:
            width = measparams["ScanArea"][0]
            height = measparams["ScanArea"][1]
        else:
            width = Max_col if Max_col > 1 else 0
            height = Max_row if Max_row > 1 else 0
        if "ScannerCenterPosition" in measparams:
            xoff = measparams["ScannerCenterPosition"][0]
            yoff = measparams["ScannerCenterPosition"][1]
        else:
            xoff = 0.0
            yoff = 0.0

        # Create the list of points centered to the origo
        X, Y = np.meshgrid(
            np.linspace(-width / 2, width / 2, Max_col),
            np.linspace(-height / 2, height / 2, Max_row),
        )
        xvec = X.ravel()
        yvec = Y.ravel()
        xpos = []
        ypos = []
        # Rotated the coordinate system to match orientation
        c, s = np.cos(angle), np.sin(angle)
        R = np.array(((c, -s), (s, c)))
        for i in range(len(xvec)):
            vec = np.array([xvec[i], yvec[i]])
            vec = np.matmul(R, vec)
            vec[0] += xoff
            vec[1] += yoff
            xpos.append(vec[0])
            ypos.append(vec[1])
        # Reshape
        xpos = np.reshape(np.array(xpos), (Max_col, Max_row))
        ypos = np.reshape(np.array(ypos), (Max_col, Max_row))

        # Transform actual data
        M = np.full((N_rows, N_cols), np.nan, dtype="float")

        for j in range(Max_row * Max_col):
            row_value = data["Row"][j * Max_omega : (j + 1) * Max_omega]
            assert np.all(row_value == row_value[0])
            col_value = data["Column"][j * (Max_omega) : (j + 1) * (Max_omega)]
            assert np.all(col_value == col_value[0])

            for jrun in range(Max_run):
                for jch in range(N_chn):
                    rawdata = data[channelnames[jch]]
                    startidx = j * Max_run * Max_omega + jrun * Max_omega
                    stopidx = j * Max_run * Max_omega + (jrun + 1) * Max_omega
                    idx = slice(startidx, stopidx)
                    rundata = rawdata[idx]
                    # Reinterpolate data if it is interferogram
                    if is_ifg:
                        current_axis = data["M"][idx]
                        f = interp1d(current_axis, rundata)
                        rundata = f(new_maxis)

                    M[jch + N_chn * jrun + Max_run * N_chn * j, :] = rundata

        # Preparing metas
        Meta_data = np.zeros((N_rows, meta_cols), dtype="object")

        # Filling up meta_data with positions, run and channelnames
        alpha = 0
        beta = 0

        for i in range(0, N_rows, N_chn * Max_run):
            if beta == Max_row:
                beta = 0
                alpha = alpha + 1

            for jrun in range(Max_run):
                idx = slice(i + N_chn * jrun, i + (jrun + 1) * N_chn)
                if is_ifg:
                    Meta_data[idx, -2] = jrun
                Meta_data[idx, -1] = channelnames
                Meta_data[idx, 1] = ypos[alpha, beta]
                Meta_data[idx, 0] = xpos[alpha, beta]

            beta = beta + 1

        # Prepare metas
        metas = [
            Orange.data.ContinuousVariable(MAP_X_VAR),
            Orange.data.ContinuousVariable(MAP_Y_VAR),
            Orange.data.StringVariable("channel"),
        ]
        # Neaspec tends to name the independent variable differently all the time
        # Try to prepare for all the names we know so far
        if is_ifg:
            waveN = new_maxis * 1e6
            metas.insert(2, Orange.data.ContinuousVariable.make("run"))
        else:
            possible_names = ["Wavenumber", "Wavelength"]
            varname = [name for name in possible_names if name in data][0]
            waveN = data[varname][0:Max_omega]

        domain = Orange.data.Domain([], None, metas=metas)
        meta_data = Table.from_numpy(domain, X=np.zeros((len(M), 0)), metas=Meta_data)
        # Add measurement parameters to attributes

        if is_ifg:
            # Taken from old NeaReaderMultiChannel
            # calculate datapoint spacing in cm for the fft widget as the optical path
            dx = 2 * np.mean(np.diff(new_maxis)) * 1e2  # convert [m] to [cm]
            # check file headers for wavenumber scaling factor
            # and apply it to the calculated spacing
            try:
                wavenumber_scaling = measparams["WavenumberScaling"]
                wavenumber_scaling = float(wavenumber_scaling)
                dx = dx / wavenumber_scaling
            except KeyError:
                pass
            # register the calculated spacing in the metadata
            measparams["Calculated Datapoint Spacing (Δx)"] = ["[cm]", dx]
            measparams["Domain Units"] = "[µm]"
            measparams["Channel Data Type"] = (
                "Polar",
                "i.e. Amplitude and Phase separated",
            )

        meta_data.attributes = measparams

        return waveN, M, meta_data


class NeaReaderGSF(FileFormat, SpectralFileFormat):
    EXTENSIONS = (".gsf",)
    DESCRIPTION = 'NeaSPEC legacy spectrum files'

    def read_spectra(self):
        file_channel = str(self.filename.split(' ')[-2]).strip()
        folder_file = str(self.filename.split(file_channel)[-2]).strip()

        channel_p = ""
        channel_a = ""
        file_gsf_a = ""
        file_gsf_p = ""
        file_html = ""

        if "P" in file_channel:
            channel_p = file_channel
            channel_a = file_channel.replace("P", "A")
            file_gsf_p = self.filename
            file_gsf_a = self.filename.replace("P raw.gsf", "A raw.gsf")
            file_html = folder_file + ".html"
        elif "A" in file_channel:
            channel_a = file_channel
            channel_p = file_channel.replace("A", "P")
            file_gsf_a = self.filename
            file_gsf_p = self.filename.replace("A raw.gsf", "P raw.gsf")
            file_html = folder_file + ".html"

        data_gsf_a = self._gsf_reader(file_gsf_a)
        data_gsf_p = self._gsf_reader(file_gsf_p)
        info = self._html_reader(file_html)

        final_data, parameters, final_metas = self._format_file(
            data_gsf_a, data_gsf_p, info, channel_a, channel_p
        )

        metas = [
            Orange.data.ContinuousVariable.make("column"),
            Orange.data.ContinuousVariable.make("row"),
            Orange.data.ContinuousVariable.make("run"),
            Orange.data.StringVariable.make("channel"),
        ]

        domain = Orange.data.Domain([], None, metas=metas)
        meta_data = Table.from_numpy(
            domain,
            X=np.zeros((len(final_data), 0)),
            metas=np.asarray(final_metas, dtype=object),
        )

        meta_data.attributes = parameters

        depth = np.arange(0, int(parameters['Pixel Area (X, Y, Z)'][3]))

        return depth, final_data, meta_data

    def _format_file(self, gsf_a, gsf_p, parameters, channel_a, channel_p):
        info = {}
        for row in parameters:
            key = row[0].strip(':')
            value = [v for v in row[1:] if len(v)]
            if len(value) == 1:
                value = value[0]
            info.update({key: value})

        info.update(
            {'Reader': 'NeaReaderGSF'}
        )  # key used in confirmation for complex fft calculation

        averaging = int(info['Averaging'])
        px_x = int(info['Pixel Area (X, Y, Z)'][1])
        px_y = int(info['Pixel Area (X, Y, Z)'][2])
        px_z = int(info['Pixel Area (X, Y, Z)'][3])

        data_complete = []
        final_metas = []
        for y in range(0, px_y):
            amplitude = gsf_a[y].reshape((1, px_x * px_z * averaging))[0]
            phase = gsf_p[y].reshape((1, px_x * px_z * averaging))[0]
            i = 0
            f = i + px_z
            for x in range(0, px_x):
                for run in range(0, averaging):
                    data_complete += [amplitude[i:f]]
                    data_complete += [phase[i:f]]
                    final_metas += [[x, y, run, channel_a]]
                    final_metas += [[x, y, run, channel_p]]
                    i = f
                    f = i + px_z

        # calculate datapoint spacing in cm for the fft widget
        number_of_points = px_z
        try:
            scan_size = float(
                info["Interferometer Center/Distance"][2].replace(",", "")
            )  # Microns
        except KeyError:
            scan_size = None
        if scan_size is not None:
            scan_size = scan_size * 1e-4  # Convert to cm
            step_size = (scan_size * 2) / (number_of_points - 1)
            # metadata info for the fft widget calculation
            info["Calculated Datapoint Spacing (Δx)"] = ["[cm]", step_size]

        # metadata info for selecting the correct fft method in the fft widget
        info["Channel Data Type"] = "Polar", "i.e. Amplitude and Phase separated"

        return np.asarray(data_complete), info, final_metas

    def _html_reader(self, path):
        class HTMLTableParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self._current_row = []
                self._current_table = []
                self._in_cell = False

                self.tables = []

            def handle_starttag(self, tag, attrs):
                if tag == "td":
                    self._in_cell = True

            def handle_endtag(self, tag):
                if tag == "tr":
                    self._current_table.append(self._current_row)
                    self._current_row = []
                elif tag == "table":
                    self.tables.append(self._current_table)
                    self._current_table = []
                elif tag == "td":
                    self._in_cell = False

            def handle_data(self, data):
                if self._in_cell:
                    self._current_row.append(data.strip())

        p = HTMLTableParser()
        with open(path, "rt", encoding="utf8") as f:
            p.feed(f.read())
        return p.tables[0]

    def _gsf_reader(self, path):
        X, _, _ = reader_gsf(path)
        return np.asarray(X)


class NeaReaderMultiChannel(FileFormat, SpectralFileFormat):
    EXTENSIONS = (".txt",)
    DESCRIPTION = "NeaSPEC multichannel (raw) IFGs"

    def __init__(self, filename):
        super().__init__(filename)
        self.info = {}

        self.original_df = pd.DataFrame()
        self.cartesian_df = pd.DataFrame()
        self.resampled_df = pd.DataFrame()

        self.domain = np.array([])

        self.original_channel_names = []
        self.cartesian_channel_names = []

    @staticmethod
    def _read_table_metas(fpath):
        # parser for the header
        def lineparser(line):
            k, v = line.strip("# ").split(":\t")
            v = v.strip().split("\t")
            v = v[0] if len(v) == 1 else v
            return k, v

        # read file header to get the number of rows to skip
        header_length = 0
        metadata_header = []
        with open(fpath, "r", encoding="utf-8") as f:
            data = f.readlines()
            metadata_header = [row for row in data if row.startswith("#")]
            header_length = len(metadata_header)
        # creating the dictionary, skipping the first line of the header
        meta_info = {}
        for line in metadata_header[1:]:
            k, v = lineparser(line)
            meta_info.update({k: v})

        return meta_info, header_length

    @staticmethod
    def _read_table_data(fpath, header_length):
        # table header
        formatted_table_header = []
        with open(fpath, "r", encoding="utf-8") as f:
            # the line containing the column headers is the first line after the metadata header
            table_header = f.readlines()[header_length]
            formatted_table_header = table_header.split("\t")
            formatted_table_header = [
                header.strip() for header in formatted_table_header
            ]
        # reading the data
        df = pd.read_csv(
            fpath,
            sep="\t",
            skiprows=header_length + 1,
            encoding="utf-8",
            names=formatted_table_header,
        ).dropna(axis=1, how="all")
        return df

    @staticmethod
    def _create_cartesian_df(df, original_channel_names):
        # create a new DataFrame with the cartesian form of the data
        # as Re(On) = OnA * cos(OnP), Im(On) = OnA * sin(OnP)
        amplitude_idx = [c[1] for c in original_channel_names if c[2] == "A"]
        phase_idx = [c[1] for c in original_channel_names if c[2] == "P"]
        channel_numbers = [idx for idx in amplitude_idx if idx in phase_idx]
        cartesian_channel_names = []
        cartesian_df = df.copy()
        cartesian_df.drop(columns=original_channel_names, inplace=True)
        for cn in channel_numbers:
            channel_A = f"O{cn}A"
            channel_P = f"O{cn}P"
            re_channel = f"O{cn}R"
            im_channel = f"O{cn}I"
            Re_data = df[channel_A] * np.cos(df[channel_P])
            Im_data = df[channel_A] * np.sin(df[channel_P])
            cartesian_df[re_channel] = Re_data
            cartesian_df[im_channel] = Im_data
            cartesian_channel_names.append(re_channel)
            cartesian_channel_names.append(im_channel)
        return cartesian_df, cartesian_channel_names

    @staticmethod
    def _cartesian_to_polar(re_data, im_data):
        amplitude = np.abs(re_data + 1j * im_data)
        phase = np.angle(re_data + 1j * im_data)
        return amplitude, phase

    @staticmethod
    def _extract_channel_names(df):
        return [channel for channel in df.columns if re.match(r"O[0-9][A|P]", channel)]

    @staticmethod
    def _domain_spacing(interferometer_distance, no_points):
        return interferometer_distance / (float(no_points) - 1.0)

    @staticmethod
    def _define_M_range(df):
        # find the smallest M value at the end of every run
        # and the largest M value at the beginning of every run
        # so that we can create a common domain for all runs
        # and interpolate the data to the same domain
        M_min = df.groupby("Run")["M"].min().max()
        M_max = df.groupby("Run")["M"].max().min()
        return M_min, M_max

    def _create_resampled_polar_df_from_cartesian(
        self, cartesian_df, cartesian_channel_names, domain, padding=100
    ):
        # create a new DataFrame resampling the cartesian data to the new domain
        # and transforming it to polar form (amplitude and phase)
        domain_indexes = range(len(domain))  # same as Depth values
        resampled_df = pd.DataFrame(
            columns=["Row", "Column", "Run", "Channel", *domain_indexes]
        )
        re_idx = [c[1] for c in cartesian_channel_names if c[2] == "R"]
        im_idx = [c[1] for c in cartesian_channel_names if c[2] == "I"]
        channel_numbers = [idx for idx in re_idx if idx in im_idx]
        # List to collect all rows for appending to the DataFrame
        dfrows = []
        for row in cartesian_df["Row"].unique():
            for col in cartesian_df["Column"].unique():
                for run in cartesian_df["Run"].unique():
                    for cn in channel_numbers:
                        re_channel = f"O{cn}R"
                        im_channel = f"O{cn}I"
                        # filter the data for the current row, column, and run
                        fdata = cartesian_df[
                            (cartesian_df["Row"] == row)
                            & (cartesian_df["Column"] == col)
                            & (cartesian_df["Run"] == run)
                        ]
                        # resampling the data with extra 100 points of padding on each side
                        # using the mean of the first and last 100 points to pad the data
                        re_data = np.interp(
                            domain,
                            fdata["M"],
                            fdata[re_channel],
                            left=np.mean(fdata[re_channel][:padding]),
                            right=np.mean(fdata[re_channel][-padding:]),
                        )
                        im_data = np.interp(
                            domain,
                            fdata["M"],
                            fdata[im_channel],
                            left=np.mean(fdata[im_channel][:padding]),
                            right=np.mean(fdata[im_channel][-padding:]),
                        )
                        # transform the data to polar form
                        amplitude, phase = self._cartesian_to_polar(re_data, im_data)
                        # add the resampled data to the list as OnA and OnP channels
                        dfrows.append(
                            {
                                "Row": row,
                                "Column": col,
                                "Run": run,
                                "Channel": f"O{cn}A",
                                **dict(enumerate(amplitude)),
                            }
                        )
                        dfrows.append(
                            {
                                "Row": row,
                                "Column": col,
                                "Run": run,
                                "Channel": f"O{cn}P",
                                **dict(enumerate(phase)),
                            }
                        )
        # Create the DataFrame from the list of rows
        resampled_df = pd.concat(
            [pd.DataFrame([row]) for row in dfrows], ignore_index=True
        )
        return resampled_df

    def create_padded_domain(self, df, padding=100):
        # creates a new domain for resampling the data:
        # the domain will be created from the M values in the data
        # its spacing interval will be calculated from metadata
        # it will contain the number of points of the original data
        # but with added padding on each side to accomodate the resampling
        # to a common interval

        if not self.info:
            raise ValueError("No metadata found")
        # parse info from the metadata to create the domain
        interferometer_units, _, interferometer_distance = self.info[
            "Interferometer Center/Distance"
        ]
        if interferometer_units != "[µm]":
            raise ValueError("Interferometer units are not in micrometers")
        interferometer_distance = (
            float(interferometer_distance) * 1e-6
        )  # convert [µm] to [m]
        px_area_units, _, _, z = self.info["Pixel Area (X, Y, Z)"]
        if px_area_units != "[px]":
            raise ValueError("Pixel area units are not in pixels")
        no_points = int(z)
        m_min, m_max = self._define_M_range(df)
        dm = self._domain_spacing(interferometer_distance, no_points)
        # create the domain with the padding
        d_start = m_min - padding * dm
        d_end = m_min + (no_points + padding) * dm
        if d_end < m_max:
            raise ValueError(
                "Could not create a domain with the given padding. \
                             M maximum values are outside the expected range."
            )
        self.domain = np.arange(d_start, d_end, dm)
        self.calculate_datapoint_spacing(dm)

    def calculate_datapoint_spacing(self, domain_spacing):
        # calculate datapoint spacing in cm for the fft widget as the optical path
        dx = 2 * float(domain_spacing) * 1e2  # convert [m] to [cm]
        # check file headers for wavenumber scaling factor
        # and apply it to the calculated spacing
        try:
            wavenumber_scaling = self.info["Wavenumber Scaling"]
            wavenumber_scaling = float(wavenumber_scaling)
            dx = dx / wavenumber_scaling
        except KeyError:
            pass
        # register the calculated spacing in the metadata
        self.info["Calculated Datapoint Spacing (Δx)"] = ["[cm]", dx]

    def create_original_df(self, fpath):
        self.filename = fpath
        self.info, header_length = self._read_table_metas(self.filename)
        self.original_df = self._read_table_data(self.filename, header_length)
        self.original_channel_names = self._extract_channel_names(self.original_df)

    def create_cartesian_df(self):
        if self.original_df.empty or not self.original_channel_names:
            raise ValueError("Original data not found")

        self.cartesian_df, self.cartesian_channel_names = self._create_cartesian_df(
            self.original_df, self.original_channel_names
        )
        # clean up the original DataFrame
        self.original_df = pd.DataFrame()

    def create_resampled_df(self, padding=100):
        if self.cartesian_df.empty or not self.cartesian_channel_names:
            raise ValueError("Cartesian data not found")
        if self.domain.size == 0:
            raise ValueError("Domain not found")
        self.resampled_df = self._create_resampled_polar_df_from_cartesian(
            self.cartesian_df,
            self.cartesian_channel_names,
            self.domain,
            padding=padding,
        )
        # clean up the cartesian DataFrame
        self.cartesian_df = pd.DataFrame()

    def read_spectra(self):
        self.create_original_df(self.filename)
        self.create_padded_domain(self.original_df, padding=100)
        self.create_cartesian_df()
        self.create_resampled_df(padding=100)
        df = self.resampled_df
        # format output to be used in the read method

        # format data
        out_data = df.drop(columns=["Row", "Column", "Run", "Channel"]).values
        out_data = out_data.astype(np.float64)
        # formatting metas
        meta_domain = [
            Orange.data.ContinuousVariable.make(MAP_X_VAR),
            Orange.data.ContinuousVariable.make(MAP_Y_VAR),
            Orange.data.ContinuousVariable.make("run"),
            Orange.data.StringVariable.make("channel"),
        ]
        out_meta = df[["Column", "Row", "Run", "Channel"]].values

        # formatting domain
        # scale the domain to micrometers
        scaled_domain = self.domain * 1e6
        self.info["Domain Units"] = "[µm]"
        orange_domain = Orange.data.Domain([], None, metas=meta_domain)
        meta_data = Table.from_numpy(
            orange_domain,
            X=np.zeros((out_data.shape[0], 0)),
            metas=out_meta,
        )
        # info for the fft widget
        self.info["Channel Data Type"] = "Polar", "i.e. Amplitude and Phase separated"
        meta_data.attributes = self.info
        return scaled_domain, out_data, meta_data
