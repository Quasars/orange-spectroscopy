from html.parser import HTMLParser
import re

import Orange
import numpy as np
from Orange.data import FileFormat, Table
from scipy.interpolate import interp1d

from orangecontrib.spectroscopy.io.gsf import reader_gsf
from orangecontrib.spectroscopy.io.util import SpectralFileFormat


class NeaReader(FileFormat, SpectralFileFormat):

    EXTENSIONS = (".nea", ".txt")
    DESCRIPTION = "NeaSPEC"

    def read_v1(self):

        with open(self.filename, "rt", encoding="utf8") as f:
            next(f)  # skip header
            l = next(f)
            l = l.strip()
            l = l.split("\t")
            ncols = len(l)

            f.seek(0)
            next(f)
            datacols = np.arange(4, ncols)
            data = np.loadtxt(f, dtype="float", usecols=datacols)

            f.seek(0)
            next(f)
            metacols = np.arange(0, 4)
            meta = np.loadtxt(
                f,
                dtype={
                    "names": ("row", "column", "run", "channel"),
                    "formats": (int, int, int, "S10"),
                },
                usecols=metacols,
            )

            # ASSUMTION: runs start with 0
            runs = np.unique(meta["run"])

            # ASSUMPTION: there is one M channel and multiple O?A and O?P channels,
            # both with the same number, both starting with 0
            channels = np.unique(meta["channel"])
            maxn = -1

            def channel_type(a):
                if a.startswith(b"O") and a.endswith(b"A"):
                    return "OA"
                elif a.startswith(b"O") and a.endswith(b"P"):
                    return "OP"
                else:
                    return "M"

            for a in channels:
                if channel_type(a) in ("OA", "OP"):
                    maxn = max(maxn, int(a[1:-1]))
            numharmonics = maxn + 1

            rowcols = np.vstack((meta["row"], meta["column"])).T
            uniquerc = set(map(tuple, rowcols))

            di = {}  # dictionary of indices for each row and column

            min_intp, max_intp = None, None

            for i, (row, col, run, chan) in enumerate(meta):
                if (row, col) not in di:
                    di[(row, col)] = {
                        "M": np.zeros((len(runs), len(datacols))) * np.nan,
                        "OA": np.zeros((numharmonics, len(runs), len(datacols)))
                        * np.nan,
                        "OP": np.zeros((numharmonics, len(runs), len(datacols)))
                        * np.nan,
                    }
                if channel_type(chan) == "M":
                    di[(row, col)][channel_type(chan)][run] = data[i]
                    if min_intp is None:  # we need the limits of common X for all
                        min_intp = np.min(data[i])
                        max_intp = np.max(data[i])
                    else:
                        min_intp = max(min_intp, np.min(data[i]))
                        max_intp = min(max_intp, np.max(data[i]))
                elif channel_type(chan) in ("OA", "OP"):
                    di[(row, col)][channel_type(chan)][int(chan[1:-1]), run] = data[i]

            X = np.linspace(min_intp, max_intp, num=len(datacols))

            final_metas = []
            final_data = []

            for row, col in uniquerc:
                cur = di[(row, col)]
                M, OA, OP = cur["M"], cur["OA"], cur["OP"]

                OAn = np.zeros(OA.shape) * np.nan
                OPn = np.zeros(OA.shape) * np.nan
                for run in range(len(M)):
                    f = interp1d(M[run], OA[:, run])
                    OAn[:, run] = f(X)
                    f = interp1d(M[run], OP[:, run])
                    OPn[:, run] = f(X)

                OAmean = np.mean(OAn, axis=1)
                OPmean = np.mean(OPn, axis=1)
                final_data.append(OAmean)
                final_data.append(OPmean)
                final_metas += [[row, col, "O%dA" % i] for i in range(numharmonics)]
                final_metas += [[row, col, "O%dP" % i] for i in range(numharmonics)]

            final_data = np.vstack(final_data)

            metas = [
                Orange.data.ContinuousVariable.make("row"),
                Orange.data.ContinuousVariable.make("column"),
                Orange.data.StringVariable.make("channel"),
            ]

            domain = Orange.data.Domain([], None, metas=metas)
            meta_data = Table.from_numpy(
                domain,
                X=np.zeros((len(final_data), 0)),
                metas=np.asarray(final_metas, dtype=object),
            )
            return X, final_data, meta_data

    def read_v2(self):

        # Find line in which data begins
        count = 0
        with open(self.filename, "rt", encoding="utf8") as f:
            while f:
                line = f.readline()
                count = count + 1
                if line[0] != "#":
                    break

            file = np.loadtxt(f)  # Slower part

        # Find the Wavenumber column
        line = line.strip().split("\t")

        for i, e in enumerate(line):
            if e == "Wavenumber":
                index = i
                break

        # Channel need to have exactly 3 letters
        Channel = line[index + 1 :]
        Channel = np.array(Channel)
        # Extract other data #
        Max_row = int(file[:, 0].max() + 1)
        Max_col = int(file[:, 1].max() + 1)
        Max_omega = int(file[:, 2].max() + 1)
        N_rows = Max_row * Max_col * Channel.size
        N_cols = Max_omega

        # Transform Actual Data
        M = np.full((int(N_rows), int(N_cols)), np.nan, dtype="float")

        for j in range(int(Max_row * Max_col)):
            row_value = file[j * (Max_omega) : (j + 1) * (Max_omega), 0]
            assert np.all(row_value == row_value[0])
            col_value = file[j * (Max_omega) : (j + 1) * (Max_omega), 1]
            assert np.all(col_value == col_value[0])
            for k in range(Channel.size):
                M[k + Channel.size * j, :] = file[
                    j * (Max_omega) : (j + 1) * (Max_omega), k + 4
                ]

        Meta_data = np.zeros((int(N_rows), 3), dtype="object")

        alpha = 0
        beta = 0
        Ch_n = int(Channel.size)

        for i in range(0, N_rows, Ch_n):
            if beta == Max_row:
                beta = 0
                alpha = alpha + 1
            Meta_data[i : i + Ch_n, 2] = Channel
            Meta_data[i : i + Ch_n, 1] = alpha
            Meta_data[i : i + Ch_n, 0] = beta
            beta = beta + 1

        waveN = file[0 : int(Max_omega), 3]
        metas = [
            Orange.data.ContinuousVariable.make("row"),
            Orange.data.ContinuousVariable.make("column"),
            Orange.data.StringVariable.make("channel"),
        ]

        domain = Orange.data.Domain([], None, metas=metas)
        meta_data = Table.from_numpy(domain, X=np.zeros((len(M), 0)), metas=Meta_data)
        return waveN, M, meta_data

    def read_spectra(self):
        version = 1
        with open(self.filename, "rt", encoding="utf8") as f:
            if f.read(2) == "# ":
                version = 2
        if version == 1:
            return self.read_v1()
        else:
            return self.read_v2()


class NeaReaderGSF(FileFormat, SpectralFileFormat):

    EXTENSIONS = (".gsf",)
    DESCRIPTION = "NeaSPEC raw files"

    def read_spectra(self):

        file_channel = str(self.filename.split(" ")[-2]).strip()
        folder_file = str(self.filename.split(file_channel)[-2]).strip()

        if "P" in file_channel:
            self.channel_p = file_channel
            self.channel_a = file_channel.replace("P", "A")
            file_gsf_p = self.filename
            file_gsf_a = self.filename.replace("P raw.gsf", "A raw.gsf")
            file_html = folder_file + ".html"
        elif "A" in file_channel:
            self.channel_a = file_channel
            self.channel_p = file_channel.replace("A", "P")
            file_gsf_a = self.filename
            file_gsf_p = self.filename.replace("A raw.gsf", "P raw.gsf")
            file_html = folder_file + ".html"
        else:
            raise ValueError("Channel not found in file name")

        data_gsf_a = self._gsf_reader(file_gsf_a)
        data_gsf_p = self._gsf_reader(file_gsf_p)
        info = self._html_reader(file_html)

        final_data, parameters, final_metas = self._format_file(
            data_gsf_a, data_gsf_p, info
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

        depth = np.arange(0, int(parameters["Pixel Area (X, Y, Z)"][3]))

        return depth, final_data, meta_data

    def _format_file(self, gsf_a, gsf_p, parameters):

        info = {}
        for row in parameters:
            key = row[0].strip(":")
            value = [v for v in row[1:] if len(v)]
            if len(value) == 1:
                value = value[0]
            info.update({key: value})

        info.update(
            {"Reader": "NeaReaderGSF"}
        )  # key used in confirmation for complex fft calculation

        averaging = int(info["Averaging"])
        px_x = int(info["Pixel Area (X, Y, Z)"][1])
        px_y = int(info["Pixel Area (X, Y, Z)"][2])
        px_z = int(info["Pixel Area (X, Y, Z)"][3])

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
                    final_metas += [[x, y, run, self.channel_a]]
                    final_metas += [[x, y, run, self.channel_p]]
                    i = f
                    f = i + px_z

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


class NeaReaderMultiChannelTXT(FileFormat, SpectralFileFormat):

    EXTENSIONS = (".txt",)
    DESCRIPTION = "NeaSPEC multichannel interferograms"

    def read_spectra(self):
        path = self.filename
        # reading the file to get the number of rows starting with an '# ' symbol
        # for getting the header info and its length
        # and separating the header from the data and its column headers
        header_length = 0
        header = []
        table_data_headers = []
        table_data_values = []
        with open(path, "r", encoding="utf-8") as f:
            data = f.readlines()
            header = [row for row in data if row.startswith("#")]
            header_length = len(header)
            if header_length == 0:
                raise KeyError(
                    "No header found in the file, please check the file format"
                )
            try:
                column_headers_line = data[header_length].strip()
                table_data_headers = np.array([dh.strip() for dh in column_headers_line.split("\t") if dh], dtype="str")
            except Exception as exc:
                print(column_headers_line)
                raise KeyError('No column headers found in the file, please check the file format') from exc
            try:
                table_data_values = np.array([row.strip().split("\t") for row in data[header_length + 1 :]], dtype="float64")
            except  Exception as exc:
                raise KeyError('No data found in the file, please check the file format') from exc

        # transforming the header into a dictionary

        # parser for the header
        def lineparser(line):
            k, v = line.strip("# ").split(":\t")
            v = v.strip().split("\t")
            v = v[0] if len(v) == 1 else v
            return k, v

        # creating the dictionary, skipping the first line of the header
        info = {}
        for line in header[1:]:
            k, v = lineparser(line)
            info.update({k: v})

        # getting the index of the columns that contain the relevant data
        # ASSUMPTION: there are multiple channels with names following
        # the format O[1-9]A and O[1-9]P for the amplitude and phase raw data
        valid_channel_cols = [
            i
            for i, dh in enumerate(table_data_headers)
            if re.match(r"O[1-9][A,P]", dh)
        ]
        data_channel_names = [table_data_headers[i] for i in valid_channel_cols]
        info.update({"Data Channels": data_channel_names})

        # info related to the data shape and final metadata
        # getting pixel area info from the header
        # and relating pixel area info to column headers of the data
        cols, rows, depth = [int(i) for i in info["Pixel Area (X, Y, Z)"][1:]]

        # getting the number of runs from the header
        runs = int(info["Averaging"])

        # defining the shape of the output data
        out_rows = len(valid_channel_cols) * runs * rows * cols
        out_cols = depth
        out_data_headers = np.arange(0, depth, 1)
        out_data = np.zeros((out_rows, out_cols), dtype="float64")

        # defining the shape of the metadata
        out_meta_ints = np.zeros((out_rows, 3), dtype="int") # Row, Column, Run
        out_meta_floats = np.zeros((out_rows, 2), dtype="float") # Z, delta_X
        out_meta_strs = np.zeros((out_rows, 1), dtype="<U10") # Channel

        # getting the index of the columns that contain the relevant meta data
        row_header_index = np.where(table_data_headers == "Row")[0][0]
        col_header_index = np.where(table_data_headers == "Column")[0][0]
        run_header_index = np.where(table_data_headers == "Run")[0][0]
        z_header_index = np.where(table_data_headers == "Z")[0][0]
        m_header_index = np.where(table_data_headers == "M")[0][0]

        # filling the output data and metadata
        # reading the data from the tableData rowise trough rows, columns and runs
        # and then columnwise trough the channels and inserting it into
        # the output table as rows using the data depth as the new data columns
        channels_len = len(valid_channel_cols)
        for row in range(rows):
            for column in range(cols):
                for run in range(runs):
                    loop_step = row * cols * runs + column * runs + run
                    input_table_index = depth * loop_step
                    data_row = table_data_values[input_table_index, row_header_index]
                    data_column = table_data_values[input_table_index, col_header_index]
                    data_run = table_data_values[input_table_index, run_header_index]
                    z_array = table_data_values[input_table_index : input_table_index + depth, z_header_index]
                    m_array = table_data_values[input_table_index : input_table_index + depth, m_header_index]
                    mean_z = np.mean(z_array)
                    delta_m = np.mean(np.diff(m_array))
                    delta_x = delta_m * 2 * 1e2 # converting interferometer distance to step size 
                    for channel in range(channels_len):
                        output_table_index = loop_step * channels_len + channel
                        channel_data = table_data_values[
                            input_table_index : input_table_index + depth,
                            valid_channel_cols[channel],
                        ]
                        out_data[output_table_index, :] = channel_data.transpose()
                        channel_name = table_data_headers[valid_channel_cols[channel]]
                        # we are intentionally changing metadata here to match the expected output
                        # by swapping the row and column index in the metadata
                        out_meta_ints[output_table_index, :] = [data_row, data_column, data_run]
                        out_meta_floats[output_table_index, :] = [mean_z, delta_x]
                        out_meta_strs[output_table_index] = channel_name

        # formatting the metadata as it is expected by Orange
        metas = [
            Orange.data.ContinuousVariable("column"),
            Orange.data.ContinuousVariable("row"),
            Orange.data.ContinuousVariable("run"),
            Orange.data.ContinuousVariable("Z"),
            Orange.data.ContinuousVariable("delta_x"),
            Orange.data.StringVariable("channel"),
        ]

        out_meta = np.hstack((out_meta_ints, out_meta_floats, out_meta_strs), dtype="object")

        domain = Orange.data.Domain([], None, metas=metas)
        meta_data = Table.from_numpy(
            domain,
            X=np.zeros((len(out_data), 0)),
            metas=out_meta,
        )

        # this info is used in the confirmation for complex fft calculation
        info.update({"Reader": "NeaReaderGSF"})
        meta_data.attributes = info

        return out_data_headers, out_data, meta_data
