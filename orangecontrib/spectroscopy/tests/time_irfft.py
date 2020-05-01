import time
import sys

import numpy as np
from Orange.data import Table

from orangecontrib.spectroscopy.data import agilentMosaicIFGReader

from orangecontrib.spectroscopy.irfft import (IRFFT, MultiIRFFT, PhaseCorrection,
                                              PeakSearch, ApodFunc,
                                              )
from orangecontrib.spectroscopy.tests.test_readers import initialize_reader

FILENAMES = ["agilent/4_noimage_agg256.seq",
             "C:\\Users\\reads\\tmp\\aff-testdata\\2017-11-10 4X-25X\\2017-11-10 4X-25X.dmt",
             "/data/staff/reads/aff-testdata/2017-11-10 4X-25X/2017-11-10 4x-25x.dmt",
             "/data/staff/reads/USAF 25X Mosaic/usaf 25x mosaic.dmt"
             ]
FILENAMES_FAST = FILENAMES[0:1]

def load_data(filename):
    if filename[-3:] == 'dmt':
        # This reader will only be selected manually due to shared .dmt extension
        reader = initialize_reader(agilentMosaicIFGReader, filename)
        return reader.read()
    else:
        return Table(filename)


def test_time_multi_fft(fn):
    print(fn)
    try:
        data = load_data(fn)
    except (IOError, OSError):
        print("Skipping, not present")
        return
    dx_ag = (1 / 1.57980039e+04 / 2) * 4
    fft = IRFFT(dx=dx_ag,
                apod_func=ApodFunc.BLACKMAN_HARRIS_4,
                zff=1,
                phase_res=None,
                phase_corr=PhaseCorrection.MERTZ,
                peak_search=PeakSearch.MINIMUM)
    mfft = MultiIRFFT(dx=dx_ag,
                      apod_func=ApodFunc.BLACKMAN_HARRIS_4,
                      zff=1,
                      phase_res=None,
                      phase_corr=PhaseCorrection.MERTZ,
                      peak_search=PeakSearch.MINIMUM)

    print(data.X.shape)

    min_row = float('inf')
    for _ in range(3):
        t = time.time()
        for row in data:
            # Array is at RowInstance.x
            _ = fft(row.x)
        dt = time.time() - t
        print("row by row", dt)
        min_row = min(min_row, dt)
    print("row by row (min)", min_row)

    min_batch = float('inf')
    chunk_size = 100
    chunks = max(1, len(data) // chunk_size)
    print(f"{chunks} chunks")
    for _ in range(3):
        t = time.time()
        for chunk in np.array_split(data.X, chunks, axis=0):
            _ = mfft(chunk, zpd=fft.zpd)  # use last zpd from fft
        dt = time.time() - t
        print("100 batch", dt)
        min_batch = min(min_batch, dt)
    print("100 batch (min)", min_batch)

    min_multi = float('inf')
    for _ in range(3):
        t = time.time()
        try:
            _ = mfft(data.X, zpd=fft.zpd)  # use last zpd from fft
        except MemoryError as e:
            print(e)
            break
        dt = time.time() - t
        print("multi", dt)
        min_multi = min(min_multi, dt)
    print("multi (min)", min_multi)

    try:
        print(f"Multirow Speedup: {min_row / min_multi}")
    except ZeroDivisionError:
        print("Speedup: infinity!")

    try:
        print(f"100 Batch Speedup: {min_row / min_batch}")
    except ZeroDivisionError:
        print("Speedup: infinity!")


if __name__ == "__main__":
    try:
        fast = sys.argv[1]
    except IndexError:
        fast = False
    finally:
        fns = FILENAMES_FAST if fast == "--fast" else FILENAMES
    for filename in fns:
        test_time_multi_fft(filename)
