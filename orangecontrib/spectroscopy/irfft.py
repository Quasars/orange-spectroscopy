from enum import IntEnum

import numpy as np


class ApodFunc(IntEnum):
    """
    Implemented apodization functions in apodize
    """
    BOXCAR = 0
    BLACKMAN_HARRIS_3 = 1
    BLACKMAN_HARRIS_4 = 2
    BLACKMAN_NUTTALL = 3


class PhaseCorrection(IntEnum):
    """
    Implemented phase correction methods
    """
    MERTZ = 0
    MERTZSIGNED = 1
    STORED = 2
    NONE = 3


class PeakSearch(IntEnum):
    """
    Implemented peak search functions
    """
    MAXIMUM = 0
    MINIMUM = 1
    ABSOLUTE = 2


def find_zpd(ifg, peak_search):
    """
    Find the zero path difference (zpd) position.

    Args:
        ifg (np.array): 1D array with a single interferogram
        peak_search (IntEnum): One of peak search functions:
            <PeakSearch.MAXIMUM: 0>         : Maximum value
            <PeakSearch.MINIMUM: 1>         : Minumum value
            <PeakSearch.ABSOLUTE: 2>        : Absolute largest value

    Returns:
        zpd: The index of zpd in ifg array.
    """
    if peak_search == PeakSearch.MAXIMUM:
        return ifg.argmax()
    elif peak_search == PeakSearch.MINIMUM:
        return ifg.argmin()
    elif peak_search == PeakSearch.ABSOLUTE:
        return ifg.argmin() if abs(ifg.min()) > abs(ifg.max()) else ifg.argmax()
    else:
        raise NotImplementedError

def wing_size(ifg, zpd):
    """Calculate negative and positive wing size (including zpd point)"""
    wing_n = zpd + 1
    wing_p = ifg.shape[-1] - zpd
    return wing_n, wing_p

def apodize(ifg, zpd, apod_func):
    """
    Perform apodization of asymmetric interferogram using selected apodization
    function

    Args:
        ifg (np.array): interferogram array (1D or 2D row-wise)
        zpd (int): Index of the Zero Phase Difference (centerburst)
        apod_func (IntEnum): One of apodization function options:
                <ApodFunc.BOXCAR: 0>            : Boxcar apodization
                <ApodFunc.BLACKMAN_HARRIS_3: 1> : Blackman-Harris (3-term)
                <ApodFunc.BLACKMAN_HARRIS_4: 2> : Blackman-Harris (4-term)
                <ApodFunc.BLACKMAN_NUTTALL: 3>  : Blackman-Nuttall (Eric Peach implementation)

    Returns:
        ifg_apod (np.array): apodized interferogram(s)
    """

    if apod_func == ApodFunc.BOXCAR:
        # Boxcar apodization AKA as-collected
        return ifg

    # Calculate negative and positive wing size
    # correcting zpd from 0-based index
    ifg_N = ifg.shape[-1]
    wing_n, wing_p = wing_size(ifg, zpd)
    # Use larger wing to define apodization window width
    M = max(wing_n, wing_p)

    # Generate apodization function
    if apod_func == ApodFunc.BLACKMAN_HARRIS_3:
        # Blackman-Harris (3-term)
        # Reference: W. Herres and J. Gronholz, Bruker
        #           "Understanding FT-IR Data Processing"
        A0 = 0.42323
        A1 = 0.49755
        A2 = 0.07922
        A3 = 0.0
    elif apod_func == ApodFunc.BLACKMAN_HARRIS_4:
        # Blackman-Harris (4-term)
        # Reference: W. Herres and J. Gronholz, Bruker
        #           "Understanding FT-IR Data Processing"
        A0 = 0.35875
        A1 = 0.48829
        A2 = 0.14128
        A3 = 0.01168
    elif apod_func == ApodFunc.BLACKMAN_NUTTALL:
        # Blackman-Nuttall
        # Reference: A. Nuttall, "Some windows with very good sidelobe behavior," IEEE
        #            Transactions on Acoustics, Speech, and Signal Processing, 1981.
        A0 = 0.3635819
        A1 = 0.4891775
        A2 = 0.1365995
        A3 = 0.0106411
    else:
        raise ValueError(f"Invalid apodization function {apod_func}")

    n = np.arange(1-M, M, 1)
    Bs = A0 \
         + A1 * np.cos(np.pi*n/(M-1)) \
         + A2 * np.cos(np.pi*2*n/(M-1)) \
         + A3 * np.cos(np.pi*3*n/(M-1))

    # Select appropriate subsection of apodization function
    if wing_n > wing_p:
        Bs = Bs[:ifg_N]
    else:
        Bs = Bs[2 * M - 1 - ifg_N:]
    # Apodize the sampled Interferogram
    try:
        ifg_apod = ifg * Bs
    except ValueError as e:
        raise ValueError("Apodization function size mismatch: %s" % e)

    return ifg_apod

def ramp(ifg, zpd):
    wing_n, wing_p = wing_size(ifg, zpd)
    # Use smaller wing to define symmetric subset size
    M = min(wing_n, wing_p)
    ramp = np.linspace(0, 1, 2 * M - 1)
    if wing_n < wing_p:
        ifg[..., :ramp.shape[0]] *= ramp
    else:
        ifg[..., -ramp.shape[0]:] *= ramp[::-1]

    return ifg

def _zero_fill_size(ifg_N, zff):
    # Calculate desired array size
    Nzff = ifg_N * zff
    # Calculate final size to next power of two for DFT efficiency
    return int(np.exp2(np.ceil(np.log2(Nzff))))

def _zero_fill_pad(ifg, zerofill):
    zeroshape = (ifg.shape[0], zerofill) if ifg.ndim == 2 else zerofill
    return np.hstack((ifg, np.zeros(zeroshape, dtype=ifg.dtype)))

def zero_fill(ifg, zff):
    """
    Zero-fill interferogram to DFT-efficient power of two.
    Assymetric to prevent zpd from changing index.

    Args:
        ifg (np.array): interferogram array (1D or 2D row-wise)
        zff (int): Zero-filling factor

    Returns:
        np.array: ifg with appended zero fill
    """
    ifg_N = ifg.shape[-1]
    # Calculate zero-fill to next power of two for DFT efficiency
    zero_fill = _zero_fill_size(ifg_N, zff) - ifg_N
    # Pad array
    return _zero_fill_pad(ifg, zero_fill)

class IRFFT():
    """
    Calculate FFT of a single interferogram sweep.

    Based on mertz module by Eric Peach, 2014
    """
    # Calculated attributes
    zpd = None
    wavenumbers = None
    spectrum = None
    phase = None


    def __init__(self, dx,
                 apod_func=ApodFunc.BLACKMAN_HARRIS_3, zff=2,
                 phase_res=None, phase_corr=PhaseCorrection.MERTZ,
                 peak_search=PeakSearch.MAXIMUM, apod_asym=False,
                ):
        self.dx = dx
        self.apod_func = apod_func
        self.apod_asym = apod_asym
        self.zff = zff
        self.phase_res = phase_res
        self.phase_corr = phase_corr
        self.peak_search = peak_search

    def __call__(self, ifg, zpd=None, phase=None):
        if ifg.ndim != 1:
            raise ValueError("ifg must be 1D array")
        # Stored phase
        self.phase = phase
        # Stored ZPD
        if zpd is not None:
            self.zpd = zpd
        else:
            self.zpd = find_zpd(ifg, self.peak_search)

        # Subtract DC value from interferogram
        ifg = ifg - ifg.mean()

        # Calculate phase on interferogram of specified size 2*L
        L = self.phase_ifg_size(ifg.shape[0])
        if L == 0: # Use full ifg for phase
            ifg = apodize(ifg, self.zpd, self.apod_func)
            ifg = zero_fill(ifg, self.zff)
            # Rotate the Complete IFG so that the centerburst is at edges.
            ifg = np.hstack((ifg[self.zpd:], ifg[0:self.zpd]))
            Nzff = ifg.shape[0]
            # Take FFT of Rotated Complete Graph
            ifg = np.fft.rfft(ifg)
            self.compute_phase(ifg)
        else:
            # Select phase interferogram as copy
            # Note that L is now the zpd index
            Ixs = ifg[self.zpd - L : self.zpd + L].copy()
            ifg = apodize(ifg, self.zpd, self.apod_func)
            ifg = ramp(ifg, self.zpd)
            ifg = zero_fill(ifg, self.zff)
            ifg = np.hstack((ifg[self.zpd:], ifg[0:self.zpd]))
            Nzff = ifg.shape[0]

            Ixs = apodize(Ixs, L, self.apod_func)
            # Zero-fill Ixs to same size as ifg (instead of interpolating later)
            Ixs = _zero_fill_pad(Ixs, Nzff - Ixs.shape[0])
            Ixs = np.hstack((Ixs[L:], Ixs[0:L]))

            ifg = np.fft.rfft(ifg)
            Ixs = np.fft.rfft(Ixs)
            self.compute_phase(Ixs)

        self.wavenumbers = np.fft.rfftfreq(Nzff, self.dx)

        if self.phase_corr == PhaseCorrection.NONE:
            self.spectrum = ifg.real
            self.phase = ifg.imag
        else:
            try:
                self.spectrum = np.cos(self.phase) * ifg.real + np.sin(self.phase) * ifg.imag
            except ValueError as e:
                raise ValueError("Incompatible phase: {}".format(e))

        return self.spectrum, self.phase, self.wavenumbers

    def phase_ifg_size(self, ifg_N):
        # Determine largest possible double-sided interferogram
        delta = np.min([self.zpd, ifg_N - 1 - self.zpd])
        # Reduce to desired resolution if specified
        # TODO make a single function to implement setting output resolution
        if self.phase_res is not None:
            # TODO check this
            L = int(1 / (self.dx * self.phase_res)) - 1
            if L < delta:
                delta = L

        return delta

    def compute_phase(self, ifg_sub_fft):
        if self.phase_corr == PhaseCorrection.NONE:
            return
        elif self.phase_corr == PhaseCorrection.STORED:
            if self.phase is None:
                raise ValueError("No stored phase provided.")
            return
        elif self.phase_corr == PhaseCorrection.MERTZ:
            self.phase = np.arctan2(ifg_sub_fft.imag, ifg_sub_fft.real)
        elif self.phase_corr == PhaseCorrection.MERTZSIGNED:
            self.phase = np.arctan(ifg_sub_fft.imag/ifg_sub_fft.real)
        else:
            raise ValueError("Invalid PhaseCorrection: {}".format(self.phase_corr))


class MultiIRFFT(IRFFT):

    def __call__(self, ifg, zpd=None, phase=None):
        if ifg.ndim != 2:
            raise ValueError("ifg must be 2D array of row-wise interferograms")
        # TODO does stored phase work / make sense here?
        self.phase = phase
        try:
            self.zpd = int(zpd)
        except TypeError:
            raise TypeError("zpd must be specified as a single value valid for all interferograms")

        # Subtract DC value from interferogram
        ifg = ifg - ifg.mean(axis=1, keepdims=True)

        # Calculate phase on interferogram of specified size 2*L
        L = self.phase_ifg_size(ifg.shape[1])
        if L == 0: # Use full ifg for phase #TODO multi is this code tested
            ifg = apodize(ifg, self.zpd, self.apod_func)
            ifg = zero_fill(ifg, self.zff)
            # Rotate the Complete IFG so that the centerburst is at edges.
            ifg = np.hstack((ifg[self.zpd:], ifg[0:self.zpd]))
            Nzff = ifg.shape[0]
            # Take FFT of Rotated Complete Graph
            ifg = np.fft.rfft(ifg)
            self.compute_phase(ifg)
        else:
            # Select phase interferogram as copy
            # Note that L is now the zpd index
            Ixs = ifg[:, self.zpd - L : self.zpd + L].copy()
            ifg = apodize(ifg, self.zpd, self.apod_func)
            ifg = ramp(ifg, self.zpd)
            ifg = zero_fill(ifg, self.zff)
            ifg = np.hstack((ifg[:, self.zpd:], ifg[:, 0:self.zpd]))
            Nzff = ifg.shape[1]

            Ixs = apodize(Ixs, L, self.apod_func)
            # Zero-fill Ixs to same size as ifg (instead of interpolating later)
            Ixs = _zero_fill_pad(Ixs, Nzff - Ixs.shape[1])
            Ixs = np.hstack((Ixs[:, L:], Ixs[:, 0:L]))

            ifg = np.fft.rfft(ifg)
            Ixs = np.fft.rfft(Ixs)
            self.compute_phase(Ixs)

        self.wavenumbers = np.fft.rfftfreq(Nzff, self.dx)

        if self.phase_corr == PhaseCorrection.NONE:
            self.spectrum = ifg.real
            self.phase = ifg.imag
        else:
            try:
                self.spectrum = np.cos(self.phase) * ifg.real + np.sin(self.phase) * ifg.imag
            except ValueError as e:
                raise ValueError("Incompatible phase: {}".format(e))

        return self.spectrum, self.phase, self.wavenumbers


class ComplexFFT(IRFFT):

    def __call__(self, ifg, zpd=None, phase=None):
        ifg -= np.mean(ifg)

        if zpd is not None:
            self.zpd = zpd
        else:
            self.zpd = find_zpd(ifg, self.peak_search)

        ifg = apodize(ifg, self.zpd, self.apod_func)
        ifg = zero_fill(ifg, self.zff)
        # Rotate the Complete IFG so that the centerburst is at edges.
        ifg = np.hstack((ifg[self.zpd:], ifg[0:self.zpd]))
        Nzff = ifg.shape[0]
        ifg = np.fft.fft(ifg)
        
        magnitude = np.abs(ifg)
        angle = np.angle(ifg)
        self.wavenumbers = np.fft.rfftfreq(Nzff, self.dx)
        self.spectrum = magnitude[:len(self.wavenumbers)]
        self.phase = angle[:len(self.wavenumbers)]

        return self.spectrum, self.phase, self.wavenumbers
