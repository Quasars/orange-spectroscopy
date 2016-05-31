import numpy as np

def peak_search(Ix):
    """
    Find the zero path difference (zpd) position.

    NB Only "Maximum" peak search is currently implemented.

    Args:
        Ix (np.array): 1D array with a single interferogram

    Returns:
        zpd: The index of zpd in Ix array.
    """
    zpd = Ix.argmax()
    return zpd

def apodization(Ix, zpd, apod_func):
    """
    Perform apodization of asymmetric interferogram using selected apodization
    function

    Args:
        Ix (np.array): 1D array with a single interferogram
        zpd (int): Index of the Zero Phase Difference (centerburst)
        apod_func (int): One of apodization function options:
                            0 : Boxcar apodization
                            1 : Blackman-Harris (3-term)
                            2 : Blackman-Harris (4-term)
                            3 : Blackman-Nuttall (Eric Peach implementation)

    Returns:
        Ix_apod (np.array): 1D array of apodized Ix
    """

    # Calculate negative and positive wing size
    # correcting zpd from 0-based index
    N = Ix.shape[0]
    wing_n = zpd + 1
    wing_p = N - (zpd + 1)

    if apod_func == 0:
        # Boxcar apodization AKA as-collected
        Bs = np.ones_like(Ix)
    elif apod_func == 1:
        # Blackman-Harris (3-term)
        # Reference: W. Herres and J. Gronholz, Bruker
        #           "Understanding FT-IR Data Processing"
        A0 = 0.42323
        A1 = 0.49755
        A2 = 0.07922
        A3 = 0.0
        n_n = np.arange(wing_n)
        n_p = np.arange(wing_p)
        Bs_n =  A0\
            + A1 * np.cos(np.pi*n_n/wing_n)\
            + A2 * np.cos(np.pi*2*n_n/wing_n)\
            + A3 * np.cos(np.pi*3*n_n/wing_n)
        Bs_p =  A0\
            + A1 * np.cos(np.pi*n_p/wing_p)\
            + A2 * np.cos(np.pi*2*n_p/wing_p)\
            + A3 * np.cos(np.pi*3*n_p/wing_p)
        Bs = np.hstack((Bs_n[::-1], Bs_p))
    elif apod_func == 2:
        # Blackman-Harris (4-term)
        # Reference: W. Herres and J. Gronholz, Bruker
        #           "Understanding FT-IR Data Processing"
        A0 = 0.35875
        A1 = 0.48829
        A2 = 0.14128
        A3 = 0.01168
        n_n = np.arange(wing_n)
        n_p = np.arange(wing_p)
        Bs_n =  A0\
            + A1 * np.cos(np.pi*n_n/wing_n)\
            + A2 * np.cos(np.pi*2*n_n/wing_n)\
            + A3 * np.cos(np.pi*3*n_n/wing_n)
        Bs_p =  A0\
            + A1 * np.cos(np.pi*n_p/wing_p)\
            + A2 * np.cos(np.pi*2*n_p/wing_p)\
            + A3 * np.cos(np.pi*3*n_p/wing_p)
        Bs = np.hstack((Bs_n[::-1], Bs_p))

    elif apod_func == 3:
        # Blackman-Nuttall (Eric Peach)
        # TODO I think this has silent problems with asymmetric interferograms
        delta = np.min([wing_n , wing_p])

        # Create Blackman Nuttall Window according to the formula given by Wolfram.
        xs = np.arange(N)
        Bs = np.zeros(N)
        Bs = 0.3635819\
            - 0.4891775 * np.cos(2*np.pi*xs/(2*delta - 1))\
            + 0.1365995 * np.cos(4*np.pi*xs/(2*delta - 1))\
            - 0.0106411 * np.cos(6*np.pi*xs/(2*delta - 1))

    # Apodize the sampled Interferogram
    try:
        Ix_apod = Ix * Bs
    except ValueError as e:
        raise ValueError("Apodization function size mismatch: %s" % e)

    return Ix_apod

def zero_filling(Ix, zff):
    """
    Zero-fill interferogram.
    Assymetric to prevent zpd from changing index.

    Args:
        Ix (np.array): 1D array with a single interferogram
        zff (int): Zero-filling factor

    Returns:
        Ix_zff: 1D array of Ix + zero fill
    """
    N = Ix.shape[0]
    # Calculate next power of two for DFT efficiency
    N_2 = int(np.exp2(np.ceil(np.log2(N))))
    # fill to N**2 * zff
    zero_fill = ((N_2 - N) + (N_2 * (zff)))
    Ix_zff = np.hstack((Ix, np.zeros(zero_fill)))
    return Ix_zff
