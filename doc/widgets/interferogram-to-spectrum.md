Interferogram to Spectrum
=========================

Fast Fourier Transform interferograms, including zero filling, apodization and phase correction.

**Inputs**

- Interferogram: input interferogram
- Stored Phase: Previously calculated phase data

**Outputs**

- Spectra: dataset with spectra
- Phases: calulated phases

The **Interferogram to Spectrum** widgets calculates real or complex FFT for interferograms.

![](images/FFT.png)

1. Info: Information about number and type of interferograms and metadata available.
2. Input Data:
   - Datapoint spacing: The pathlength step size used to calculate the output frequencies. 
     "Auto" will use values from metadata or a default HeNe fallback.
   - Sweep direction: For datasets containing 2 full interferograms that were collected with forward and backward mirror motion.
   - ZPD Peak Search: Zero-point difference peak search algorithm. Disable search to manually enter value for entire dataset. The second entry is for double interferograms.
3. FFT Options:
   - Complex FFT: Calculate the complex FFT from Amplitude and Phase data. Data can be interleaved or amplitude on first input and phase on the second.
   - Apodization Function: Windowing function to apply before FFT
     - Asymmetric Interferometer: Apply apodization asymmetrically ([reference](https://doi.org/10.1364/oe.520793))
   - Zero Filling Factor: Pad dataset with zeros to 2**zff
   - Phase Correction: Calculate phase using Mertz, or use a stored phase. "None" outputs both real/imaginary values.
4. Output:
   - Limit spectral region: Cut the output to the specified range.
