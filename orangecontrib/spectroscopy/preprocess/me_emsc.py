import numpy as np
from scipy.signal import hilbert
from sklearn.decomposition import TruncatedSVD

import Orange
from Orange.data import Table
from Orange.preprocess.preprocess import Preprocess
from Orange.data.util import get_unique_names

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.preprocess.utils import (
    SelectColumn,
    CommonDomainOrderUnknowns,
    interpolate_extend_to,
    CommonDomainRef,
    table_eq_x,
    subset_for_hash,
)
from orangecontrib.spectroscopy.preprocess.emsc import (
    weighted_wavenumbers,
    average_table_x,
)


def calculate_complex_n(ref_X, wavenumbers):
    """Calculates the scaled imaginary part and scaled fluctuating real part of the refractive index."""
    npr = ref_X
    nprs = npr / (wavenumbers * 100)

    # Extend absorbance spectrum
    dw = wavenumbers[1] - wavenumbers[0]
    wavenumbers_extended = np.hstack(
        (
            dw * np.linspace(1, 200, 200) + (wavenumbers[0] - dw * 201),
            wavenumbers,
            dw * np.linspace(1, 200, 200) + (wavenumbers[-1]),
        )
    )
    extension1 = npr[0] * np.ones(200)
    extension2 = npr[-1] * np.ones(200)
    npr_extended = np.hstack((extension1, npr, extension2))

    # Calculate Hilbert transform
    nkks_extended = -hilbert(npr_extended / (wavenumbers_extended * 100)).imag

    # Cut extended spectrum
    nkks = nkks_extended[200:-200]
    return nprs, nkks


def calculate_Qext_curves(nprs, nkks, alpha0, gamma, wavenumbers):
    def mie_hulst_extinction(rho, tanbeta):
        beta = np.arctan(tanbeta)
        cosbeta = np.cos(beta)
        return (
            2
            - 4 * np.e ** (-rho * tanbeta) * (cosbeta / rho) * np.sin(rho - beta)
            - 4
            * np.e ** (-rho * tanbeta)
            * (cosbeta / rho) ** 2
            * np.cos(rho - 2 * beta)
            + 4 * (cosbeta / rho) ** 2 * np.cos(2 * beta)
        )

    rho = np.array(
        [
            alpha0val * (1 + gammaval * nkks) * (wavenumbers * 100)
            for alpha0val in alpha0
            for gammaval in gamma
        ]
    )
    tanbeta = [nprs / (1 / gammaval + nkks) for _ in alpha0 for gammaval in gamma]
    Qext = np.array(
        [
            mie_hulst_extinction(rhoval, tanbetaval)
            for rhoval, tanbetaval in zip(rho, tanbeta)  # noqa: B905
        ]
    )
    return Qext


def orthogonalize_Qext(Qext, reference):
    m = np.dot(reference, reference)
    norm = np.sqrt(m)
    rnorm = reference / norm
    s = np.dot(Qext, rnorm)
    Qext_orthogonalized = Qext - s[:, np.newaxis] * rnorm[np.newaxis, :]
    return Qext_orthogonalized


def compress_Mie_curves(Qext_orthogonalized, numComp):
    svd = TruncatedSVD(
        n_components=numComp, n_iter=7, random_state=42
    )  # Self.ncomp needs to be specified
    svd.fit(Qext_orthogonalized)
    badspectra = svd.components_[0:numComp, :]
    return badspectra


def cal_ncomp(reference, wavenumbers, explainedVarLim, alpha0, gamma):
    nprs, nkks = calculate_complex_n(reference, wavenumbers)
    Qext = calculate_Qext_curves(nprs, nkks, alpha0, gamma, wavenumbers)
    Qext_orthogonalized = orthogonalize_Qext(Qext, reference)
    maxNcomp = reference.shape[0] - 1
    svd = TruncatedSVD(n_components=min(maxNcomp, 30), n_iter=7, random_state=42)
    svd.fit(Qext_orthogonalized)
    lda = np.array(
        [
            (sing_val**2) / (Qext_orthogonalized.shape[0] - 1)
            for sing_val in svd.singular_values_
        ]
    )
    explainedVariance = 100 * lda / np.sum(lda)
    explainedVariance = np.cumsum(explainedVariance)
    numComp = np.argmax(explainedVariance > explainedVarLim) + 1
    return numComp


class ME_EMSCFeature(SelectColumn):
    InheritEq = True


class ME_EMSCModel(SelectColumn):
    InheritEq = True


class _ME_EMSC(CommonDomainOrderUnknowns, CommonDomainRef):
    def __init__(
        self,
        reference,
        weights,
        ncomp,
        alpha0,
        gamma,
        maxNiter,
        fixedNiter,
        positiveRef,
        domain,
    ):
        CommonDomainOrderUnknowns.__init__(self, domain)
        CommonDomainRef.__init__(self, reference, domain)
        assert len(reference) == 1
        self.weights = weights
        self.ncomp = ncomp
        self.alpha0 = alpha0
        self.gamma = gamma
        self.maxNiter = maxNiter
        self.fixedNiter = fixedNiter
        self.positiveRef = positiveRef

    def transformed(self, X, wavenumbers):
        # wavenumber have to be input as sorted
        # compute average spectrum from the reference

        def make_basic_emsc_mod(ref_X):
            N = wavenumbers.shape[0]
            m0 = -2.0 / (wavenumbers[0] - wavenumbers[N - 1])
            c_coeff = 0.5 * (wavenumbers[0] + wavenumbers[N - 1])
            M_basic = []
            for x in range(0, 3):
                M_basic.append((m0 * (wavenumbers - c_coeff)) ** x)
            M_basic.append(ref_X)  # always add reference spectrum to the model
            M_basic = np.vstack(M_basic).T
            return M_basic

        def cal_emsc_basic(M_basic, spectrum):
            m = np.linalg.lstsq(M_basic, spectrum, rcond=-1)[0]
            corrected = spectrum
            for x in range(0, 3):
                corrected = corrected - (m[x] * M_basic[:, x])
            corrected = corrected / m[3]
            scaled_spectrum = corrected
            return scaled_spectrum

        def make_emsc_model(badspectra, referenceSpec):
            M = np.ones([len(wavenumbers), self.ncomp + 2])
            M[:, 1 : self.ncomp + 1] = np.array([spectrum for spectrum in badspectra.T])
            M[:, self.ncomp + 1] = referenceSpec
            return M

        def cal_emsc(M, X):
            correctedspectra = np.zeros((X.shape[0], X.shape[1] + M.shape[1]))
            for i, rawspectrum in enumerate(X):
                m = np.linalg.lstsq(M, rawspectrum, rcond=-1)[0]
                corrected = rawspectrum
                for x in range(0, 1 + self.ncomp):
                    corrected = corrected - (m[x] * M[:, x])
                corrected = corrected / m[1 + self.ncomp]
                corrected[np.isinf(corrected)] = (
                    np.nan
                )  # fix values caused by zero weights
                corrected = np.hstack((corrected, m))  # append the model parameters
                correctedspectra[i] = corrected

            params = correctedspectra[:, -(self.ncomp + 2) :]
            res = X - np.dot(params, M.T)  # Have to check if this is correct FIXME
            return correctedspectra, res

        def iteration_step(spectrum, reference, wavenumbers, M_basic, alpha0, gamma):
            # scale with basic EMSC:
            reference = cal_emsc_basic(M_basic, reference)
            # some BLAS implementation can raise an exception in the upper call (MKL)
            # while some other only return an array of NaN (OpenBLAS), therefore
            # raise an exception manually
            if np.all(np.isnan(reference)):
                raise np.linalg.LinAlgError()

            # Apply weights
            reference = reference * wei_X
            reference = reference[0]

            # set negative parts to zero
            nonzeroReference = reference.copy()
            nonzeroReference[nonzeroReference < 0] = 0

            if self.positiveRef:
                reference = nonzeroReference

            # calculate Qext-curves
            nprs, nkks = calculate_complex_n(nonzeroReference, wavenumbers)
            Qext = calculate_Qext_curves(nprs, nkks, alpha0, gamma, wavenumbers)
            Qext = orthogonalize_Qext(Qext, reference)

            badspectra = compress_Mie_curves(Qext, self.ncomp)

            # build ME-EMSC model
            M = make_emsc_model(badspectra, reference)

            # calculate parameters and corrected spectra
            newspectrum, res = cal_emsc(M, spectrum)

            return newspectrum, res

        def iterate(
            spectra,
            correctedFirsIteration,
            residualsFirstIteration,
            wavenumbers,
            M_basic,
            alpha0,
            gamma,
        ):
            newspectra = np.full(correctedFirsIteration.shape, np.nan)
            numberOfIterations = np.full(spectra.shape[0], np.nan)
            residuals = np.full(spectra.shape, np.nan)
            RMSEall = np.full([spectra.shape[0]], np.nan)
            for i in range(correctedFirsIteration.shape[0]):
                corrSpec = correctedFirsIteration[i]
                rawSpec = spectra[i, :]
                rawSpec = rawSpec.reshape(1, -1)
                RMSE = [
                    round(
                        np.sqrt(
                            (1 / len(residualsFirstIteration[i, :]))
                            * np.sum(residualsFirstIteration[i, :] ** 2)
                        ),
                        4,
                    )
                ]
                for iterationNumber in range(2, self.maxNiter + 1):
                    try:
                        newSpec, res = iteration_step(
                            rawSpec,
                            corrSpec[: -self.ncomp - 2],
                            wavenumbers,
                            M_basic,
                            alpha0,
                            gamma,
                        )
                    except np.linalg.LinAlgError:
                        newspectra[i, :] = np.full(
                            [rawSpec.shape[1] + self.ncomp + 2], np.nan
                        )
                        residuals[i, :] = np.full(rawSpec.shape, np.nan)
                        RMSEall[i] = np.nan
                        break
                    corrSpec = newSpec[0, :]
                    rmse = round(np.sqrt((1 / len(res[0, :])) * np.sum(res**2)), 4)
                    RMSE.append(rmse)
                    # Stop criterion
                    if iterationNumber == self.maxNiter:
                        newspectra[i, :] = corrSpec
                        numberOfIterations[i] = iterationNumber
                        residuals[i, :] = res
                        RMSEall[i] = RMSE[-1]
                        break
                    elif self.fixedNiter and iterationNumber < self.fixedNiter:
                        continue
                    elif (
                        iterationNumber == self.maxNiter
                        or iterationNumber == self.fixedNiter
                    ):
                        newspectra[i, :] = corrSpec
                        numberOfIterations[i] = iterationNumber
                        residuals[i, :] = res
                        RMSEall[i] = RMSE[-1]
                        break
                    elif iterationNumber > 2 and not self.fixedNiter:
                        if (rmse == RMSE[-2] and rmse == RMSE[-3]) or rmse > RMSE[-2]:
                            newspectra[i, :] = corrSpec
                            numberOfIterations[i] = iterationNumber
                            residuals[i, :] = res
                            RMSEall[i] = RMSE[-1]
                            break
            return newspectra, RMSEall, numberOfIterations

        ref_X = interpolate_extend_to(self.reference, wavenumbers)
        ref_X = ref_X[0]

        wei_X = weighted_wavenumbers(self.weights, wavenumbers)

        ref_X = ref_X * wei_X
        ref_X = ref_X[0]

        nonzeroReference = ref_X
        nonzeroReference[nonzeroReference < 0] = 0

        if self.positiveRef:
            ref_X = nonzeroReference

        resonant = True  # Possibility for using the 2008 version

        if resonant:  # if this should be any point, we need to terminate after 1 iteartion for the non-resonant one
            nprs, nkks = calculate_complex_n(ref_X, wavenumbers)
        else:
            npr = np.zeros(len(wavenumbers))
            nprs = npr / (wavenumbers * 100)
            nkks = np.zeros(len(wavenumbers))

        # For the first iteration, make basic EMSC model
        M_basic = make_basic_emsc_mod(
            ref_X
        )  # Consider to make the M_basic in the init since this one does not change.

        # Calculate scattering curves for ME-EMSC
        Qext = calculate_Qext_curves(nprs, nkks, self.alpha0, self.gamma, wavenumbers)
        Qext = orthogonalize_Qext(Qext, ref_X)
        badspectra = compress_Mie_curves(Qext, self.ncomp)

        # Establish ME-EMSC model
        M = make_emsc_model(badspectra, ref_X)

        # Correcting all spectra at once for the first iteration
        newspectra, res = cal_emsc(M, X)

        if self.fixedNiter == 1 or self.maxNiter == 1:
            res = np.array(res)
            numberOfIterations = np.ones([1, newspectra.shape[0]])
            RMSEall = [
                round(np.sqrt((1 / res.shape[1]) * np.sum(res[specNum, :] ** 2)), 4)
                for specNum in range(newspectra.shape[0])
            ]  # ADD RESIDUALS!!!!! FIXME
            newspectra = np.hstack(
                (
                    newspectra,
                    numberOfIterations.reshape(-1, 1),
                    np.array(RMSEall).reshape(-1, 1),
                )
            )
            return newspectra

        # Iterate
        newspectra, RMSEall, numberOfIterations = iterate(
            X, newspectra, res, wavenumbers, M_basic, self.alpha0, self.gamma
        )
        newspectra = np.hstack(
            (newspectra, numberOfIterations.reshape(-1, 1), RMSEall.reshape(-1, 1))
        )
        return newspectra

    def __eq__(self, other):
        return (
            CommonDomainRef.__eq__(self, other)
            and self.ncomp == other.ncomp
            and np.array_equal(self.alpha0, other.alpha0)
            and np.array_equal(self.gamma, other.gamma)
            and self.maxNiter == other.maxNiter
            and self.fixedNiter == other.fixedNiter
            and self.positiveRef == other.positiveRef
            and (
                self.weights == other.weights
                if not isinstance(self.weights, Table)
                else table_eq_x(self.weights, other.weights)
            )
        )

    def __hash__(self):
        weights = (
            self.weights
            if not isinstance(self.weights, Table)
            else subset_for_hash(self.weights.X)
        )
        return hash(
            (
                CommonDomainRef.__hash__(self),
                weights,
                self.ncomp,
                tuple(self.alpha0),
                tuple(self.gamma),
                self.maxNiter,
                self.fixedNiter,
                self.positiveRef,
            )
        )


class MissingReferenceException(Exception):
    pass


class ME_EMSC(Preprocess):
    def __init__(
        self,
        reference=None,
        weights=None,
        ncomp=False,
        n0=np.linspace(1.1, 1.4, 10),  # noqa: B008
        a=np.linspace(2, 7.1, 10),  # noqa: B008
        h=0.25,
        max_iter=30,
        fixed_iter=False,
        positive_reference=True,
        output_model=False,
        ranges=None,
    ):
        # the first non-kwarg can not be a data table (Preprocess limitations)
        # ranges could be a list like this [[800, 1000], [1300, 1500]]
        if reference is None:
            raise MissingReferenceException()
        self.reference = reference
        if len(self.reference) > 1:
            self.reference = average_table_x(self.reference)
        self.weights = weights
        self.ncomp = ncomp
        self.output_model = output_model
        explainedVariance = 99.96

        self.maxNiter = max_iter
        self.fixedNiter = fixed_iter
        self.positiveRef = positive_reference

        self.n0 = n0
        self.a = a
        self.h = h

        self.alpha0 = (4 * np.pi * self.a * (self.n0 - 1)) * 1e-6
        self.gamma = (
            self.h
            * np.log(10)
            / (4 * np.pi * 0.5 * np.pi * (self.n0 - 1) * self.a * 1e-6)
        )

        if not self.ncomp:
            wavenumbers_ref = np.array(sorted(getx(self.reference)))
            ref_X = interpolate_extend_to(self.reference, wavenumbers_ref)[0]
            self.ncomp = cal_ncomp(
                ref_X, wavenumbers_ref, explainedVariance, self.alpha0, self.gamma
            )
        else:
            self.explainedVariance = False

    def __call__(self, data):
        # creates function for transforming data
        common = _ME_EMSC(
            reference=self.reference,
            weights=self.weights,
            ncomp=self.ncomp,
            alpha0=self.alpha0,
            gamma=self.gamma,
            maxNiter=self.maxNiter,
            fixedNiter=self.fixedNiter,
            positiveRef=self.positiveRef,
            domain=data.domain,
        )
        # takes care of domain column-wise, by above transformation function
        atts = [
            a.copy(compute_value=ME_EMSCFeature(i, common))
            for i, a in enumerate(data.domain.attributes)
        ]
        model_metas = []
        n_badspec = self.ncomp
        # Check if function knows about bad spectra
        used_names = set(
            [var.name for var in data.domain.variables + data.domain.metas]
        )
        if self.output_model:
            i = len(data.domain.attributes)
            for o in range(1):
                n = get_unique_names(used_names, "EMSC parameter " + str(o))
                model_metas.append(
                    Orange.data.ContinuousVariable(
                        name=n, compute_value=ME_EMSCModel(i, common)
                    )
                )
                i += 1
            for o in range(n_badspec):
                n = get_unique_names(used_names, "EMSC parameter bad spec " + str(o))
                model_metas.append(
                    Orange.data.ContinuousVariable(
                        name=n, compute_value=ME_EMSCModel(i, common)
                    )
                )
                i += 1
            n = get_unique_names(used_names, "EMSC scaling parameter")
            model_metas.append(
                Orange.data.ContinuousVariable(
                    name=n, compute_value=ME_EMSCModel(i, common)
                )
            )
            i += 1
            n = get_unique_names(used_names, "Number of iterations")
            model_metas.append(
                Orange.data.ContinuousVariable(
                    name=n, compute_value=ME_EMSCModel(i, common)
                )
            )
            i += 1
            n = get_unique_names(used_names, "RMSE")
            model_metas.append(
                Orange.data.ContinuousVariable(
                    name=n, compute_value=ME_EMSCModel(i, common)
                )
            )
        domain = Orange.data.Domain(
            atts, data.domain.class_vars, data.domain.metas + tuple(model_metas)
        )
        return data.from_table(domain, data)
