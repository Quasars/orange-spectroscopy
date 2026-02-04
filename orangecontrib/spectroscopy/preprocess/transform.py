from enum import Enum

import numpy as np
import Orange.data
from Orange.preprocess.preprocess import Preprocess

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.preprocess.utils import (
    SelectColumn,
    CommonDomainRef,
    WrongReferenceException,
    replace_infs,
)


class SpecTypes(Enum):
    """
    Spectral types possibly supported by spectral transforms
    """

    ABSORBANCE = "Absorbance"
    TRANSMITTANCE = "Transmittance"
    SINGLECHANNEL = "Single Channel"


class AbsorbanceFeature(SelectColumn):
    InheritEq = True


class _AbsorbanceCommon(CommonDomainRef):
    def transformed(self, data):
        if self.reference is not None:
            # Calculate from single-channel data
            ref_X = self.interpolate_extend_to(self.reference, getx(data))
            if len(data):  # numpy does not like to divide shapes (0, b) by (a, b)
                absd = ref_X / data.X
                np.log10(absd, absd)
            else:
                absd = data.X.copy()
        else:
            # Calculate from transmittance data
            absd = np.log10(data.X)
            absd *= -1
        # Replace infs from either np.true_divide or np.log10
        return replace_infs(absd)

    def __eq__(self, other):
        # pylint: disable=useless-parent-delegation
        return super().__eq__(other)

    def __hash__(self):
        # pylint: disable=useless-parent-delegation
        return super().__hash__()


class TransformOptionalReference(Preprocess):
    def __init__(self, reference=None):
        if reference is not None and len(reference) != 1:
            raise WrongReferenceException("Reference data should have length 1")
        self.reference = reference

    def __call__(self, data):
        common = self._cl_common(self.reference, data.domain)
        newattrs = [
            Orange.data.ContinuousVariable(
                name=var.name, compute_value=self._cl_feature(i, common)
            )
            for i, var in enumerate(data.domain.attributes)
        ]
        domain = Orange.data.Domain(newattrs, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)


class Absorbance(TransformOptionalReference):
    """
    Convert data to absorbance.

    Set reference to calculate from single-channel spectra, otherwise convert from transmittance.

    Parameters
    ----------
    reference : reference single-channel (Orange.data.Table)
    """

    _cl_common = _AbsorbanceCommon
    _cl_feature = AbsorbanceFeature

    from_types = (SpecTypes.TRANSMITTANCE, SpecTypes.SINGLECHANNEL)


class TransmittanceFeature(SelectColumn):
    InheritEq = True


class _TransmittanceCommon(CommonDomainRef):
    def transformed(self, data):
        if self.reference is not None:
            # Calculate from single-channel data
            if len(data):  # numpy does not like to divide shapes (0, b) by (a, b)
                ref_X = self.interpolate_extend_to(self.reference, getx(data))
                transd = data.X / ref_X
            else:
                transd = data.X.copy()
        else:
            # Calculate from absorbance data
            transd = data.X.copy()
            transd *= -1
            np.power(10, transd, transd)
        # Replace infs from either np.true_divide or np.log10
        return replace_infs(transd)

    def __eq__(self, other):
        # pylint: disable=useless-parent-delegation
        return super().__eq__(other)

    def __hash__(self):
        # pylint: disable=useless-parent-delegation
        return super().__hash__()


class Transmittance(TransformOptionalReference):
    """
    Convert data to transmittance.

    Set reference to calculate from single-channel spectra, otherwise convert from absorbance.

    Parameters
    ----------
    reference : reference single-channel (Orange.data.Table)
    """

    from_types = (SpecTypes.ABSORBANCE, SpecTypes.SINGLECHANNEL)

    _cl_common = _TransmittanceCommon
    _cl_feature = TransmittanceFeature
