"""Material dispersion and the paper's measurement comb.

Everything else in the package takes the refractive-index model as a
callable, so a different resist is one function away; the default is
AZ4562, the photoresist of Xiao et al. (Light Sci. Appl. 11:323, 2022),
as a Cauchy fit to the dispersion of their Fig. S8 (average n ~ 1.63
over 400-1100 nm, their S2-1). The same two Cauchy coefficients are
hard-coded in the OpticStudio DLLs (us_mdl_rings.cpp) and in the
verification scripts -- keep the three in sync if the material changes.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: speed of light in micrometres per second (all lengths are in um)
C_UM_PER_S: float = 2.99792458e14

#: signature of a refractive-index model: lam_um -> n (real part)
IndexModel = Callable[[ArrayLike], NDArray[np.float64]]

#: the paper's 14 measurement wavelengths for sample S3 [um]:
#: 400-1100 nm in 50 nm steps, 800 nm omitted (their Fig. 2e / Fig. 4)
PAPER_COMB_14: NDArray[np.float64] = np.array(
    [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
     0.75, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10])


def n_az4562(lam_um: ArrayLike) -> NDArray[np.float64]:
    """Real refractive index of AZ4562 photoresist.

    Cauchy fit ``n = 1.594 + 0.01152 / lam^2`` to the paper's Fig. S8;
    the fit is to data ending at 1100 nm and extrapolates smoothly
    (n ~ 1.60) into the SWIR, but absorption is not modelled.

    Parameters
    ----------
    lam_um : array_like
        Vacuum wavelength(s) in micrometres.

    Returns
    -------
    NDArray[float64]
        n at each wavelength, same shape as the input.
    """
    lam = np.asarray(lam_um, dtype=float)
    return 1.594 + 0.01152 / lam ** 2
