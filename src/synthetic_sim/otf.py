"""!
@file otf.py
@brief Contains functions to simulate the optical transfer function of the
optical system, with high configurability as set by the parameters of the
system.

@details Code provided by a former student.
"""

import numpy as np
from dataclasses import dataclass
from scipy.special import jv
from typing import Callable
from typing import Sequence


@dataclass
class PsfParameters:
    """!
    @brief Class to store PSF parameters.

    @details Class to store the parameters used to evaluate an approximate
    Gibson-Lanni PSF. Default values are provided except for the PSF size.
    """

    size: tuple  # (nx, ny)
    num_basis: int = 100  # the number of basis functions used
    num_samp: int = 1000  # number of samples to determine basis coefficients
    oversampling: float = 2.0  # oversampling ratio
    NA: float = 1.4  # numerical aperture
    lambda0: float = 610.0e-9  # emission wavelength (metres)
    mag: float = 100.0  # magnification
    ns: float = 1.33  # specimen refractive index (RI)
    ng0: float = 1.5  # coverslip RI, design value
    ng: float = 1.5  # coverslip RI, experimental value
    ni0: float = 1.5  # immersion RI, design value
    ni: float = 1.5  # immersion RI, experimental value
    ti0: float = 150e-6  # working distance, design value (m)
    tg0: float = 170e-6  # coverslip thickness, design value (m)
    tg: float = 170e-6  # coverslip thickness, experimental value (m)
    res_lateral: float = 100e-9  # lateral pixel size (m)
    particle_z: Sequence[float] = (2000e-9,)  # position of particle (m)
    z: Sequence[float] = (0.0,)  # change in working distance (m)
    bessel: Callable = jv  # used to evaluate bessel functions


def calc_psf(params):
    """!
    @brief Calculate an approximate Gibson-Lanni PSF based on the
    parameters provided.

    @details Code ported from MATLAB, original copyright Jizhou Li, 2016,
    The Chinese University of Hong Kong.

    @param params (PsfParameters) - dataclass storing the PSF parameters
    @returns np.ndarray representing the PSF
    """
    nx, ny = params.size
    nz = len(params.z)
    bessel = params.bessel
    x0 = (nx - 1) / 2
    y0 = (ny - 1) / 2
    z0 = (nz - 1) / 2
    max_radius = np.round(np.sqrt((nx - x0) ** 2 + (ny - y0) ** 2)) + 1
    R = np.arange(params.oversampling * max_radius) / params.oversampling
    Ti = params.ti0 + np.array(params.z)  # shape (Nz,)
    a = min(
        x / params.NA
        for x in [
            params.NA,
            params.ns,
            params.ni,
            params.ni0,
            params.ng0,
            params.ng,
        ]
    )
    rho = np.linspace(0, a, params.num_samp)  # shape (K,)

    # 1. Approximate exp(iW) as Bessel series
    k0 = 2 * np.pi / params.lambda0
    r = R * params.res_lateral
    A = k0 * params.NA * r
    Aa = A * a  # shape (K,)

    k_min = 2 * np.pi / 545e-9  # min wavelength
    NA_max = 1.4  # max numerical aperture
    an = (
        (3 * np.arange(1, params.num_basis + 1) - 2)  # shape (M,)
        * (params.NA / NA_max)
        * (k0 / k_min)
    )
    an_rho = np.outer(an, rho)  # shape (M, K)
    J = bessel(0, an_rho)  # shape (M, K)

    J0_A = bessel(0, Aa)  # shape (K,)
    J1_A = A * bessel(1, Aa)  # shape (K,)

    anJ0A = np.outer(J0_A, an)  # shape (K, M)
    an_a = an * a  # shape (M,)
    an2 = an**2  # shape (M,)
    B1ana = bessel(1, an_a)  # shape (M,)
    B0ana = bessel(0, an_a)  # shape (M,)

    Ele = anJ0A * B1ana[np.newaxis, :] - np.outer(J1_A, B0ana)  # shape (K, M)
    domin = an2 - (A**2)[:, np.newaxis]  # shape (K, M)
    Ele *= a / domin

    particle_z = np.atleast_1d(params.particle_z)
    c1 = params.ns * particle_z  # shape (Npz,)
    c2 = params.ni * (Ti - params.ti0)  # shape (Nz,)
    c3 = params.ng * (params.tg - params.tg0)  # scalar
    c4 = params.NA * rho  # shape (K,)

    opd_s = c1[:, np.newaxis] * np.sqrt(
        1 - (c4 / params.ns) ** 2
    )  # shape (Npz, K)
    opd_i = c2[:, np.newaxis] * np.sqrt(
        1 - (c4 / params.ni) ** 2
    )  # shape (Nz, K)
    opd_g = c3 * np.sqrt(1 - (c4 / params.ng) ** 2)  # shape (K,)
    opd = opd_i + opd_s[:, np.newaxis, :] + opd_g  # shape (Npz, Nz, K)

    # determine coefficients
    W = k0 * opd  # shape (Npz, Nz, K)
    Ffun = np.cos(W) + 1j * np.sin(W)  # shape (Npz, Nz, K)
    fun2 = Ffun.reshape(
        Ffun.shape[0] * Ffun.shape[1], Ffun.shape[2]
    )  # shape (Npz*Nz, K)
    Ci, *_ = np.linalg.lstsq(J.T, fun2.T, rcond=None)  # shape (M, Npz*Nz)
    Ci = Ci.reshape(Ci.shape[0], *Ffun.shape[:2])  # shape (M, Npz, Nz)

    # 2. get PSF in each slice
    ciEle = np.tensordot(Ele, Ci, 1)
    psf0 = np.abs(ciEle) ** 2  # shape (K, Npz, Nz)

    # 3. apply axial symmetry
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    r_pixel = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)  # shape (nx, ny)
    index = np.floor(r_pixel * params.oversampling).astype(
        int
    )  # shape (nx, ny)
    dis_r = (r_pixel - R[index]) * params.oversampling

    npz = len(particle_z)
    psf = np.zeros([nx, ny, nz, npz])
    for zi in range(nz):
        for pzi in range(npz):
            h = psf0[:, pzi, zi]
            slice = h[index + 1] * dis_r + h[index] * (1 - dis_r)
            psf[:, :, zi, pzi] = slice

    psf /= np.max(psf.ravel())
    return psf
