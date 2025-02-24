__all__ = ["I_dl_domega_dphi", "I_s_lambda", "integrated_I_s"]

import ctypes
import functools
import os
import sys
from logging import warning

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import constants as const
from scipy import LowLevelCallable, integrate
from scipy.interpolate import CubicSpline

from doppler_dimming_lib.c_libraries.c_lib import get_c_library
from doppler_dimming_lib.spectra.spectra import spectrum_config

# from doppler_dimming_lib.spectra.spectra import set_spectrum_level
from doppler_dimming_lib.utils import N_e_analytical, timeit


class args_struct(ctypes.Structure):
    """Class used for passing arguments to c library

    Args:
        ctypes.Structure: parent class
    """

    # structure passed to C library
    _fields_ = [
        ("n_points", ctypes.c_int),
        ("c_wls", ctypes.POINTER(ctypes.c_double)),
        ("c_Flambdas", ctypes.POINTER(ctypes.c_double)),
        ("los_n_points", ctypes.c_int),
        ("los_positions", ctypes.POINTER(ctypes.c_double)),
        ("los_densities", ctypes.POINTER(ctypes.c_double)),
    ]


def _get_user_data_ptr(
    los_positions=np.zeros(shape=1), los_densities=np.zeros(shape=1)
):
    wls, Fls = spectrum_config.COMPLETE_SPECTRUM
    los_positions = np.ascontiguousarray(los_positions, dtype=np.double)
    los_densities = np.ascontiguousarray(los_densities, dtype=np.double)

    user_data = args_struct(
        ctypes.c_int(wls.shape[0]),
        ctypes.cast(wls.ctypes.data, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(Fls.ctypes.data, ctypes.POINTER(ctypes.c_double)),
        los_positions.shape[0],
        ctypes.cast(los_positions.ctypes.data, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(los_densities.ctypes.data, ctypes.POINTER(ctypes.c_double)),
    )

    user_data_ptr = ctypes.cast(ctypes.pointer(user_data), ctypes.c_void_p)

    return user_data_ptr


def _get_omega_boundary(
    x: float, rho: float, lambda_cm: float, T_e: float, W_cm: float, component: int
) -> list:
    """Util function that returns the integration limits for omega,
    used by scipy.integrate.nquad

    Args:
        x (float): LOS variable
        rho (float): radial distance from sun center in R_sun
        lambda_cm (float): wavelenght
        T_e (float): electron T
        W_cm (float): solar wind speed in cm/s

    Returns:
        list: omega boundaries
    """
    cos_omega_star = np.cos(np.arcsin(1.0 / np.sqrt(x * x + rho * rho)))

    return [cos_omega_star, 1.0]  # cos_omega


def I_dl_domega_dphi(
    rho: float, _lambda: float, T_e: float, W: float, verbose: bool = True
) -> float:
    """Interior integral, separated to allow calculations without integrating in dLOS

    Args:
        rho (float): heliocentric distance in solar radii
        _lambda (float): wavelenght in angstrom
        T_e (float): electron temperature in Kelvin
        W (float): wind speed in km/s
        verbose (bool, optional): print additional info at each step for debugging purposes. Defaults to True.

    Returns:
        float: photospheric irradiance integrated over solar disc
    """
    if verbose:
        print(f"\nCalculating I_dl_domega_dphi...")
        print(f"{_lambda = } Angstrom")
        print(f"{rho = } R_sun")
        print(f"{T_e = } K")
        print(f"{W = } Km/s")

    lambda_cm = _lambda * u.Angstrom.to(u.cm)
    W_cm = (W * u.km / u.s).to(u.cm / u.s).value
    args = [rho, lambda_cm, T_e, W_cm]

    global C_LIBRARY
    integrand = LowLevelCallable(
        C_LIBRARY.I_dlambda_domega, user_data=_get_user_data_ptr()
    )

    integration_limits = [
        [0.0, 2.0 * np.pi],  # phi
        _get_omega_boundary,  # cos_omega
    ]
    opts = [
        {
            "points": [0, 2 * np.pi],
        },  # phi
        {
            "points": [0, 1],
        },  # cos_omega
    ]
    result = (
        integrate.nquad(
            # functions.integrand,
            integrand,
            integration_limits,
            opts=opts,
            args=[0]
            + args
            + [
                0
            ],  # x is now a parameter set to 0 (plane of sky), 0 is radial component
        )[0]
        + integrate.nquad(
            # functions.integrand,
            integrand,
            integration_limits,
            opts=opts,
            args=[0] + args + [1],  # 1 is tangential
        )[0]
    )

    return result


def I_s_lambda(*args, **kwargs) -> float:
    """Wrapper for integrated_I_s used to compute total brightness by passing component=3 to integrated_I_s.

    Returns:
        float: integrated total brightness
    """
    return integrated_I_s(*args, **kwargs, component=3)


# @timeit
def integrated_I_s(
    rho: float,
    _lambda: float,
    T_e: float,
    wind_speed: float,
    component: int,
    min_x: float = None,
    max_x: float = None,
    N_e_function=N_e_analytical,
    verbose=True,
) -> float:
    """Integrate thomson scattering over photospheric wavelenghts, solar disc dimension and line of sight (main integral from Cram 1976.

    Args:
        rho (float): heliocentric distance in solar radii
        _lambda (float): wavelenght in angstrom
        T_e (float): electron temperature
        W (float): wind speed in km/s
        component (int): polarized component to compute. With respect to solar limb, 0 is radial, 1 is tangent and 3 is the sum (total brightness)
        min_x (float, optional): integration limit for LOS away from observer in solar radii. Defaults to 10.
        max_x (float, optional): integration limit for LOS towards observer in solar radii. Defaults to 10.
        N_e_function (callable, optional): function used to calculate the electron density as a function of heliocentric distance. Defaults to utils.N_e_analytical.
        verbose (bool, optional): print additional info at each step. Defaults to True.

    Returns:
        float: integrated emission
    """
    # ----------
    # Tweakables, omega is dependent on x
    # ----------
    subinterval_limit = 10
    max_chebyshev_order = 5
    # ----------

    if min_x is None:
        min_x = 10.0
        min_x = rho**3
        min_x = np.inf
    if max_x is None:
        max_x = 10.0
        max_x = rho**3
        max_x = np.inf

    # sanity check on rho (can't be a solar radius or less)
    if rho <= 1:
        # print(f"ERROR: can't integrate rho <= 1 ({rho} provided)")
        warning(f"Integral undefined for rho <= 1 Rsun ({rho} Rsun). Defaulting to 0.")
        return 0

    if verbose:
        print(f"\nCalculating I_s_lambda...")
        print(f"{_lambda = } Angstrom")
        print(f"{rho = } R_sun")
        print(f"{T_e = } K")
        print(f"{wind_speed = } Km/s")

    lambda_cm = _lambda * u.Angstrom.to(u.cm)
    W_cm_s = (wind_speed * u.km / u.s).to(u.cm / u.s).value
    args = [rho, lambda_cm, T_e, W_cm_s, component]

    normalization = const.R_sun.to(u.cm).value

    global C_LIBRARY

    # SLOW
    if False:
        los_positions = np.linspace(-10, 10, 1000, dtype=np.double)
        rs = np.sqrt(
            los_positions * los_positions + rho * rho
        )  # * np.sign(los_positions)
        los_N_e = np.array([N_e_function(r) for r in rs], dtype=np.double)

        assert len(los_positions) == len(los_N_e)

        integrand = LowLevelCallable(
            C_LIBRARY.I_dlambda_domega_dx,
            user_data=_get_user_data_ptr(los_positions, los_N_e),
        )

        def get_omega_boundary(*args):
            # integrand(phi, omega, x, rho, lambda, Te, W, component)
            # range_phi(omega, x, rho, ...)
            # range_omega(x, rho, ...)
            x = args[0]
            rho = args[1]
            cos_omega_star = np.cos(np.arcsin(1.0 / np.sqrt(x * x + rho * rho)))
            return [cos_omega_star, 1.0]

        ranges = ranges = [
            [0.0, 2.0 * np.pi],  # phi
            # _get_omega_boundary,  # cos_omega
            get_omega_boundary,  # cos_omega
            [-min_x, max_x],  # x
        ]

        opts = [
            {"epsrel": 0.01},  # phi
            {"epsrel": 0.01},  # cos_omega
            {"epsrel": 0.01},  # x
        ]

        normalization = const.R_sun.to(u.cm).value

        result = integrate.nquad(
            # functions.integrand,
            integrand,
            ranges,
            opts=opts,
            args=args,
        )[0]

        return result * normalization

    if not callable(N_e_function):
        integrand = LowLevelCallable(
            C_LIBRARY.I_dlambda_domega, user_data=_get_user_data_ptr()
        )
        xs = N_e_function[0]
        Nes = N_e_function[1]

        Is = np.zeros_like(xs)
        for i in range(len(Is)):
            ranges = [
                [0.0, 2.0 * np.pi],  # phi
                _get_omega_boundary,  # cos_omega
            ]
            opts = [{"epsrel": 0.01}, {"epsrel": 0.01}]
            Is[i] = (
                Nes[i]
                * integrate.nquad(
                    # functions.integrand,
                    integrand,
                    ranges,
                    opts=opts,
                    args=[xs[i]] + args,
                )[0]
            )

        # trapezoid integration
        sum = 0
        for i in range(len(xs) - 1):
            sum += np.abs(xs[i + 1] - xs[i]) * (Is[i] + Is[i + 1])

        return normalization * 0.5 * sum

    else:
        integrand = LowLevelCallable(
            C_LIBRARY.I_dlambda_domega, user_data=_get_user_data_ptr()
        )

        def integrand_dx(x):
            # integral is over x, not r!
            ranges = [
                [0.0, 2.0 * np.pi],  # phi
                _get_omega_boundary,  # cos_omega
            ]

            # dictionary of singularities, first = innermost integral
            opts = [
                {
                    # "maxp1": max_chebyshev_order,
                    # "limit": subinterval_limit,
                    # "points": [0, 2 * np.pi],
                    "epsrel": 0.01,
                },  # phi
                {
                    # "maxp1": max_chebyshev_order,
                    # "limit": subinterval_limit,
                    # "points": [0, 1],
                    "epsrel": 0.01,
                },  # cos_omega
            ]

            r = np.sqrt(x * x + rho * rho)

            N_e = N_e_function(r)  # * r / x  #  dx = r/sqrt(r^2 - rho^2) dr

            return (
                N_e
                * integrate.nquad(
                    # functions.integrand,
                    integrand,
                    ranges,
                    opts=opts,
                    args=[x] + args,
                )[0]
            )

        # inf_bound = np.sqrt(min_r * min_r - rho * rho)
        # sup_bound = np.sqrt(max_r * max_r - rho * rho)

        sun_to_inf = integrate.quad(
            integrand_dx,
            0.0,
            min_x,
            # limit=subinterval_limit,
            # maxp1=max_chebyshev_order,
            # epsrel=0.01,  # absolute errorr
            # points=[0, np.sqrt(rho * rho)],
            full_output=1,
        )

        symmetric_corona = False

        if symmetric_corona:
            return normalization * (2.0 * sun_to_inf[0])
        else:
            sun_to_sup = integrate.quad(
                integrand_dx,
                0.0,
                max_x,
                # limit=subinterval_limit,
                # maxp1=max_chebyshev_order,
                # epsrel=0.01,  # absolute errorr
                # points=[0, np.sqrt(rho * rho)],
                full_output=1,
            )

            return normalization * (sun_to_sup[0] + sun_to_inf[0])


if __name__ == "__main__":
    # print(I_s_lambda(2, 3700, 1.0e6, 0))

    if False:
        I1 = sigma.ufloat(3300, 10)
        I2 = sigma.ufloat(3000, 10)
        R = I1 / I2
        # R = sigma.ufloat(1.1, 0.001)

        for function in [I_s_lambda, I_dl_domega_dphi]:
            R_as_spline = get_R_as_spline(
                function,
                rho=2,
                W=0,
            )

            T = get_T_from_R(R_as_spline, R)
            print(T)

        Rs = np.linspace(1.06, 1.12, 10)

        fig, ax = plt.subplots(dpi=200)
        ax.plot(Rs, R_as_spline(Rs))
        for val in [
            T.nominal_value,
            T.nominal_value + T.std_dev,
            T.nominal_value - T.std_dev,
        ]:
            ax.axhline(val)
        plt.show()


C_LIBRARY = get_c_library()

# read it once and save it as global saves time
# GLOBAL_SAV_MODEL = read_sav_model("../idl_integration/cor3d_test_512_mod.sav")
# GLOBAL_SAV_N = np.ascontiguousarray(GLOBAL_SAV_MODEL[3], dtype=np.double)
