import functools
import json
import os
import sys
from functools import lru_cache
from logging import error, info

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline

from doppler_dimming_lib.integrals import I_s_lambda
from doppler_dimming_lib.utils import N_e_analytical


def set_codex_transmittancies(
    approx_filter: bool = False,
    skipevery: int = 10,
    min_transmittancy: float = 0.001,
):
    """Reads the excel file containing the filter transmittancies and sets the dictionary containing arrays of wavelenghts and transmittancies for each filter, ordered by center wavelenght.

    Args:
        approx_filter (bool, optional): approximate filter transmissivity with a box function. This may some time, but results are often inconsistent. Defaults to False.
        skipevery (int, optional): sample one transmittance every n entries. Defaults to 10.
        min_transmittancy (float, optional): minimum transmittancy to consider, lower ones will be considered 0. Defaults to 0.001 (0.1%).
    """
    filter_ids = FILTER_CENTERS.keys()
    # filter_centers = FILTER_CENTERS.values()
    filter_name_cols = ["Filter " + filter_id for filter_id in filter_ids]
    filter_bp_cols = ["393 +/- 5 nm", "398 +/- 5 nm", "405 +/- 5 nm", "423 +/- 5 nm"]

    filter_transmittancies = {}

    for filter_name_col, filter_bp_col, filter_id in zip(
        filter_name_cols, filter_bp_cols, filter_ids
    ):
        if approx_filter:
            with open(
                os.path.join(
                    os.path.dirname(__file__),
                    _CODEX_FILTERS_DATA_PATH,
                    "approx_filter_params.json",
                ),
                "r",
            ) as f:

                params = json.load(f)
            min, max, top, center = params[filter_name_col].values()

            wavelenghts = np.linspace(min, max, 10)
            transmittancies = np.ones(len(wavelenghts)) * top / 100.0  # normalize

        else:  # convolve using xls file
            df = pd.read_excel(
                os.path.join(
                    os.path.dirname(__file__),
                    _CODEX_FILTERS_DATA_PATH,
                    "Measurements_Codex_Filters.xlsx",
                ),
                dtype=float,
                usecols=[0, 1, 3, 4, 6, 7, 9, 10, 12, 13],
            )
            df.dropna()

            df = df[df[filter_bp_col] > (min_transmittancy * 100)]
            wavelenghts = df[filter_name_col].to_numpy()
            transmittancies = df[filter_bp_col].to_numpy() / 100  # normalized

            wavelenghts = wavelenghts[::skipevery]
            transmittancies = transmittancies[::skipevery]

            wavelenghts = wavelenghts * 10  # *10 nm to angstrom conversion

            filter_transmittancies[filter_id] = [wavelenghts, transmittancies]

    if False:  # show filters
        fig, ax = plt.subplots()
        for id in filter_ids:
            ax.plot(*filter_transmittancies[id], label=f"Filter {id}")
        plt.legend()
        plt.show()

    global _FILTER_TRANSMITTANCIES
    _FILTER_TRANSMITTANCIES = filter_transmittancies


def convolve_codex_filter(_lambda: float, I, verbose: bool = True) -> float:
    """Convolve a function I with the codex filter at _lambda transmissivity. For each available filter transmissivity T, computes the sum of

    Args:
        _lambda (float): filter center in Angstrom.
        I (callable): function to convolve, argument is lambda in Armstrong
        verbose (bool, optional): print info about computation. Defaults to True.

    Returns:
        float: convolution between I(lambda) and the filter transmissivity
    """

    if _lambda not in FILTER_CENTERS.values():
        error(f"Invalid wavelenght provided for filter ({_lambda})")
        sys.exit()
    else:
        for filter_id, filter_center in FILTER_CENTERS.items():
            if _lambda == filter_center:
                matching_id = filter_id

    wavelenghts, transmittancies = _FILTER_TRANSMITTANCIES[matching_id]

    if verbose:
        info(f"Convolving filter response ({len(wavelenghts)} points...)")
    Is = np.array([I(x) for x in wavelenghts])

    # return np.sum(cumulative_trapezoid(wavelenghts, Is * transmittancies, initial=0))

    sum = 0
    for i in range(0, len(wavelenghts[:-1])):
        sum += (
            Is[i] * transmittancies[i] + Is[i + 1] * transmittancies[i + 1]
        ) * np.abs(
            wavelenghts[i + 1] - wavelenghts[i]
        )  # trapezoid
    sum *= 0.5

    if verbose:
        info("Done!")

    return sum


def simulate_codex_filter(
    filter_name: str,
    rho: float,
    T_e: float,
    wind_speed: float,
    N_e_function=N_e_analytical,
    verbose: bool = True,
) -> float:
    """Simulate the measurement from a CODEX filter by convolving its response with Cram, 1976 integral

    Args:
        filter_name (str): filter name, can be "T1", "S1", "T2" or "S2"
        rho (float): heliocentric distance in solar radii
        T_e (float): electron temperature in Kelvin
        wind_speed (float): wind speed in km/s
        N_e_function (callable, optional): electron density as a function of the heliocentric distance (in rsun). Defaults to utils.N_e_analytical.

    Returns:
        float: value measured with selected filter
    """

    def J_s_lambda(_lambda):
        """Wrapper for I_s_lambda for passing it to convolve_codex_filter"""
        return I_s_lambda(
            rho=rho,
            _lambda=_lambda,
            T_e=T_e,
            wind_speed=wind_speed,
            N_e_function=N_e_function,
            verbose=verbose,
        )  # verbose disabled here, too many prints

    J_s = convolve_codex_filter(
        FILTER_CENTERS[filter_name], I=J_s_lambda, verbose=verbose
    )

    return J_s


# @lru_cache
def simulate_codex_measure(
    rho: float,
    T_e: float,
    wind_speed: float,
    N_e_function=N_e_analytical,
    verbose: bool = True,
) -> np.ndarray:
    """Simulate the measurement from the four CODEX filters by convolving their response with Cram, 1976 integral

    Args:
        rho (float): heliocentric distance in solar radii
        T_e (float): electron temperature in Kelvin
        wind_speed (float): wind speed in km/s
        N_e_function (callable, optional): electron density as a function of the heliocentric distance (in rsun). Defaults to utils.N_e_analytical.

    Returns:
        np.ndarray(4): filters measured values, ordered by filter wavelength (T1=393, S1=398, T2=405, S2=423)
    """

    Js = np.array(
        [
            simulate_codex_filter(name, rho, T_e, wind_speed, N_e_function, verbose)
            for name in FILTER_CENTERS.keys()
        ],
        dtype=float,
    )
    return Js


def get_T_W_from_codex(rho: float, Js: list, p0: list) -> list:
    """Uses least squares to find the values of temperature and solar wind that better fit the input Js.

    Args:
        rho (float): heliocentric distance in solar radii
        Js (list): codex intesities at the four wavelenghts, ordered
        p0 (list): initial guess for temperature and wind speed

    Returns:
        list: fitted electron temperature and solar wind speed
    """
    bounds = [
        [0, 0],
        [1.0e7, 1000],
    ]
    bounds = [0, 1.0e7]
    ftol = 1.0e-3

    if True:  # use leastq

        def difference(params, rho, Js):
            T_e = params[0]
            W = params[1]
            W = 0

            simulated_Js = simulate_codex_measure(rho, T_e, W)
            return Js - simulated_Js

        result = optimize.least_squares(
            difference, p0, bounds=bounds, args=(rho, Js), ftol=ftol, verbose=2
        )

        return result.x

    else:  # use curve_fit

        def get_codex_measure(_lambda, T_e, W):
            def J_s_lambda(_lambda):
                return I_s_lambda(rho, _lambda, T_e, W, verbose=False)

            result = []
            for x in _lambda:
                result.append(convolve_codex_filter(x, J_s_lambda, approx_filter=False))
            return result
            return convolve_codex_filter(_lambda, J_s_lambda, approx_filter=False)

        wls = np.array(
            [T1_CENTER_CONST, S1_CENTER_CONST, T2_CENTER_CONST, S2_CENTER_CONST],
            dtype=float,
        )

        params, pcov = optimize.curve_fit(
            get_codex_measure,
            xdata=wls,
            ydata=Js,
            p0=p0,
            bounds=bounds,
        )

        return params


from doppler_dimming_lib.utils import timeit


# @functools.lru_cache
@functools.cache
def get_R_as_spline(integral_function, lambda1: float, lambda2: float, **kwargs):
    """Find the cubic spline interpolator of the ratio function(T1)/function(T2). All the parameters of function must be provided as kwargs.

    Args:
        function (callable): function to perform the ratio numerator/denominator
        lambda1 (float): numerator wavelenght in angstrom
        lambda2 (float): denominator wavelenght in angstrom

    Returns:
        PPoly: spline cubic interpolating 5.e5 < T < 3.e6
    """

    # print("Extracting R(T)...")

    num_of_samples = 10  # previous
    num_of_samples = 5  # found that 5 points are enough to approximate with good precision, see thesis

    # interpolate a cubic spline to R(T)
    Ts = np.linspace(5.0e5, 2.0e6, num_of_samples)
    Rs = []
    for T in Ts:
        R1 = integral_function(T_e=T, _lambda=lambda1, **kwargs, verbose=False)
        R2 = integral_function(T_e=T, _lambda=lambda2, **kwargs, verbose=False)
        Rs.append(R1 / R2)

    # rescale non-integrated spline into the range of integrated spline. Allows a more precise range without the need to integrate filters for all points, BUT TIME IS 3X to 6X INCREASED
    if False:
        for filter_id, filter_center in FILTER_CENTERS.items():
            if lambda1 == filter_center:
                filtername1 = filter_id
            if lambda2 == filter_center:
                filtername2 = filter_id

        rj1 = simulate_codex_filter(
            filter_name=filtername1, T_e=Ts[0], verbose=False, **kwargs
        ) / simulate_codex_filter(
            filter_name=filtername2, T_e=Ts[0], verbose=False, **kwargs
        )
        rj2 = simulate_codex_filter(
            filter_name=filtername1, T_e=Ts[-1], verbose=False, **kwargs
        ) / simulate_codex_filter(
            filter_name=filtername2, T_e=Ts[-1], verbose=False, **kwargs
        )

        Rs = np.array(Rs)
        Ts = np.array(Ts)
        rescaled_rs = (Rs - np.min(Rs)) / (np.max(Rs) - np.min(Rs)) * (rj2 - rj1) + rj1

        if False:  # show rescaling difference
            fig, ax = plt.subplots()
            ax.plot(Ts, Rs, "r+", label="original")
            ax.plot(Ts, rescaled_rs, "b+", label="rescaled")
            ax.plot([Ts[0], Ts[-1]], [rj1, rj2], label="two points")
            ax.legend()
            plt.show()

        Rs = rescaled_rs

    Rs, Ts = zip(*sorted(zip(Rs, Ts)))  # without this cubicspline doesnt work

    pol = CubicSpline(Rs, Ts)

    if False:
        fig, ax = plt.subplots(dpi=200, constrained_layout=True)
        ax.plot(Rs, Ts, "b+")
        ax.plot(Rs, pol(Rs), "r-")
        ax.set_xlabel("R")
        ax.set_ylabel(r"$T_e$")
        plt.show()

    return pol


def get_T_from_R(ratio: float, rho: float, **kwargs):
    """Retrieve the electron tempearture from a CODEX measured ratio between filters. First gets R as spline for the selected rho, then gets T from the fitted spline.

    Args:
        ratio (float): ratio between two filters S2/T2
        rho (float): heliocentric distance in solar radii
        W (float, optional): wind speed in km/s. Defaults to 0.

    Returns:
        uncertainties.ufloat: inferred electron temperature
    """

    lambda1 = FILTER_CENTERS["S2"]
    lambda2 = FILTER_CENTERS["T2"]

    pol = get_R_as_spline(I_s_lambda, lambda1, lambda2, rho=rho, **kwargs)
    # inferred_T = get_T_from_polyfit(pol, ratio)
    inferred_T = pol(ratio)

    return inferred_T


_CODEX_FILTERS_DATA_PATH = "./data/codex_filters/"

with open(
    os.path.join(
        os.path.dirname(__file__),
        _CODEX_FILTERS_DATA_PATH,
        "approx_filter_params.json",
    ),
    "r",
    encoding="utf-8",
) as read_file:
    _FILTER_PARAMS = json.load(read_file)

FILTER_CENTERS = {
    "T1": 3935,
    "S1": 3987,
    "T2": 4055,
    "S2": 4234,
}

_FILTER_TRANSMITTANCIES = {}
set_codex_transmittancies()
