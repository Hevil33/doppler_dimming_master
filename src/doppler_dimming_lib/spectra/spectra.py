__all__ = ["get_spectrum_from_txt", "set_spectrum_level"]

import ctypes
import functools
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u


class spectrum_config:
    SPECTRUM_LEVEL = None
    COMPLETE_SPECTRUM = None


@functools.lru_cache
def get_user_data(spectrum_level: int) -> list:
    """Wrapper for retrieving data from a photospheric spectrum. Wavelenghts and irradiances are returned as contiguous arrays to allow passing to c library.

    Args:
        spectrum_level (int): spectrum to load, see get_spectrum_from_txt.

    Returns:
        list: array of wavelenghts and array of irradiances.
    """
    spectrum = get_spectrum_from_txt(spectrum_level)
    wls = np.ascontiguousarray(np.array(spectrum[0], dtype=np.double))
    Fls = np.ascontiguousarray(np.array(spectrum[1], dtype=np.double))

    return wls, Fls


def set_spectrum_level(spectrum_level: int = 0) -> ctypes.c_void_p:
    """Sets the spectrum to be used by this library.

    Args:
        spectrum_level (int, optional): spectrum to load, from sparser (0) to denser (2), see get_spectrum_for_txt. Defaults to 0.

    Returns:
        ctypes.c_void_p: pointer to spectra data, to be passed to c library
    """
    spectrum_config.SPECTRUM_LEVEL = spectrum_level
    spectrum_config.COMPLETE_SPECTRUM = get_user_data(spectrum_level)


def get_spectrum_from_txt(precision_level: int = 0) -> np.ndarray:
    """Returns the spectrum from txt file. Use the precision level
    to determine the sampling quality.

    Args:
        precision_level (int, optional): : 0 to 2 increasing spectrum sampling, sparse (0, Thekaekara), medium (1, Thuillier) or dense (2, MODTRAN). Defaults to 0 for faster computations.

    Returns:
        np.ndarray: spectrum wavelenghts in [cm] and irradiances in [erg / s cm2 sr A].
    """

    print(f"Getting spectrum data from file (precision level {precision_level})...")

    # models_folder = "../data/photospheric_spectra/"
    # models_folder = os.path.join(
    #    os.path.dirname(__file__), "../data/photospheric_spectra/"
    # )

    models_folder = os.path.join(
        os.path.dirname(__file__), "..", "data", "photospheric_spectra"
    )
    models_folder = os.path.abspath(models_folder)
    models_folder = os.path.join(models_folder, "")  # adds trailing slash

    if precision_level == 0:  # Thekaekara spectrum
        dataset = pd.read_table(
            models_folder + "thekaekara.txt",
            usecols=[0, 1],
            skiprows=2,
            sep="\t| ",
            engine="python",
        )
        wl_unit = u.nm
        I_unit = u.W / (u.m * u.m * u.nm)
    elif precision_level == 1:  # Thuillier spectrum
        dataset = pd.read_table(
            models_folder + "second_spectrum.txt",
            skiprows=15,
            usecols=[0, 1],
            sep="\t| ",
            engine="python",
        )
        wl_unit = u.nm
        I_unit = 1.0e-6 * u.W / (u.cm * u.cm * u.nm)
    elif precision_level == 2:  # nrel MODTRAN spectrum
        dataset = pd.read_csv(
            models_folder + "AllMODEtr.txt",
            sep="\t| ",
            usecols=[1, 4],
            skiprows=2,
            engine="python",
        )
        wl_unit = u.nm
        I_unit = u.W / (u.m * u.m * u.nm)
    erg_s_cm2_A_sr = u.erg / (u.s * u.cm * u.cm * u.Angstrom * u.sr)
    # sun_solid_angle = 6.794e-5  # steradians
    earth_distance_SR = const.au / const.R_sun  # 215.03
    # earth_distance_SR = 211.65  # R_sun
    all_ws, all_Is = [], []
    for wavelenght, I in zip(
        dataset[dataset.columns[0]].values,
        dataset[dataset.columns[1]].values,
    ):
        wl = wavelenght * wl_unit.to(u.cm)
        # I = I * I_unit / (sun_solid_angle * u.sr) # mine
        I = (
            I * (earth_distance_SR * earth_distance_SR / np.pi) * (I_unit / u.sr)
        )  # Cram 1976, cap 2.B
        I_erg = I.to(erg_s_cm2_A_sr)
        all_ws.append(wl)
        all_Is.append(I_erg.value)
    complete_spectrum = [all_ws, all_Is]

    print("Done!")
    return np.array(complete_spectrum)
