import sys
import time

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy import integrate
from scipy.interpolate import CubicSpline, interp1d


def get_pix_to_rsun(input_map):
    return input_map.meta["CDELT1"] / input_map.meta["RSUN_ARC"]


def timeit(func):
    """Wrapper for timing functions' execution times. Prints the difference between starting and ending time.

    Args:
        func (callable): function of which to measure the execution time
    """

    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


def N_e_analytical(r: float) -> float:
    """Returns the electron density as a radial function from sun center.

    Args:
        r (float): radial distance in R_sun

    Returns:
        float: electron density in cm-3
    """

    # return 1.67 * 10 ** (4 + 4.04 / r) #PROBLEM: this model seems to be very different from the others!!! (orders of magnitude)

    return 1.0e8 * (
        0.036 * r ** (-1.5) + 1.55 * r ** (-6.0) + 2.99 * r ** (-16.0)
    )  # Baumbach 1937, cm-3

    p0 = 20
    p1 = -0.02
    p2 = 6.05
    phi = 0
    A = p0 * np.exp(-0.5 * ((phi - p1) / p2) ** 2)
    return A * 1.0e8 * (1.55 * r**-6 + 2.99 * r**-16)  #  allen 1947

    a = [8.6e-4, 4.5915, -0.95714, -3.4846, 5.663, 2.4406]  # poles
    a = [2.6e-3, 5.5986, 0.82902, -5.6654, 3.9784, 5.4155]  # current sheet

    z = 1.0 / r
    z2 = z * z
    return 1.0e8 * (
        a[0]
        * np.exp(a[1] * z + a[5] * z2)
        * z2
        * (1.0 + a[2] * z + a[3] * z2 + a[4] * z2 * z)
    )  # Guhathakurta 2006, cm-3


def T_e_analytical(rho: float, N_e_function=N_e_analytical, **kwargs) -> float:
    """Returns an analytical electron temperature as a function of doppler_dimming_lim.utils.N_e_analytical from Lemaire & Stegen 2016

    Args:
        rho (float): heliocentric distance in solar radii
        N_e_function (callable): density as a function of heliocentric distance (in sr). Use a polynomial to interpolate data if necessary

    Returns:
        float: electron temperature profile
    """
    # from Lemaire Stegen 2016
    if False:  # plot
        fig, ax = plt.subplots(dpi=200)
        r = np.linspace(1, 10, 10)
        ax.plot(r, [doppler_dimming_lib.T_e_analytical(ri) for ri in r])
        plt.show()

    def N_e_physical_boundary(*args, **kwargs):
        N_e = N_e_function(*args, **kwargs)
        if N_e > 0:
            return N_e
        else:
            return 0

    almost_infinity = 1.0e4  # approximate big radius

    if rho <= 1:  # underfined if less than a solar radius
        return 0

    N_e_rho = N_e_physical_boundary(rho, **kwargs)

    return (
        (
            0.5
            * const.G.value
            * const.M_sun.value
            * const.m_p.value
            / (const.k_B.value * const.R_sun.value)
        )
        * integrate.quad(
            lambda r: N_e_physical_boundary(r) / (r * r), rho, almost_infinity, points=0
        )[0]
        / N_e_rho
    )


def get_sun_center_from_map(map):
    """Returns the coordinates of the pixel containing the sun center. Use this instead of CRPIX!!!!

    Args:
        map (sunpy.map.Map): map containing the image and reference frame

    Returns:
        list: coordinate x and y
    """
    coord = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=map.coordinate_frame)

    coord_pixel = map.wcs.world_to_pixel(coord)
    return coord_pixel
