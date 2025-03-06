import sys

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
import tqdm
from astropy.coordinates import SkyCoord
from astropy.io import fits
from scipy.io import readsav

from doppler_dimming_lib.utils import (
    T_e_analytical,
    get_pix_to_rsun,
    get_sun_center_from_map,
)


def read_sav_model(filename: str) -> list:
    """Reads a sav idl file and returns the parameters list.

    Args:
        filename (str): path to idl save file (.sav)

    Returns:
        list: list of 3d array containing the red parameters. In order: positions array, sun radial distances, sun latitude angle, densities, electron temperatures, proton temperatures, solar wind speed.
    """
    print(f"Retrieving data from {filename} model...")
    sav_dict = readsav(filename)
    vers, xout, rdist, theta, n_e, t_e, t_p, wind = sav_dict.values()
    return xout, rdist, theta, n_e, t_e, t_p, wind


def Ne_exponential_fit(x: float, b: np.ndarray) -> float:
    """Fitted ne based on metis pB.

    Args:
        x (float): heliocentric radial distance in Rsun
        b (np.ndarray): coefficients array

    Returns:
        float: electron density at a given heliocentric radial distance
    """
    # inn_fov = 1.6
    # rsun_arc = 2938.775

    return 1.0e8 * (
        b[0] * x ** b[1] + b[2] * x ** b[3] + b[4] * x ** b[5] + b[6] * x ** b[7]
    )


def geom_factor_function(r: float) -> float:
    """Geometric factor function from Van de Hulst.

    Args:
        r (float): heliocentric distance in solar radii

    Returns:
        float: computed geometric factor
    """
    u = 0.63

    # function undefined if r < 1 (under solar surface)
    if r < 1:
        return 0

    omega = np.arcsin(1.0 / r)

    dueapb = (1.0 - u) / (1.0 - 1.0 / 3.0 * u) * (2.0 * (1.0 - np.cos(omega))) + u / (
        1.0 - 1.0 / 3.0 * u
    ) * (
        1.0
        - np.cos(omega) ** 2
        / np.sin(omega)
        * np.log((1.0 + np.sin(omega)) / np.cos(omega))
    )

    dueamb = (1.0 - u) / (1.0 - 1.0 / 3.0 * u) * (
        2.0 / 3.0 * (1.0 - np.cos(omega) ** 3)
    ) + u / (1.0 - 1.0 / 3.0 * u) * (
        1.0 / 4.0
        + np.sin(omega) ** 2 / 4.0
        - np.cos(omega) ** 4
        / (4.0 * np.sin(omega))
        * np.log((1.0 + np.sin(omega)) / np.cos(omega))
    )

    ag = (dueapb + dueamb) / 4.0
    bg = (dueapb - dueamb) / 2.0
    geom = ag - bg

    geom = 1 / geom

    return geom


def get_pix_to_rsun(input_map: sunpy.map.Map) -> float:
    """Returns the platescale in pixel/Rsun

    Args:
        input_map (sunpy.map.Map): map for which to calculate the ratio

    Returns:
        float: pixel/Rsun platescale
    """
    return input_map.meta["CDELT1"] / input_map.meta["RSUN_ARC"]


def get_3d_param(
    param: str,
    pix_to_rsun: float,
    coefficients: np.ndarray,
    pixel_pos: list,
    sun_center_pos: list,
    z_pix: float,
) -> float:
    """Returns the electron density or temperature from a Metis map fitted with DDT. For the electron temperature calculation see utls.T_e_analytical.

    Args:
        param (str): parameter to calculate, either "Ne" or "Te"
        map_los (sunpy.map.Map): map contaning the parameter and metadata in sunpy format
        coefficients (np.ndarray): [8, 360] array containing the fitted coefficients in the first axis.
        pixel_pos (list): [y, x] pixel coordinates at which to calculate the parameter.
        sun_center_pos (list): [y, x] pixel coordinates of the sun center. Provided by the user to avoid useless calculations.
        x_los (float): distance in pixels along the line of sight at which to calculate the parameter.

    Returns:
        float: electron density at set location
    """

    y_center, x_center = sun_center_pos
    pixel_y, pixel_x = pixel_pos
    x_pix = pixel_x - x_center
    y_pix = pixel_y - y_center
    # x_pix += 0.5
    # y_pix += 0.5
    # z_pix += 0.5

    rho_pix = np.sqrt((x_pix * x_pix) + (y_pix * y_pix) + (z_pix * z_pix))
    rho_rsun = rho_pix * pix_to_rsun

    project_spheric = True
    if project_spheric:
        # following two formulas are the same
        # polar_angle_i = int(
        #    np.rad2deg(
        #        np.arccos(np.sign(x_pix) * np.sqrt(x_pix * x_pix + z_pix * z_pix) / rho_pix)
        #    )
        # )
        polar_angle_i = int(
            np.rad2deg(
                np.arctan2(
                    y_pix,
                    np.sign(x_pix) * np.sqrt(x_pix * x_pix + z_pix * z_pix),
                )
            )
        )
    else:
        # use simplest one for stability
        polar_angle_i = int(np.rad2deg(np.arctan2(y_pix, x_pix)))

    if False:  # check if scale is ok
        fig, ax = plt.subplots()
        ax.imshow(map_los.data, origin="lower")
        occulter_rsun = map_los.meta["INN_FOV"] * 3600 / map_los.meta["RSUN_ARC"]
        pix_to_rsun = map_los.meta["CDELT1"] / map_los.meta["RSUN_ARC"]
        occulter_pix = int(occulter_rsun / pix_to_rsun)
        ax.add_artist(plt.Circle([x_center, y_center], occulter_pix))
        plt.show()

    geom = geom_factor_function(rho_rsun)

    if param == "Ne":
        # rho_rsun = np.sqrt(x_pix * x_pix + z_pix * z_pix)
        N_e = Ne_exponential_fit(rho_rsun, b=coefficients[:, polar_angle_i]) * geom

        return N_e if N_e > 0 else 0

    elif param == "Te":

        def Ne(r, *args, **kwargs):
            return Ne_exponential_fit(r, b=coefficients[:, polar_angle_i]) * geom

        T_e = T_e_analytical(rho_rsun, Ne, b=coefficients[:, polar_angle_i])

        return T_e if T_e > 0 else 0

    elif param == "wind":

        ...


"""
def get_geom_factor(ne_map, ne_polar_data, fitted_polar):
    occulter_rsun = ne_map.meta["INN_FOV"] * 3600 / ne_map.meta["RSUN_ARC"]
    pix_to_rsun = get_pix_to_rsun(ne_map)
    occulter_pix = int(occulter_rsun / pix_to_rsun)

    ne_polar_padded = np.pad(ne_polar_data, ((occulter_pix, 0), (0, 0)), "constant")
    fitted_polar_padded = np.pad(fitted_polar, ((occulter_pix, 0), (0, 0)), "constant")

    
    geom_factor_padded = np.zeros_like(fitted_polar_padded)
    np.divide(
        ne_polar_padded,
        fitted_polar_padded,
        where=fitted_polar_padded != 0,
        out=geom_factor_padded,
    )
    

    rmax_pix = (
        int(ne_polar_padded.shape[0] * np.sqrt(3)) + 1
    )  # should be greater than the cube half diagonal
    rhos_rsun = np.arange(0, rmax_pix, 1) * pix_to_rsun
    geom_factor_radial = np.array(
        [geom_factor_function(r) if r >= 1 else 0 for r in rhos_rsun]
    )
    geom_factor_calculated = np.repeat(geom_factor_radial[:, np.newaxis], 360, axis=1)
    # geom_factor_calculated[0:occulter_pix, :] = 0

    if False:  # check correct padding
        fig, ax = plt.subplots()
        mappable = ax.imshow(geom_factor_calculated, origin="lower")
        fig.colorbar(mappable)
        ax.set_title("Geometric factor")

        fig, ax = plt.subplots()
        mappable = ax.imshow(fitted_polar_padded, origin="lower", vmin=0, vmax=7.0e6)
        fig.colorbar(mappable)
        ax.set_title("Fitted")
        fig, ax = plt.subplots()
        mappable = ax.imshow(ne_polar_padded, origin="lower", vmin=0, vmax=7.0e6)
        fig.colorbar(mappable)
        ax.set_title("Map from metis")
        fig, ax = plt.subplots()
        ne_data = ne_map.data
        mappable = ax.imshow(ne_data, origin="lower", vmin=0, vmax=7.0e6)
        fig.colorbar(mappable)
        ax.set_title("Metis densities")

        plt.show()

    return geom_factor_calculated, fitted_polar_padded
    return geom_factor_padded, fitted_polar_padded
"""


def resample_data(new_dimension, ne_header, ne_data, ne_polar_data):
    _map = sunpy.map.Map(ne_data, ne_header)
    resampled = _map.resample([new_dimension, new_dimension] * u.pixel)

    resampled_header = resampled.meta

    polar_map = sunpy.map.Map(ne_polar_data, ne_header)
    new_polar_map_dimension = int(
        ne_polar_data.shape[0] * new_dimension / ne_data.shape[0]
    )
    resampled_polar = polar_map.resample([360, new_polar_map_dimension] * u.pixel)

    return resampled_header, resampled.data, resampled_polar.data


def datacube_from_map(ne_map, coefficients):
    side_pix = ne_map.data.shape[0]

    dc = np.zeros(shape=(side_pix, side_pix, side_pix), dtype=float)

    pix_to_rsun = get_pix_to_rsun(ne_map)

    # datacube center coordinate (xy defined by map, z half by default)
    x_center, y_center = get_sun_center_from_map(ne_map)
    z_center = side_pix / 2

    for z_pix in tqdm.tqdm(range(side_pix), desc="Filling density datacube"):
        for y_pix in range(side_pix):
            for x_pix in range(side_pix):
                # input()
                # print(z_pix, y_pix, x_pix)
                # print(z_pix + zstart_pix, y_pix + ystart_pix, x_pix + xstart_pix)

                dc[z_pix, y_pix, x_pix] = get_3d_param(
                    "Ne",
                    pix_to_rsun,
                    coefficients,
                    pixel_pos=[y_pix + 0.5, x_pix + 0.5],
                    sun_center_pos=[y_center + 0.5, x_center + 0.5],
                    z_pix=z_pix - z_center + 0.5,
                )

    # coordinates in rsun/
    side_rsun = side_pix * pix_to_rsun
    zstart_rsun = (-z_center) * pix_to_rsun
    ystart_rsun = (-y_center) * pix_to_rsun
    xstart_rsun = (-x_center) * pix_to_rsun

    # order z, y, z is wrong, either change all code or make custom function
    # coordinates = np.meshgrid(
    #    np.linspace(ystart_rsun, ystart_rsun + side_rsun, side_pix),
    #    np.linspace(xstart_rsun, xstart_rsun + side_rsun, side_pix),
    #    np.linspace(zstart_rsun, zstart_rsun + side_rsun, side_pix),
    #    indexing="ij",
    # )
    # coordinates = [coordinates[2], coordinates[1], coordinates[0]]

    xs = np.linspace(xstart_rsun, xstart_rsun + side_rsun, side_pix)
    ys = np.linspace(ystart_rsun, ystart_rsun + side_rsun, side_pix)
    zs = np.linspace(zstart_rsun, zstart_rsun + side_rsun, side_pix)

    coordinates = np.empty(shape=(3, len(zs), len(ys), len(xs)))
    for zi, z_rsun in enumerate(zs):
        for yi, y_rsun in enumerate(ys):
            for xi, x_rsun in enumerate(xs):
                coordinates[0, zi, yi, xi] = z_rsun - 0.5 * pix_to_rsun
                coordinates[1, zi, yi, xi] = y_rsun - 0.5 * pix_to_rsun
                coordinates[2, zi, yi, xi] = x_rsun - 0.5 * pix_to_rsun

    # check that coordinates are in correct order z, y, x
    assert np.std(coordinates[0][:, 0, 0]) != 0
    assert np.std(coordinates[1][0, :, 0]) != 0
    assert np.std(coordinates[2][0, 0, :]) != 0

    assert np.std(coordinates[0][0, :, 0]) < 1.0e-7
    assert np.std(coordinates[0][0, 0, :]) < 1.0e-7
    assert np.std(coordinates[1][:, 0, 0]) < 1.0e-7
    assert np.std(coordinates[1][0, 0, :]) < 1.0e-7
    assert np.std(coordinates[2][:, 0, 0]) < 1.0e-7
    assert np.std(coordinates[2][0, :, 0]) < 1.0e-7

    r_dc = np.sqrt(
        coordinates[0] * coordinates[0]
        + coordinates[1] * coordinates[1]
        + coordinates[2] * coordinates[2]
    )
    occulter_rsun = ne_map.meta["INN_FOV"] * 3600 / ne_map.meta["RSUN_ARC"]
    dc[r_dc < 1.0 * occulter_rsun] = 0

    """
    x, y = 0, 0
    fig, ax = plt.subplots(dpi=200)
    ax.plot(coordinates[0][:, y, x], dc[:, y, x], "+-r", label="f(z)")
    ax.plot(
        r_dc[:, y, x] * np.sign(coordinates[0][:, y, x]),
        dc[:, y, x],
        "+-b",
        label="f(r)",
    )
    plt.legend()
    plt.show()

    """

    if False:  # debug: show the datacube
        vmin = 0
        vmax = 5.5e6

        # subset
        # dc = np.where(r_dc < 3, dc, 0)
        # r_dc = np.where(r_dc < 3, r_dc, 0)

        fig, ax = plt.subplots(2, 2, constrained_layout=True)
        ax = ax.ravel()
        # ax.imshow(dc[int(dc.shape[0] / 2), :, :] - ne_map.data)
        ax[0].imshow(dc[int(dc.shape[0] / 2), :, :], vmin=1.0e4, vmax=1.0e6)
        ax[0].set_title("reconstructed")
        ax[1].imshow(ne_map.data, vmin=1.0e4, vmax=1.0e6)
        ax[1].set_title("original")
        ax[2].imshow(ne_map.data - dc[int(dc.shape[0] / 2), :, :])
        ax[2].set_title("original - reconstructed")
        ax[3].imshow(ne_map.data / dc[int(dc.shape[0] / 2), :, :])
        ax[3].set_title("original / reconstructed")

        fig, ax = plt.subplots()
        half_side = int(side_pix / 2)
        ax.plot(
            r_dc[:, half_side, half_side], dc[:, half_side, half_side], label="z cut"
        )
        ax.plot(
            r_dc[half_side, :, half_side], dc[half_side, :, half_side], label="y cut"
        )
        ax.plot(
            r_dc[half_side, half_side, :], dc[half_side, half_side, :], label="x cut"
        )
        plt.legend()

        logdc = np.log10(dc)
        logdc = dc
        fig, ax = plt.subplots(2, 2, dpi=200, tight_layout=True)
        ax = ax.ravel()
        ax[0].imshow(
            logdc[:, :, int(logdc.shape[2] / 2)],
            vmin=vmin,
            vmax=vmax,
            origin="lower",
        )  # x
        ax[0].set_ylabel("z")
        ax[0].set_xlabel("y")
        ax[0].plot(y_center, z_center, "r+", label="sun center")

        ax[1].imshow(
            logdc[:, int(logdc.shape[1] / 2), :],
            vmin=vmin,
            vmax=vmax,
            origin="lower",
        )  # y
        ax[1].set_ylabel("z")
        ax[1].set_xlabel("x")
        ax[1].plot(x_center, z_center, "r+", label="sun center")

        ax[2].imshow(
            logdc[int(logdc.shape[0] / 2), :, :],
            vmin=vmin,
            vmax=vmax,
            origin="lower",
        )  # z
        ax[2].set_ylabel("y")
        ax[2].set_xlabel("x")
        ax[2].plot(x_center, y_center, "r+", label="sun center")

        mappable = ax[3].imshow(ne_map.data, origin="lower", vmin=vmin, vmax=vmax)
        ax[3].set_title("Original map")
        ax[3].plot(x_center, y_center, "r+", label="sun center")

        for i in range(4):
            fig.colorbar(mappable, ax=ax[i])

        plt.legend()
        plt.show()

    return dc, coordinates


def datacube_from_file(filename, size_in_pixel):

    with fits.open(filename) as file:
        ne_header = file[0].header
        ne_data = file[0].data
        ne_polar_data = file[1].data
        ne_coeffs_data = file[2].data

    ne_header, ne_data, ne_polar_data = resample_data(
        size_in_pixel, ne_header, ne_data, ne_polar_data
    )

    ne_map = sunpy.map.Map(ne_data, ne_header)

    if False:  # check difference between crpix and sun center from wcs transform
        fig, ax = plt.subplots(subplot_kw=dict(projection=ne_map))
        ne_map.plot(axes=ax)
        ne_map.draw_limb()

        coord = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=ne_map.coordinate_frame)

        pixels = ne_map.wcs.world_to_pixel(coord)
        ax.plot(pixels[0], pixels[1], "ro", label="center with function")
        ax.plot(ne_map.meta["CRPIX1"], ne_map.meta["CRPIX2"], "r+", label="crpix")
        ax.legend()

        plt.show()

    # not needed
    """
    occulter_rsun = ne_map.meta["INN_FOV"] * 3600 / ne_map.meta["RSUN_ARC"]
    pix_to_rsun = ne_map.meta["CDELT1"] / ne_map.meta["RSUN_ARC"]
    fitted_polar = np.zeros_like(ne_polar_data, dtype=float)
    for angle in range(ne_polar_data.shape[1]):
        for pixel in range(ne_polar_data.shape[0]):
            pixel = int(pixel)
            fitted_polar[pixel, angle] = Ne_exponential_fit(
                pixel * pix_to_rsun + occulter_rsun,
                ne_coeffs_data[:, angle],
            )"

    # geom_factor_padded, fitted_polar_padded = get_geom_factor(
    #    ne_map, ne_polar_data, fitted_polar
    # )
    """

    dc, coordinates = datacube_from_map(ne_map, ne_coeffs_data)

    return dc, coordinates
