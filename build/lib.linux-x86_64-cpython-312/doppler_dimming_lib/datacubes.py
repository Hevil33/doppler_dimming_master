import sys

import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
from astropy import units as u
from astropy.io import fits

from doppler_dimming_lib.metis import Ne_exponential_fit, geom_factor_function
from doppler_dimming_lib.utils import get_sun_center_from_map


def datacube_from_map(_map, **kwargs):

    side_pix = _map.data.shape[0]
    dc = np.zeros(shape=(side_pix, side_pix, side_pix), dtype=float)

    x_center, y_center = get_sun_center_from_map(_map)

    ...


def map_from_ddt(filename: str, hdu_ID: int = 0, resample_to: list = None):
    with fits.open(filename) as file:
        header = file[0].header
        data = file[hdu_ID].data
    _map = sunpy.map.Map(header, data)

    if resample_to is not None:
        _map = _map.resample(resample_to * u.pix)

    return _map
