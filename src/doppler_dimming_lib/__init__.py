"""
AAAAAAAAAAAAAAAAAAAAAA
"""

import logging

from .codex import (
    FILTER_CENTERS,
    convolve_codex_filter,
    get_R_as_spline,
    get_T_from_R,
    get_T_W_from_codex,
    set_codex_transmittancies,
    simulate_codex_filter,
    simulate_codex_images,
    simulate_codex_measure,
)
from .integrals import I_dl_domega_dphi, I_s_lambda, integrated_I_s
from .inversion import invert_temperature_parallel
from .metis import datacube_from_map, density_dc_from_ddt, get_pix_to_rsun
from .spectra.spectra import get_spectrum_from_txt, set_spectrum_level
from .utils import N_e_analytical, T_e_analytical, get_sun_center_from_map, timeit

# import doppler_dimming_lib.spectra.spectra


logging.basicConfig(
    level=logging.INFO, format="%(levelname)s - %(asctime)s - %(message)s"
)


del logging
