import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import sunpy
from astropy.io import fits
from tqdm import tqdm

import doppler_dimming_lib as db


def main():
    metis_ne_filename = "/home/herve/DDT/ddtdata/input/wl/ne/for_ddt_solo_L2_metis-vl-pb_20210114T003001_V01_ne.fits"
    image_side = 8
    tot_processors = 8

    densities_dc, coordinates_dc = db.density_dc_from_ddt(metis_ne_filename, image_side)

    densities_dc = np.ones_like(densities_dc) * np.nanmean(densities_dc)

    wind_speed = np.ones(shape=(image_side, image_side))

    T_e_pos = db.metis.Te_pos_from_ne(metis_ne_filename, image_side)
    dummy_temps = np.linspace(1.0e6, 2.0e6, image_side)
    for i in range(T_e_pos.shape[0]):
        T_e_pos[i, :] = dummy_temps[i]

    codex_images = db.simulate_codex_images(
        image_side,
        wind_speed,
        densities_dc,
        coordinates_dc,
        T_e_pos,
        tot_processors=tot_processors,
    )

    codex_images_log = np.log10(codex_images)
    fig, ax = plt.subplots(2, 2, dpi=200)
    ax = ax.ravel()
    for i in range(4):
        ax[i].imshow(
            codex_images_log[i],
            # vmin=np.median(codex_images_log[i]) - np.std(codex_images_log[i]),
            # vmax=np.median(codex_images_log[i]) + np.std(codex_images_log[i]),
            origin="lower",
        )
        ax[i].set_title(f"{i}")

    # test inversion

    # serial version
    if False:
        start = time.perf_counter()
        r_dc = np.sqrt(
            coordinates_dc[0] * coordinates_dc[0]
            + coordinates_dc[1] * coordinates_dc[1]
            + coordinates_dc[2] * coordinates_dc[2]
        )
        inferred_Te_serial = db.inversion.invert_temperature_batch(
            codex_images, wind_speed, densities_dc, r_dc, 0
        )
        end = time.perf_counter()
        print()
        print(f"ended serial in {timedelta(seconds=end-start)}, starting parallel")
        print()

    inferred_Te_pos = db.invert_temperature_parallel(
        codex_images,
        wind_speed,
        densities_dc,
        coordinates_dc,
        total_processors=tot_processors,
    )

    fig, ax = plt.subplots(1, 2)
    ax = ax.ravel()
    ax[0].imshow(T_e_pos, origin="lower")
    ax[0].set_title("input Te")
    ax[1].imshow(inferred_Te_pos, origin="lower")
    ax[1].set_title("inferred Te")
    plt.show()

    start = time.perf_counter()

    rho = 2
    _lambda = 3700
    T_e = 1.0e6
    W = 250

    for i in range(10):
        I = db.I_s_lambda(rho, _lambda, T_e, W, verbose=False)

        I = db.integrated_I_s(rho, _lambda, T_e, W, verbose=False, component=3)
        print(f"{I = }")

        I = db.I_dl_domega_dphi(rho, _lambda, T_e, W, verbose=False)

        def J_s_lambda(_lambda):
            return db.integrals.I_s_lambda(rho, _lambda, T_e, W, verbose=False)

        J_s_4233 = db.convolve_codex_filter(4234, J_s_lambda, verbose=False)
        J_s_4055 = db.convolve_codex_filter(4055, J_s_lambda, verbose=False)

    end = time.perf_counter()

    print(f"Elapsed: {end-start} s")


if __name__ == "__main__":
    main()
