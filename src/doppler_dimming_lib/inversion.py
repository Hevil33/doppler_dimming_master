"""
Temperature inversion for images using parallelization for speedup
"""

import multiprocessing as mp
import sys
import time
from datetime import timedelta
from logging import info

import numpy as np
from tqdm import tqdm

from . import codex, parallelization


def invert_temperature_batch(
    codex_images: np.ndarray,
    wind_speed_pos: np.ndarray | float,
    densities_dc: np.ndarray,
    helio_distances_dc: np.ndarray,
    pid: list,
) -> np.ndarray:
    """Get the temperature by inverting CODEX images ratio.

    Args:
        codex_images (np.ndarray): images from CODEX measurements. Should be a [4, m, n] array containing 4 filters measurements [T1, S1, T2, S2] (in order) of shape [m, n].
        wind_speed_pos (np.ndarray): [m, n] array or float containing the wind speed intensity in the POS for each pixel of the CODEX measurements. If a float is provided, homogeneous wind speed is assumed.
        densities_dc (np.ndarray): datacube of shape [l, m, n] containing electron densities in cm^-3. The first index contains values along the LOS for each pixel, and POS is assumed to be at int(l/2).
        helio_distances_dc (np.ndarray): datacube of shape [l, m, n] containing heliocentric distances of each density datacube entry in solar radii. The first index contains values along the LOS for each pixel, and POS is assumed to be at int(l/2).
        pid (list): id used for printing progress.

    Returns:
        np.ndarray: temperatures resulting from inversion in the POS, in Kelvin.
    """

    r_pos = helio_distances_dc[int(helio_distances_dc.shape[0] / 2), :, :]
    inferred_Ts = np.zeros_like(r_pos)

    if not isinstance(wind_speed_pos, np.ndarray):
        wind_speed_pos = np.ones_like(r_pos) * wind_speed_pos

    for y in tqdm(
        range(r_pos.shape[0]),
        desc=f"Processor {pid}",
        position=pid,
        leave=False,
        # file=sys.stdout,
    ):
        for x in range(r_pos.shape[1]):
            Js = codex_images[:, y, x]
            rho = r_pos[y, x]

            if (Js[3] == 0) or (Js[2] == 0) or (rho <= 1.0):
                inferred_Ts[y, x] = 0
            else:
                ratio = Js[3] / Js[2]
                densities_los = densities_dc[:, y, x]
                r_los = helio_distances_dc[:, y, x]
                wind_speed = wind_speed_pos[y, x]
                # N_e_function = interp1d(
                #    r_los, densities_los, fill_value="extrapolate"
                # )
                # Nes = [N_e_function(r) for r in r_los]

                N_e_function = np.array([r_los, densities_los], dtype=np.double)

                inferred_Ts[y, x] = codex.get_T_from_R(
                    ratio, rho, wind_speed=wind_speed, N_e_function=N_e_function
                )
        # pbar.update(1)

    return inferred_Ts


def invert_temperature_parallel(
    codex_images: np.ndarray,
    wind_speed_pos: np.ndarray,
    densities_dc: np.ndarray,
    coordinates_dc: np.ndarray,
    total_processors: int = None,
) -> np.ndarray:
    """Get the temperature by inverting CODEX images ratio. This version first splits the domain for X processors and then applies invert_temperature_batch to each subdomain, before merging the result.

    Args:
        codex_images (np.ndarray): images from CODEX measurements. Should be a [4, m, n] array containing 4 filters measurements [T1, S1, T2, S2] (in order) of shape [m, n].
        wind_speed_pos (np.ndarray): [m, n] array or float containing the wind speed intensity in the POS for each pixel of the CODEX measurements. If a float is provided, homogeneous wind speed is assumed.
        densities_dc (np.ndarray): datacube of shape [l, m, n] containing electron densities in cm^-3. The first index contains values along the LOS for each pixel, and POS is assumed to be at int(l/2).
        coordinates_dc (np.ndarray): datacube of shape [l, m, n] containing coordinates of each density datacube entry in solar radii. The first index contains values along the LOS for each pixel, and POS is assumed to be at int(l/2).
        total_processors (int, optional): number of processors for parallelization. This number should be tweaked to avoid asymmetric division of the domain which can increase the computation time. If set to None, 4 processors are used. Defaults to None.

    Returns:
        np.ndarray: temperatures resulting from inversion in the POS, in Kelvin.
    """

    info("Inverting dummy parallel")
    if total_processors is None:
        total_processors = 4

    helio_distances_dc = np.sqrt(
        coordinates_dc[0] * coordinates_dc[0]
        + coordinates_dc[1] * coordinates_dc[1]
        + coordinates_dc[2] * coordinates_dc[2]
    )

    splitted_codex_images = [
        parallelization.batch_2d_array(codex_images[i], total_processors)
        for i in range(4)
    ]

    if not isinstance(wind_speed_pos, np.ndarray):
        wind_speed_pos = np.ones(shape=helio_distances_dc.shape[1:]) * wind_speed_pos

    splitted_wind_speed = parallelization.batch_2d_array(
        wind_speed_pos, total_processors
    )
    splitted_densities_dc = parallelization.batch_3d_array(
        densities_dc, total_processors
    )
    splitted_r_dc = parallelization.batch_3d_array(helio_distances_dc, total_processors)

    processor_ids = [i for i in range(total_processors)]

    args = (
        [
            np.array(
                [
                    splitted_codex_images[0][i],
                    splitted_codex_images[1][i],
                    splitted_codex_images[2][i],
                    splitted_codex_images[3][i],
                ]
            ),
            splitted_wind_speed[i],
            splitted_densities_dc[i],
            splitted_r_dc[i],
            processor_ids[i],
        ]
        for i in range(total_processors)
    )

    info("Starting parallel temperature inversion")
    start = time.perf_counter()
    with mp.Pool(processes=total_processors) as p:
        result = p.starmap(
            invert_temperature_batch,
            args,
            # chunksize=2,
        )

    end = time.perf_counter()
    # sys.stdout.flush()
    # info(f"Parallel inversion ended in {end - start} s")
    info(f"Parallel temperature inversion ended in {timedelta(seconds=end - start)}")

    inferred_Ts = parallelization.merge_2d_batched_array(result)
    return inferred_Ts

    inferred_Ts = np.zeros_like(r_pos)
    for i in range(processors_y):
        for j in range(processors_x):
            inferred_Ts[
                i * (chunk_size_y) : (i + 1) * chunk_size_y,
                j * (chunk_size_x) : (j + 1) * chunk_size_x,
            ] = result[i + processors_y * j].reshape(chunk_size_y, chunk_size_x)

    return inferred_Ts
