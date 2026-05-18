# import matplotlib.pyplot as plt
# import micropolarray as ml
import multiprocessing as mp
import sys

# doppler_dimming_lib.set_spectrum_level(0)
import numpy as np
from numpy import linalg

# sys.path.append("./src/")

import doppler_dimming_lib as db


@db.utils.timeit
def working_test(rho, _lambda, T_e, W):
    """Simple test to see if I_s_lambda, I_dl_domega_dphi and convolve_codex_filter are working"""

    print("Working test begin")
    I = db.I_s_lambda(rho, _lambda, T_e, W)
    print(f"{I = }")

    I = db.I_dl_domega_dphi(rho, _lambda, T_e, W)
    print(f"{I = }")

    def J_s_lambda(_lambda):
        return db.integrals.I_s_lambda(rho, _lambda, T_e, W, verbose=False)

    J_s_4233 = db.convolve_codex_filter(db.FILTER_CENTERS["S2"], J_s_lambda)
    J_s_4055 = db.convolve_codex_filter(db.FILTER_CENTERS["T2"], J_s_lambda)

    print("Working test ended successfully")


"""
@db.utils.timeit
def test_inversion():

    codex_images = db.simulate_codex_images(
        image_side, wind_pos, metis_ne_filename, T_e_pos
    )

    inferred_Ts = db.inversion.invert_temperature_batch(
        codex_images,
        r_pos,
        wind_speed,
        densities_dc,
        r_dc,
        pid=0,
    )


@db.utils.timeit
def test_parallelized_inversion():
    tot_nprocs = 13

    def test_func_to_parallelize(): ...

    args = []

    print(f"Starting parallel calculation, {tot_nprocs=}")
    with mp.Pool(processes=tot_nprocs) as p:
        result = p.starmap(
            test_func_to_parallelize,
            args,
        )

    merged_result = merge_2d_batched_array(result)
    assert np.array_equal(sample_array, merged_result)

    # info(f"Parallel computation ended in: {end - start} s")
    print(f"Parallel computation is correct")
"""

if __name__ == "__main__":

    working_test(rho=2, _lambda=3700, T_e=1.0e6, W=250)
    # test_parallelized_inversion()
