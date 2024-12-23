# import matplotlib.pyplot as plt
# import micropolarray as ml


import sys

# doppler_dimming_lib.set_spectrum_level(0)
import numpy as np
from numpy import linalg

sys.path.append("./src/")
import doppler_dimming_lib as db


@db.utils.timeit
def test_simple_algorithm(rho, _lambda, T_e, W):
    """Simple test to see if I_s_lambda, I_dl_domega_dphi and convolve_codex_filter are working"""

    print("Working test begin")
    I = db.I_s_lambda(rho, _lambda, T_e, W)
    print(f"{I = }")

    I = db.I_dl_domega_dphi(rho, _lambda, T_e, W)
    print(f"{I = }")

    def J_s_lambda(_lambda):
        return db.integrals.I_s_lambda(rho, _lambda, T_e, W, verbose=False)

    J_s_4233 = db.convolve_codex_filter(4234, J_s_lambda)
    J_s_4055 = db.convolve_codex_filter(4055, J_s_lambda)

    print("Working test ended successfully")


if __name__ == "__main__":

    test_simple_algorithm(rho=2, _lambda=3700, T_e=1.0e6, W=250)
