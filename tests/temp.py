import time

import doppler_dimming_lib as db


def main():
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
