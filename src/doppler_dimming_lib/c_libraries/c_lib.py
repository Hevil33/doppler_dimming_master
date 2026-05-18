import ctypes
import os


def get_c_library(path: str = None):
    """Sets up the c_library object which is used by the library.
    This object contains C integrand functions.

    Args:
        path (str, optional): path to shared compiled library. If None, __file__+./integrands_lib.so is used. Defaults to None.

    Returns:
        ctypes.CDLL: CDLL library containing integrand functions.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "./integrands_lib.so")

    c_library = ctypes.CDLL(os.path.abspath(path))
    c_library.N_e_from_function.argtypes = (ctypes.c_double,)
    c_library.N_e_from_function.restype = ctypes.c_double
    # N_e = LowLevelCallable(c_library.N_e_from_function)

    c_library.I_lambda_mu.restype = ctypes.c_double
    c_library.I_lambda_mu.argtypes = (
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    )

    c_library.q_lambda.argtypes = (ctypes.c_double,)
    c_library.q_lambda.restype = ctypes.c_double

    c_library.I_dlambda_domega.restype = ctypes.c_double  # define res type
    c_library.I_dlambda_domega.argtypes = (
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
    )  # define input types

    c_library.I_dlambda_domega_dx.restype = ctypes.c_double  # define res type
    c_library.I_dlambda_domega_dx.argtypes = (
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p,
    )  # define input types

    return c_library
