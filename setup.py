import setuptools

ext_modules = [
    setuptools.Extension(
        "integrands_lib.so",
        sources=[
            "src/doppler_dimming_lib/c_libraries/integrands_lib.c",
        ],
    )
]

setuptools.setup(
    ext_modules=ext_modules,
)
