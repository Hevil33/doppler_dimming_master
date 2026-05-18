# Installation 

Copy the repository locally 

`git clone https://github.com/Hevil33/doppler_dimming_master`

Go into the downloaded folder and install the library with pip

`cd doppler_dimming_master`
`pip install .`

Ensure to install sunpy via

`pip install sunpy[all]`

C files need to be compiled for this module to work, which should be done automatically while installing.
If not, go into the installation folder (usually pythonX.X/site-packages) and run the script

`./compile.sh`

This script contains the command to compile the libraries which is

`gcc -lm -shared -fPIC -o integrands_lib.so integrands_lib.c`

gcc is used by default. 
If you don't have gcc, any other C compiler will work as long as `integrands_lib.so` is correctly produced in the same folder. 

## Test 

Warnings may appear, to check if the library was installed correctly run the test inside the tests folder. From the downloaded repository use your python version X.X 

`pythonX.X test.py`

