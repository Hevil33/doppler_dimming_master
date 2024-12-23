#!/bin/bash

make clean
sphinx-apidoc -f -o ./source ../src/doppler_dimming_lib 
sphinx-build -M html ./source ./build
sphinx-build -M latexpdf ./source ./build
