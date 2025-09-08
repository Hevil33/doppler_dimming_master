#!/bin/bash

# Run this file to compile the shared C libraries that are used within scipy to speed up computations. With linux you can run 
#  
# $ ./compile_c.sh 
#  
# or you can copy the line 
# 
# gcc -lm -shared -fPIC -o integrands_lib.so integrands_lib.c
# 
# to compile it with gcc or other compilers. Make sure that the linked library is shared and has the same name "integrands_lib.so".

gcc -lm -shared -fPIC -o integrands_lib.so integrands_lib.c