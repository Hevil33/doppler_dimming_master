# doppler_dimming_lib

This is a simple example package. You can use [GitHub-flavored Markdown](https://guides.github.com/features/mastering-markdown/) to write your content.

# Installation 

Copy the repository locally 

`git clone https://github.com/Hevil33/doppler_dimming_master`

then cd in the downloaded folder install the library with pip

`pip install .`


lastly, to compile the auxiliary c libraries, cd into the site-packages folder and run the `compile.sh`. It will use gcc as default, if you don't have it you can use any other compiler as long as it produces the right `integrands_lib.so` in the same folder. The syntax is

`gcc -lm -shared -fPIC -o integrands_lib.so integrands_lib.c`