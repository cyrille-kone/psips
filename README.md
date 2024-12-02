### Install dependencies and compile 

To compile the code a compiler compatible with `C++17` should be installed. We recommend using `GCC10+`. 
`Cmake` should also be installed. The other in requirements.txt should be installed with `conda` or `mamba`. 

To compile the code the following instructions should be executed in the terminal from the project directory

`` mkdir build && cd build ``
and from the ``build`` directory run 

``cmake ..`` and ` make ` to compile. 

### Run and reproduce 

To run an experiment after compilation run the command `./code` form the build dir 
Additional details for each experiment are commented in the header file `src/xp.hpp` and in `src/main.cpp`.


