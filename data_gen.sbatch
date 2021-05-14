#!/bin/bash
module purge; 
module load Python/Anaconda_v10.2019

source deactivate
source activate my_py_env

python data_gen_np.py data 3000 300000 32 42 100