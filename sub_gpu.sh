#!/bin/bash -l 
#$ -l h_rt=72:00:00 
#$ -j y 
#$ -N run4
#$ -pe omp 2
#$ -l gpus=1
#$ -l gpu_c=7.0

module purge 
module load python3/3.7.9 
module load cmake/3.19.4 
module load openmpi/3.1.4_gnu-8.1 
module load scalapack/2.0.2_gcc-8.1_openmpi-3.1.4 
module load openblas/0.3.7 
module load gcc/8.1.0 
source /projectnb/cui-buchem/tanmoy/packages/gromacs-dftbplus-ml-rl/install/bin/GMXRC 
export PLUMED_USE_LEPTON=yes 

module load intel/2019
module load openmpi/3.1.4_intel-2019
module load gcc/5.5.0
module load cuda/10.0
module load python3/3.6.5
module load tensorflow/1.13.1
export PATH=/projectnb/cui-buchem/tanmoy/projects/RL/md_tools-master/install/bin:$PATH

module load cuda/10.0
module load tensorflow/1.13.1
module load gcc

python3 run_mp.py param.json &>> out