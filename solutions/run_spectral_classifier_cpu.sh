#!/bin/bash
        
#SBATCH --job-name=spectral_classifier_cpu
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=dcgp_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --error  job_spectral_classifier_cpu-%j.err
#SBATCH --output job_spectral_classifier_cpu-%j.out
# Enable cuda and load the python module and enable the venv.
module load python/3.11.7
module load gcc/12.2.0  

source /leonardo_work/cin_staff/mcelori1/ML4Astrophysics/pyvenv_ml_cpu/bin/activate

srun python spectral_classifier_cpu.py 
