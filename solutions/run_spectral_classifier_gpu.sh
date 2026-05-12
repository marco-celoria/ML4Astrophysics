#!/bin/bash
        
#SBATCH --job-name=spectral_classifier_gpu
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --error  job_spectral_classifier_gpu_optuna-%j.err
#SBATCH --output job_spectral_classifier_gpu_optuna-%j.out
# Enable cuda and load the python module and enable the venv.
module load python/3.11.7
module load cuda/12.2 
module load gcc/12.2.0  

source /leonardo_work/cin_staff/mcelori1/ML4Astrophysics/pyvenv_ml_gpu/bin/activate

srun python spectral_classifier_gpu_optuna.py 
