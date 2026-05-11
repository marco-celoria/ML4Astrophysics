#!/bin/bash
        
#SBATCH --job-name=jupyter_environ
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --error logs/jupyter-%j.err
#SBATCH --output logs/jupyter-%j.out
# Enable cuda and load the python module and enable the venv.
module load python/3.11.7
module load cuda/12.2 
module load gcc/12.2.0  

source /leonardo_work/cin_staff/mcelori1/ML4Astrophysics/pyvenv/bin/activate

# Get the worker list associated to this slurm job
worker_list=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))

# Set the first worker as the head node and get his ip
head_node=${worker_list[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Print ssh tunnel instruction
jupyter_port=$(($RANDOM%(64511-50000+1)+50000))
jupyter_token=${USER}_${jupyter_port}
echo ===================================================
echo [INFO]: To access the Jupyter server, remember to open a ssh tunnel with: 
echo ssh -L $jupyter_port:$head_node_ip:$jupyter_port ${USER}@login07-ext.leonardo.cineca.it -N
echo then you can connect to the jupyter server at http://127.0.0.1:$jupyter_port/lab?token=$jupyter_token
echo ===================================================

# Start the head node
echo [INFO]: Starting jupyter notebook server on $head_node 

# Note that the jupyter notebook command is available only because we have enabled the venv
command="jupyter lab --ip=0.0.0.0 --port=${jupyter_port} --NotebookApp.token=${jupyter_token}"
echo [INFO]: $command
$command &

echo [INFO]: Your env is up and running.

sleep infinity


