#!/bin/sh
#SBATCH --partition=gpu --qos=gpu
#SBATCH --gres=gpu:V100:2
#SBATCH --time=72:00:00  #change HERE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=187000
# Memory per node specification is in MB. It is optional.
# The default limit is 3000MB per core.
#SBATCH --job-name="puzzles"     #change HERE
##SBATCH --array=3-5
#SBATCH --output=./results/console/console_%A.out
#SBATCH --error=./results/console/console_%A.err
#SBATCH --mail-user=ningjiwe@buffalo.edu   #change HERE
#SBATCH --mail-type=ALL
##SBATCH --requeue
#Specifies that the job will be requeued after a node failure.
#The default is that the job will not be requeued.
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

#module load python/my-python-3x.lua  #change HERE

#module load tensorflow/1.12.0-gpu-py36
#module load cuda/10.1
#module load intel/13.1
#module load intel-mpi/4.1.3
#module list
ulimit -s unlimited
#
# The initial srun will trigger the SLURM prologue on the compute nodes.
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
echo NPROCS=$NPROCS
#The PMI library is necessary for srun
#export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
#python -u ./pipeline.py ${SLURM_ARRAY_TASK_ID}
python -u ./hitdeck_exact.py
echo "All Done!"
