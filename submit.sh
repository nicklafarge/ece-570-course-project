#!/bin/bash

#SBATCH -o .templogs/sc.out-%a
#SBATCH -a 1-24

# Load Python Module
source $HOME/VirtualEnvs/research3/bin/activate
export PYTHONPATH="$PYTHONPATH:$HOME/Projects/ece-570-course-project/"

# Output information
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Number of Tasks: " $SLURM_ARRAY_TASK_COUNT

# Run script
python $HOME/Projects/ece-570-course-project/runner.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT