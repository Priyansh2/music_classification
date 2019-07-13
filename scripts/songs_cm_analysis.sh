#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --qos=medium
#SBATCH -p long
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
source /home/$USER/csnli/py2/bin/activate
export dir="/home/priyansh.agrawal/csnli"
python $dir/songs_cm_analysis.py
