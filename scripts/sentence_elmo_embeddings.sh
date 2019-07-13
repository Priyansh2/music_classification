#!/bin/bash
#SBATCH -A research
#SBATCH -n 40
#SBATCH --qos=medium
#SBATCH -p long
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
source /home/priyansh.agrawal/music_classification/bilm-tf/flair/bin/activate
module add cuda/9.0
module add cudnn/7-cuda-9.0
export result_dir="/scratch/$USER"
export output_dir="/scratch/$USER/elmo_data_songs"
if [ -d $result_dir ]
then
rm -rf $result_dir
fi
if [ -d $output_dir ]
then
rm -rf $output_dir
fi

mkdir -p /scratch/$USER/elmo_data_songs/batches
mkdir -p /scratch/$USER/elmo_models
#rsync -avzP ada:/share1/$USER/word_embeddings/elmo_models/music /scratch/$USER/elmo_models
rsync -avzP ada:/share1/$USER/elmo_data_songs /scratch/$USER
export dir="/home/priyansh.agrawal/music_classification/elmo_scripts"
python $dir/sentence_elmo_embeddings.py
#rsync -avzP /scratch/$USER/elmo_data_songs ada:/share1/$USER
