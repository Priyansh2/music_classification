#!/bin/bash
#SBATCH -A research
#SBATCH -n 20
#SBATCH --qos=medium
#SBATCH -p long
#SBATCH --gres=gpu:3
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
export dir="/home/priyansh.agrawal/music_classification/bilm-tf"
export resource_dir="elmo_data_cm"
export output_dir="cm_twitter"

#export resource_dir="elmo_data_songs"
#export output_dir="music"

export data_dir="/home/priyansh.agrawal/music_classification/elmo_scripts/$resource_dir"
export result_dir="/scratch/$USER/elmo_models"
if [ -d $result_dir ]
then
rm -rf $result_dir
fi
mkdir -p $result_dir
source /home/priyansh.agrawal/music_classification/bilm-tf/elmo/bin/activate
module load cuda/8.0
module load cudnn/5.1-cuda-8.0
python $dir/bin/train_elmo.py --train_prefix $data_dir/xtrain.txt --vocab_file $data_dir/tr_vocab.txt --save_dir $result_dir
python $dir/bin/run_test.py --test_prefix $data_dir/xval.txt --vocab_file $data_dir/tr_vocab.txt --save_dir $result_dir
python bin/dump_weights.py --save_dir /scratch/priyansh.agrawal/elmo_models --outfile /scratch/priyansh.agrawal/elmo_models/weights.hdf5
rsync -avzP $result_dir ada:/share1/$USER/$output_dir
