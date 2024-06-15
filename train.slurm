#!/bin/bash


#SBATCH --job-name=mri2mra:jun7
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asagil@uw.edu

#SBATCH --account=escience
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=30gb
#SBATCH --gpus=gpu-a40:1
#SBATCH --time=7-00:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/gscratch/scrubbed/asagil/t1w-mra
#SBATCH --export=all
#SBATCH --output=./out_%j.txt # where STDOUT goes
#SBATCH --error=./err_%j.txt # where STDERR goes


/gscratch/escience/asagil/miniconda3/envs/mri2mra/bin/python "train_model.py" --image_model "UNet" --loss_function "perceptuallossvgg16" --loss_layer "block3_conv2" --dataset_root "data" --dataset_name "tfrec" --job_name "mri2mra:jun7" --batch_size 128 --image_shape 427 427 --n_train_images 328 --n_valid_images 55 --n_epochs 200