#!/bin/bash

#SBATCH --job-name="MAMeEx24"

#SBATCH --qos=debug

#SBATCH --workdir=.

#SBATCH --output=/home/nct01/nct01029/AutoLab1/Experiments/24/MAMeEx24_%j.out

#SBATCH --error=/home/nct01/nct01029/AutoLab1/Experiments/24/MAMeEx24_%j.err

#SBATCH --cpus-per-task=40

#SBATCH --gres gpu:1

#SBATCH --time=01:00:00

module purge; module load gcc/8.3.0 ffmpeg/4.2.1 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 opencv/4.1.1 python/3.7.4_ML
cd /home/nct01/nct01029/AutoLab1/Experiments/24
python ex24.py