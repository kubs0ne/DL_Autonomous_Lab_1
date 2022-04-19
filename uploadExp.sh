scp -r Experiments/25 nct01029@dt01.bsc.es:/home/nct01/nct01029/AutoLab1/Experiments #Upload the current experiment
ssh nct01029@plogin1.bsc.es 'sbatch AutoLab1/Experiments/25/ex25.sh' #Submit sbatch 