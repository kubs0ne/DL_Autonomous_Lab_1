scp -r Experiments/19 nct01036@dt01.bsc.es:/home/nct01/nct01036/AutoLab1/Experiments #Upload the current experiment
ssh nct01036@plogin1.bsc.es 'sbatch AutoLab1/Experiments/19/evTest.sh' #Submit sbatch
