#!/bin/bash

declare -a time=(1 2 3 4 5 6) #0.01045 0.0209 0.03136 0.04181 0.05226 0.06272)


for i in ${!time[@]}; do #6); do
    qsub -q medium -l walltime=04:00:00 -v T1=${time[i]},T2=1,T3=0 run_fits.sh
        #./run_flux_loop.sh ${angles[i]} $j 1 0 $1
        #echo ${angles[i]} " " $j
done
