#!/bin/bash

main(){

source /home/mmacmahon/dp/bin/activate
#T1 = theta, T2 = file id T3 = bool (true = use all IDs), T4 = decayid to use if T3 false T5 = isFD? T6 = is_rhc? T7 = is_ftfp?

echo "$T1 $T2 $T3 $T4"
if [[ 1 == $T6 ]]
then
    echo "RHC"
fi
if [[ 1 == $T7 ]]
then
    echo "FTFP"
fi

if [[ 1 == $T5 ]]
then
    echo "FD"
    python /unix/dune2/mmacmahon/dune-prism/scripts/offaxis_fluxes_loop_fd.py $T1 $T2 $T3 $T4 $T6 $T7 $T8
    
else
    # For ND
    python /unix/dune2/mmacmahon/dune-prism/scripts/offaxis_fluxes_loop.py $T1 $T2 $T3 $T4 $T6 $T7 $T8

fi

}

main "$@"
