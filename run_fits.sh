#!/bin/bash
  
main(){

source /home/mmacmahon/dp/bin/activate

python /unix/dune2/mmacmahon/dune-prism/scripts/off_axis_nuisance_all.py $T1 $T2 $T3 #file name, bool run_over_time, bool run_contour


}

main "$@"
