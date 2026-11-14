declare -a angles=(0.0 0.001568 0.003136 0.004704 0.006272 0.00784 0.009408 0.010976 0.012544 0.014112 0.01568 0.017248 0.018816 0.020384 0.021952 0.02352 0.025088 0.026656 0.028224 0.029792 0.03136 0.032928 0.034496 0.036064 0.037632 0.0392 0.040768 0.042336 0.043904 0.045472 0.04704 0.048608 0.050176 0.051744 0.053312 0.05488 0.056448 0.058016 0.059584 0.061152 0.06272)

for k in ${!angles[@]}; do
    for i in 1 2 3 4 5 6 7 8 9 10 13 14 ; do
	cat *_thetaoa${angles[k]}_decayid"$i"_NDLAr_$1_smallND2_qgsp.txt > thetaoa${angles[k]}_decayid"$i"_NDLAr_$1_smallND2.txt
	#echo "]" >> 3_thetaoa${angles[k]}_decayid"$i"_NDLAr_weights_smallND2.txt 
    done
    for i in 11 12 ; do
	cat *_thetaoa${angles[k]}_decayid"$i"pi_NDLAr_$1_smallND2_qgsp.txt > thetaoa${angles[k]}_decayid"$i"_pi_NDLAr_$1_smallND2.txt
	cat *_thetaoa${angles[k]}_decayid"$i"kp_NDLAr_$1_smallND2_qgsp.txt > thetaoa${angles[k]}_decayid"$i"_kp_NDLAr_$1_smallND2.txt
    done
done
