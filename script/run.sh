#!/bin/bash

for loss_type in "rnll" "mse" "nll"
do
    for dataset in   "SMAP_E13" "SMAP_P1" "SMD_machine2-1" "NAB_Ambient" "NAB_Texi" "NAB_Machine"  "SMD_machine3-4" # "MSL_P11" "MSL_F7" # "SMAP_E1" "UCR_InternalBleeding16" "UCR_InternalBleeding17"  # "SMAP_E13" "SMAP_P1" "SMD_machine2-1" "NAB_Ambient" "NAB_Texi" "NAB_Machine"  "SMD_machine3-4" #   #  "!SMD_machine1-3"  
    do 
        for seed in  {1,2,3,4,5}
        do
            echo "run in '$dataset' with seed: $seed loss_type: $loss_type"
            if [ $loss_type == "mse" ]; then
                python experiment.py --seed $seed --dataset $dataset --loss_type $loss_type --win_size 128 --n_epoch 5
            else
                python experiment.py --seed $seed --dataset $dataset --loss_type $loss_type --win_size 128
            fi
        done
    done
done

