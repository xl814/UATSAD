#!/bin/bash

echo "script: $0"
echo "args: $1"
if [ "$1" == "SMAP_E13" ]; then
    dataset="SMAP_E13"
    win_size=64
    lambda1=2
    lambda2=1
    epochs2=10
elif [ "$1" == "SMAP_P1" ]; then
    dataset="SMAP_P1"
    win_size=96
    lambda1=2
    lambda2=1
    epochs2=5
elif [ "$1" == "SMAP_E1" ]; then
    dataset="SMAP_E1"
    win_size=64
    lambda1=2
    lambda2=1
    epochs2=10
elif [ "$1" == "NAB_Ambient" ]; then
    dataset="NAB_Ambient"
    win_size=128
    lambda1=2
    lambda2=1
    epochs2=10
elif [ "$1" == "NAB_Taxi" ]; then
    dataset="NAB_Taxi"
    win_size=128
    lambda1=2
    lambda2=1
    epochs2=10
elif [ "$1" == "NAB_Machine" ]; then
    dataset="NAB_Machine"
    win_size=128
    lambda1=2
    lambda2=1
    epochs2=10
elif [ "$1" == "SMD_machine3-4" ]; then
    dataset="SMD_machine3-4"
    win_size=128
    lambda1=2
    lambda2=1
    epochs2=10
elif [ "$1" == "SMD_machine2-1" ]; then
    dataset="SMD_machine2-1"
    win_size=64
    lambda1=2
    lambda2=1
    epochs2=10
elif [ "$1" == "MSL_P11" ]; then
    dataset="MSL_P11"
    win_size=96
    lambda1=2
    lambda2=1
    epochs2=5
elif [ "$1" == "MSL_F7" ]; then
    dataset="MSL_F7"
    win_size=128
    lambda1=2
    lambda2=1
    epochs2=10
elif [ "$1" == "UCR_InternalBleeding16" ]; then
    dataset="UCR_InternalBleeding16"
    win_size=64
    lambda1=2
    lambda2=1
    epochs2=10
elif [ "$1" == "UCR_InternalBleeding17" ]; then
    dataset="UCR_InternalBleeding17"
    win_size=64
    lambda1=2
    lambda2=1
    epochs2=10
else 
    echo "Unknown dataset"
    exit 1
fi
for loss_type in "rnll" "mse" "nll"
do
    for seed in {1,2,3,4,5}
    do
        echo "run in '$dataset' with seed: $seed loss_type: $loss_type win_size: $win_size lambda1: $lambda1 lambda2: $lambda2 epochs2: $epochs2"
        python ./src/experiment.py --seed $seed --dataset $dataset --loss_type $loss_type --win_size $win_size --lambda1 $lambda1 --lambda2 $lambda2 --epochs2 $epochs2
    done
done

