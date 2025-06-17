#!/bin/bash

echo "script: $0"
echo "args: $1"
if [ "$1" == "SMAP_E13" ]; then
    learning_rate=1e-3
    dataset="SMAP_E13"
    win_size=128
    lambda1=3
    lambda2=3
elif [ "$1" == "SMAP_P1" ]; then
    learning_rate=1e-3
    dataset="SMAP_P1"
    win_size=128
    lambda1=3
    lambda2=3
elif [ "$1" == "SMAP_E1" ]; then
    learning_rate=1e-3
    dataset="SMAP_E1"
    win_size=64
    lambda1=2
    lambda2=1
elif [ "$1" == "NAB_Ambient" ]; then
    learning_rate=1e-3
    dataset="NAB_Ambient"
    win_size=128
    lambda1=3
    lambda2=3
elif [ "$1" == "NAB_Taxi" ]; then
    learning_rate=1e-3
    dataset="NAB_Taxi"
    win_size=128
    lambda1=3
    lambda2=3
elif [ "$1" == "NAB_Machine" ]; then
    learning_rate=1e-3
    dataset="NAB_Machine"
    win_size=128
    lambda1=3
    lambda2=3
elif [ "$1" == "SMD_machine3-4" ]; then
    learning_rate=1e-3
    dataset="SMD_machine3-4"
    win_size=128
    lambda1=3
    lambda2=3
elif [ "$1" == "SMD_machine2-1" ]; then
    learning_rate=1e-3
    dataset="SMD_machine2-1"
    win_size=128
    lambda1=3
    lambda2=3
elif [ "$1" == "MSL_P11" ]; then
    learning_rate=1e-4
    dataset="MSL_P11"
    win_size=128
    lambda1=2
    lambda2=1
elif [ "$1" == "MSL_F7" ]; then
    learning_rate=1e-4
    dataset="MSL_F7"
    win_size=128
    lambda1=2
    lambda2=1
elif [ "$1" == "UCR_InternalBleeding16" ]; then
    learning_rate=1e-3
    dataset="UCR_InternalBleeding16"
    win_size=64
    lambda1=2
    lambda2=1
elif [ "$1" == "UCR_InternalBleeding17" ]; then
    learning_rate=1e-3
    dataset="UCR_InternalBleeding17"
    win_size=64
    lambda1=2
    lambda2=1
else 
    echo "Unknown dataset"
    exit 1
fi
for loss_type in "rnll" "mse" "nll"
do
    for seed in  {1,2,3,4,5}
    do
        echo "run in '$dataset' with seed: $seed loss_type: $loss_type win_size: $win_size learning_rate: $learning_rate lambda1: $lambda1 lambda2: $lambda2"
        python ./src/experiment.py --seed $seed --dataset $dataset --loss_type $loss_type --win_size $win_size --learning_rate $learning_rate --lambda1 $lambda1 --lambda2 $lambda2
    done
done

