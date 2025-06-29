#!/bin/bash

echo "script: $0"
echo "args: $1"

if [ "$1" == "" ]; then
    for alea in {0.707,1.0,1.414} 
    do 
        for anomaly_value in {10,20,30}
        do
            for seed in {1,2,3,4,5}
            do  
                echo "run toy experiment with seed: $seed alea: $alea anomaly_value: $anomaly_value "
                if [ $anomaly_value == 30 ]; then
                    python ./src/toy_experiment.py --seed $seed --alea $alea --anomaly_value $anomaly_value --epochs2 15 --metric True
                else 
                    python ./src/toy_experiment.py --seed $seed --alea $alea --anomaly_value $anomaly_value --epochs2 10 --metric True
                fi
            done
        done
    done
elif [ "$1" == "sumplement" ]; then
    for learning_rate in {1e-3,5e-4,1e-4,5e-5}
    do
        for lambda1 in {2,3,4}
        do
            for lambda2 in {1,2,3}
            do
                for alea in 0.707   
                do 
                    for anomaly_value in {10,20,30} 
                    do
                        for seed in {1,2,3,4,5}
                        do
                            echo "run toy experiment with seed: $seed alea: $alea anomaly_value: $anomaly_value lambda1: $lambda1 lambda2: $lambda2 lr: $learning_rate"
                            python ./src/toy_experiment.py --seed $seed --alea $alea --anomaly_value $anomaly_value \
                            --lambda1 $lambda1 --lambda2 $lambda2 --learning_rate $learning_rate --metric True
                        done
                    done
                done
            done
        done
    done
else
    echo "Nothing is running"
fi




