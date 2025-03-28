#!/bin/bash

python main.py --dataset camelyon17 --model transmil --save True --epoch 100 --cal_ratio 0.5 --batch_size 1 --alpha 0.5 --gpu 0 -mil True -auc True --final_activation_function sigmoid --loss bce --test_score thr

python main.py --dataset camelyon17 --model transmil --save True --epoch 100 --cal_ratio 0.5 --batch_size 1 --alpha 0.5 --gpu 0 -mil True -auc True --final_activation_function sigmoid --loss bce --test_score aps

python main.py --dataset camelyon17 --model attention --save True --epoch 100 --cal_ratio 0.5 --batch_size 1 --alpha 0.5 --gpu 0 -mil True -auc True --final_activation_function sigmoid --loss bce --test_score thr

python main.py --dataset camelyon17 --model attention --save True --epoch 100 --cal_ratio 0.5 --batch_size 1 --alpha 0.5 --gpu 0 -mil True -auc True --final_activation_function sigmoid --loss bce --test_score aps

python main.py --dataset camelyon17 --model rrtmil --save True --epoch 100 --cal_ratio 0.5 --batch_size 1 --alpha 0.5 --gpu 0 -mil True -auc True --final_activation_function sigmoid --loss bce --test_score thr

python main.py --dataset camelyon17 --model rrtmil --save True --epoch 100 --cal_ratio 0.5 --batch_size 1 --alpha 0.5 --gpu 0 -mil True -auc True --final_activation_function sigmoid --loss bce --test_score aps