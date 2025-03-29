#!/bin/bash

for i in {1..10}; do     python main.py --dataset camelyon17 --model transmil --save True --epoch 100 --cal_ratio 0.5 --batch_size 1 --alpha 0.1 --gpu 0 -mil True -auc True --final_activation_function sigmoid --loss bce --test_score thr --seed $i --save_feature False --optimizer adam -lr 0.0001; done

for i in {10..20}; do     python main.py --dataset camelyon17 --model transmil --save True --epoch 100 --cal_ratio 0.5 --batch_size 1 --alpha 0.1 --gpu 0 -mil True -auc True --final_activation_function sigmoid --loss bce --test_score aps --random True --seed $i --save_feature False --optimizer adam -lr 0.0001; done

for i in {20..30}; do     python main.py --dataset camelyon17 --model rrtmil --save True --epoch 100 --cal_ratio 0.5 --batch_size 1 --alpha 0.1 --gpu 0 -mil True -auc True --final_activation_function sigmoid --loss bce --test_score thr --seed $i --save_feature False --optimizer adam -lr 0.0001; done

for i in {30..40}; do     python main.py --dataset camelyon17 --model rrtmil --save True --epoch 100 --cal_ratio 0.5 --batch_size 1 --alpha 0.1 --gpu 0 -mil True -auc True --final_activation_function sigmoid --loss bce --test_score aps --random True --seed $i --save_feature False --optimizer adam -lr 0.0001; done