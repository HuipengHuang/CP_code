#!/bin/bash

# TransMIL runs (1-20)
for i in {1..30}; do
    python main.py --dataset camelyon16 --model transmil --save True --epoch 200 --cal_ratio 0.5 --batch_size 1 --alpha 0.1 --gpu 0 -mil True -auc True --final_activation_function softmax --loss standard --test_score thr --seed $i --save_feature False --optimizer adam -lr 0.0002 --weight_decay 1e-5 --extract_feature_model resnet50 --input_dimension 1024
done

for i in {30..60}; do
    python main.py --dataset camelyon16 --model transmil --save True --epoch 200 --cal_ratio 0.5 --batch_size 1 --alpha 0.1 --gpu 0 -mil True -auc True --final_activation_function softmax --loss standard --test_score aps --random True --seed $i --save_feature False --optimizer adam -lr 0.0002 --weight_decay 1e-5 --extract_feature_model resnet50 --input_dimension 1024
done

# RRTMIL runs (20-40)
for i in {60..90}; do
    python main.py --dataset camelyon16 --model rrtmil --save True --epoch 200 --cal_ratio 0.5 --batch_size 1 --alpha 0.1 --gpu 0 -mil True -auc True --final_activation_function softmax --loss standard --test_score thr --seed $i --save_feature False --optimizer adam -lr 0.0002 --weight_decay 1e-5 --extract_feature_model resnet50 --input_dimension 1024
done

