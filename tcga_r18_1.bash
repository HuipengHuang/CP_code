#!/bin/bash

# TransMIL runs (6 times)
for i in {1..6}; do
    python main.py --dataset tcga_lung_cancer --model transmil --save True --epoch 200 --cal_ratio 0.5 --batch_size 1 --alpha 0.1 --gpu 0 -mil True -auc True --final_activation_function softmax --loss standard --test_score thr --seed $i --save_feature False --optimizer adam -lr 0.0002 --weight_decay 1e-5 --extract_feature_model resnet18 --input_dimension 512 --cross_validation 5
done

for i in {7..12}; do
    python main.py --dataset tcga_lung_cancer --model transmil --save True --epoch 200 --cal_ratio 0.5 --batch_size 1 --alpha 0.1 --gpu 0 -mil True -auc True --final_activation_function softmax --loss standard --test_score aps --random True --seed $i --save_feature False --optimizer adam -lr 0.0002 --weight_decay 1e-5 --extract_feature_model resnet18 --input_dimension 512 --cross_validation 5
done

# RRTMIL runs (6 times)
for i in {13..18}; do
    python main.py --dataset tcga_lung_cancer --model rrtmil --save True --epoch 200 --cal_ratio 0.5 --batch_size 1 --alpha 0.1 --gpu 0 -mil True -auc True --final_activation_function softmax --loss standard --test_score thr --seed $i --save_feature False --optimizer adam -lr 0.0002 --weight_decay 1e-5 --extract_feature_model resnet18 --input_dimension 512 --cross_validation 5
done