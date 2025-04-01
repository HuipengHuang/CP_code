#!/bin/bash
for i in {15..20}; do
    python main.py --dataset camelyon16 --model rrtmil --save True --epoch 200 --cal_ratio 0.5 --batch_size 1 --alpha 0.1 --gpu 1 -mil True -auc True --final_activation_function softmax --loss standard --test_score aps --random True --seed $i --save_feature False --optimizer adam -lr 0.0002 --weight_decay 1e-5 --extract_feature_model resnet50 --input_dimension 1024 --save_memory True
done

# GradAttention runs (40-60)
for i in {20..25}; do
    python main.py --dataset camelyon16 --model gradattention --save True --epoch 200 --cal_ratio 0.5 --batch_size 1 --alpha 0.1 --gpu 1 -mil True -auc True --final_activation_function softmax --loss standard --test_score thr --seed $i --save_feature False --optimizer adam -lr 0.0002 --weight_decay 1e-5 --extract_feature_model resnet50 --input_dimension 1024 --save_memory True
done

for i in {25..30}; do
    python main.py --dataset camelyon16 --model gradattention --save True --epoch 200 --cal_ratio 0.5 --batch_size 1 --alpha 0.1 --gpu 1 -mil True -auc True --final_activation_function softmax --loss standard --test_score aps --random True --seed $i --save_feature False --optimizer adam -lr 0.0002 --weight_decay 1e-5 --extract_feature_model resnet50 --input_dimension 1024 --save_memory True
done