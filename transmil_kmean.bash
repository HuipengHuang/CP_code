for i in {1..10}; do
    python main.py --dataset camelyon16 --model transmil --save True --epoch 67 --cal_ratio 0.5 --batch_size 1 --alpha 0.1 --gpu 0 -mil True -auc True --final_activation_function softmax --loss standard --test_score thr  --save_feature False --optimizer adam -lr 0.0002 --weight_decay 1e-5 --extract_feature_model resnet50 --input_dimension 1024 --aggregation kmean --k 2
done

for i in {1..10}; do
    python main.py --dataset camelyon16 --model transmil --save True --epoch 67 --cal_ratio 0.5 --batch_size 1 --alpha 0.1 --gpu 0 -mil True -auc True --final_activation_function softmax --loss standard --test_score thr --save_feature False --optimizer adam -lr 0.0002 --weight_decay 1e-5 --extract_feature_model resnet50 --input_dimension 1024 --aggregation kmean --k 5
done

for i in {1..10}; do
    python main.py --dataset camelyon16 --model transmil --save True --epoch 67 --cal_ratio 0.5 --batch_size 1 --alpha 0.1 --gpu 0 -mil True -auc True --final_activation_function softmax --loss standard --test_score thr --save_feature False --optimizer adam -lr 0.0002 --weight_decay 1e-5 --extract_feature_model resnet50 --input_dimension 1024 --aggregation kmean --k 10
done


for i in {10..20}; do
    python main.py --dataset camelyon16 --model transmil --save True --epoch 67 --cal_ratio 0.5 --batch_size 1 --alpha 0.1 --gpu 0 -mil True -auc True --final_activation_function softmax --loss standard --test_score aps --random True --save_feature False --optimizer adam -lr 0.0002 --weight_decay 1e-5 --extract_feature_model resnet50 --input_dimension 1024 --aggregation kmean --k 2
done

for i in {10..20}; do
  python main.py --dataset camelyon16 --model transmil --save True --epoch 67 --cal_ratio 0.5 --batch_size 1 --alpha 0.1 --gpu 0 -mil True -auc True --final_activation_function softmax --loss standard --test_score aps --random True --save_feature False --optimizer adam -lr 0.0002 --weight_decay 1e-5 --extract_feature_model resnet50 --input_dimension 1024 --aggregation kmean --k 5
done

for i in {10..20}; do
  python main.py --dataset camelyon16 --model transmil --save True --epoch 67 --cal_ratio 0.5 --batch_size 1 --alpha 0.1 --gpu 0 -mil True -auc True --final_activation_function softmax --loss standard --test_score aps --random True --save_feature False --optimizer adam -lr 0.0002 --weight_decay 1e-5 --extract_feature_model resnet50 --input_dimension 1024 --aggregation kmean --k 10
done

