source /usr/local/anaconda3/bin/activate ~/my_envs/ARD_python3.7_pytorch1.0.0_cuda100/

## Baseline

 python main_baseline.py        --model_net resnet-18 --dataset cars --n_gpu 0 --dropout 0.0 --use_valid_set 0 

## Baseline + Mixup

 python main_baseline_mixup.py  --model_net resnet-18 --dataset cars --mixup_coeff 0.4 --n_gpu 0 --dropout 0.0 --use_valid_set 0 

## MMCE

 python main_baseline_MMCE.py  --model_net resnet-18 --dataset cars --n_gpu 0 --dropout 0.0 --use_valid_set 0 --lamda 0.5

## MMCE + MIXUP

 python main_mixup_MMCE.py     --model_net resnet-18 --dataset cars --mixup_coeff 0.4 --n_gpu 0 --dropout 0.0 --use_valid_set 0 --lamda 2.0 --cost_over_mix_image 0 

## ARC

 python main_baseline_ARC.py  --model_net resnet-18 --dataset cars --n_gpu 0 --dropout 0.0 --use_valid_set 0 --cost_type 'avg[square[conf_sub_acc]]' --bins_for_train 1 --lamda 10

## ARC + MIXUP

 python main_mixup_ARC.py     --model_net resnet-18 --dataset cars --mixup_coeff 0.4 --n_gpu 0 1 --dropout 0.0 --use_valid_set 0 --cost_type 'avg[square[conf_sub_acc]]' --bins_for_train 5 15 30 --lamda 28

## ARC over validation set and CE over train

 python main_valid.ARC_train.CE.py --model_net resnet-18 --dataset cifar10 --n_gpu 0 --dropout 0.0 --use_valid_set 1 --cost_type 'square[avgconf_sub_acc]' --lamda 0.0 --bins_for_train 1 
 python main_valid.ARC_train.CE.py --model_net resnet-18 --dataset cifar10 --n_gpu 0 --dropout 0.0 --use_valid_set 1 --cost_type 'square[avgconf_sub_acc]' --lamda 2.0 --bins_for_train 1 
 python main_valid.ARC_train.CE.py --model_net resnet-18 --dataset cifar10 --n_gpu 0 --dropout 0.0 --use_valid_set 1 --cost_type 'square[avgconf_sub_acc]' --lamda 4.0 --bins_for_train 1 
 python main_valid.ARC_train.CE.py --model_net resnet-18 --dataset cifar10 --n_gpu 0 --dropout 0.0 --use_valid_set 1 --cost_type 'square[avgconf_sub_acc]' --lamda 8.0 --bins_for_train 1 
 python main_valid.ARC_train.CE.py --model_net resnet-18 --dataset cifar10 --n_gpu 0 --dropout 0.0 --use_valid_set 1 --cost_type 'square[avgconf_sub_acc]' --lamda 16.0 --bins_for_train 1 
