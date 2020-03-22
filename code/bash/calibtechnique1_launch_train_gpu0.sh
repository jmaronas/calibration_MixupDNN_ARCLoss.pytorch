modelall=('densenet-121')
dropout=(0.0)
dataset=( "cifar100" ) #cifar10 cifar100 birds cars svhn
lamda=( "0.5" "1" "2" "4" "8")
n_gpu="0 1"

for model in "${modelall[@]}"
do
  for drop in "${dropout[@]}"
  do
    for data in "${dataset[@]}"
    do

      for lam in "${lamda[@]}"
      do
	#costtype="avg[square[conf_sub_acc]]"

         #python ../main_calibracion1_valid.arc_train.ce.py --model_net $model --dropout $drop --dataset $data --n_gpu $n_gpu --cost_type $costtype --lamda $lam 

    	 costtype="square[avgconf_sub_acc]"

         python ../main_calibracion1_valid.arc_train.ce.py --model_net $model --dropout $drop --dataset $data --n_gpu $n_gpu --cost_type $costtype --lamda $lam 

         python ../main_calibracion2_valid.arc_train.ce.py --model_net $model --dropout $drop --dataset $data --n_gpu $n_gpu --cost_type $costtype --lamda $lam  --bins_for_train 5 





      done
    done
  done
done



