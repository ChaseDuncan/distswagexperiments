
##########
# distributed-SWAG-Diag  VGG16 CIFAR10 
# users 1,2,4,10  
# global epochs 200   
# samples 1,5,10,20

dir="/shared/mrfil-data/cddunca2/pgmproject/experiments/"

local_epochs=1
local_epochs_sampling=30
global_epochs=150

# SWAG-diagonal
rank_param=0
for num_samples in 1 5 10 20;do
  for num_users in 1 2 4 10;do
    exp_name="VC-${num_users}-${global_epochs}-${local_epochs}-${local_epochs_sampling}-${num_samples}-${rank_param}"
    python main.py --exp_name=${exp_name} --num_users=${num_users} --global_epochs=${global_epochs} --threshold_test_metric=0.86\
      --local_epochs=${local_epochs} --local_epochs_sampling=${local_epochs_sampling} --num_samples=${num_samples} \
      --fname="${exp_name}.npz" --rank_param=${rank_param} --train_batch_size=1024 --test_batch_size=1024 --global_lr=.2\
      --data_source=CIFAR10 --model=VGG16 --data_parallel --criterion=crossentropy   --dir="${dir}${exp_name}/"
  done
done

# SWAG with low rank approx of covariance
for num_samples in 1 5 10 20;do
  for num_users in 2 4 10;do
    rank_param=$(($num_users-1))
    exp_name="VC-${num_users}-${global_epochs}-${local_epochs}-${local_epochs_sampling}-${num_samples}-${rank_param}"
    python main.py --exp_name=${exp_name} --num_users=${num_users} --global_epochs=${global_epochs} --threshold_test_metric=0.86 \
      --local_epochs=${local_epochs} --local_epochs_sampling=${local_epochs_sampling} --num_samples=${num_samples} \
      --fname="${exp_name}.npz" --rank_param=${rank_param} --train_batch_size=1024 --test_batch_size=1024 --global_lr=.2\
      --data_source=CIFAR10 --model=VGG16 --data_parallel --criterion=crossentropy   --dir="${dir}${exp_name}/"
  done
done
