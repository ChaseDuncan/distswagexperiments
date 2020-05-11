
##########
# distributed-SWAG-Diag  VGG16 CIFAR10 
# users 1,2,4,10  
# global epochs 200   
# samples 1,5,10,20

dir="/shared/mrfil-data/cddunca2/pgmproject/experiments/"

local_epochs=1
local_epochs_sampling=30
global_epochs=150

num_users=1
num_samples=1

for num_samples in 1 5 10 20;do
  for num_users in 1 2 4 10;do
    exp_name="VC-${num_users}-${global_epochs}-${local_epochs}-${local_epochs_sampling}-${num_samples}"
    python main.py --exp_name=${exp_name} --num_users=${num_users} --global_epochs=${global_epochs} \
      --local_epochs=${local_epochs} --local_epochs_sampling=${local_epochs_sampling} --num_samples=${num_samples} \
      --data_source=CIFAR10 --model=VGG16 --data_parallel --criterion=crossentropy   --dir="${dir}${exp_name}"
  done
done
