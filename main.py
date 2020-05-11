import numpy as np
import pandas as pd
import copy
import argparse
import json
import time
import os
import sys

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from src.sampling import iid, non_iid
from src.models import LR, MLP, CNNMnist,VGG, VGG16
from src.utils import global_aggregate, network_parameters, test_inference
from src.local_train import LocalUpdate
from src.attacks import attack_updates
from src.defense import defend_updates
from src.utils import predict, bn_update
from collections import OrderedDict, Counter
import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")
START_TIME = time.time()
############################## Reading Arguments ##############################

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, required=True,
    help="name of the experiment")
parser.add_argument('--num_users', type=int, default=5, 
    help="number of clients to create")
parser.add_argument('--train_batch_size', type=int, default=128, help="batch size for client training")
parser.add_argument('--test_batch_size', type=int, default=32, help="batch size for testing data")
parser.add_argument('--local_epochs_sampling', type=int, default=20, help="Number of local epochs without global aggregation (Phase2)")
# rank_param must be smaller than num_users*frac_clients, rank_param=0 will sample the diagonal matrix only
parser.add_argument('--rank_param', type=int, default=4, help="Low rank approxmation parameter")
# number of samples for computing bayesian averaging
parser.add_argument('--num_samples', type=int, default=20, help="Number of samples in testing phase")
parser.add_argument('--frac_users_phase2', type=float, default=1.0, help="Number of clients to run phase 2")
parser.add_argument('--device', type=str, default="gpu", help="device for Torch", choices=['cpu', 'gpu'])

parser.add_argument('--num_shards_user', type=int, default=5, help="number of classes to give to the user")
parser.add_argument('--dir', type=str, default='./outputs', 
    help="directory for savingoutputs")
parser.add_argument('--fname', type=str, default='test.npz', 
    help="directory for savingoutputs")
parser.add_argument('--model', type=str, default="MLP", 
		help="network structure to be used for training", 
		choices=['LR', 'MLP', 'CNN', 'VGG16'])
parser.add_argument('--data_source', type=str, default="CIFAR10", 
		help="dataset to be used", 
		choices=['MNIST', 'CIFAR10'])
parser.add_argument('--criterion', type=str, default="NLL", 
		help="evaluation criterion", 
		choices=['crossentropy', 'NLL'])
parser.add_argument('--frac_clients', type=float, default=1, 
		help="proportion of clients to use for local updates")
parser.add_argument('--global_optimizer', type=str, default='fedavg', 
		help="global optimizer to be used", 
		choices=['fedavg', 'fedavgm', 'scaffold', 'fedadam', 'fedyogi'])
parser.add_argument('--global_epochs', type=int, default=10, 
		help="number of global federated rounds")
parser.add_argument('--global_lr', type=float, default=0.05, 
		help="learning rate for global steps")
parser.add_argument('--local_epochs', type=int, default=1, 
		help="number of local client training steps")
parser.add_argument('--local_lr', type=float, default=0.05, 
		help="learning rate for local updates")
parser.add_argument('--local_optimizer', type=str, default='sgd', 
		help="local optimizer to be used", 
		choices=['sgd', 'adam', 'pgd'])
parser.add_argument('--batch_print_frequency', type=int, default=100, 
		help="frequency after which batch results need to be printed to the console")
parser.add_argument('--global_print_frequency', type=int, default=1, 
		help="frequency after which global results need to be printed to the console")
parser.add_argument('--global_store_frequency', type=int, default=1000, 
		help="frequency after which global results should be written to CSV")
parser.add_argument('--threshold_test_metric', type=float, default=0.95, 
		help="threshold after which the code should end")
parser.add_argument('--data_parallel', action='store_true', default=False,
		help="whether or not to use nn.DataParallel to train")
parser.add_argument('--momentum', type=float, default=0.0, help="momentum value for SGD")
parser.add_argument('--seed', type=int, default=0, help="seed for running the experiments")
parser.add_argument('--sampling', type=str, default="iid", help="sampling technique for client data", choices=['iid', 'non_iid'])
parser.add_argument('--train_test_split', type=float, default=1.0, help="train test split at the client end")
parser.add_argument('--mu', type=float, default=0.1, help="proximal coefficient for FedProx")
parser.add_argument('--beta1', type=float, default=0.9, help="parameter for FedAvgM and FedAdam")
parser.add_argument('--beta2', type=float, default=0.999, help="parameter for FedAdam")
parser.add_argument('--eps', type=float, default=1e-2, help="epsilon for adaptive methods")
parser.add_argument('--frac_byz_clients', type=float, default=0.0, help="proportion of clients that are picked in a round")
parser.add_argument('--is_attack', type=int, default=0, help="whether to attack or not")
parser.add_argument('--attack_type', type=str, default='fall', help="attack to be used", choices=['fall', 'label_flip', 'little', 'gaussian'])
parser.add_argument('--fall_eps', type=float, default=-5.0, help="epsilon value to be used for the Fall Attack")
parser.add_argument('--little_std', type=float, default=1.5, help="standard deviation to be used for the Little Attack")
parser.add_argument('--is_defense', type=int, default=0, help="whether to defend or not")
parser.add_argument('--defense_type', type=str, default='median', help="aggregation to be used", choices=['median', 'krum', 'trimmed_mean'])
parser.add_argument('--trim_ratio', type=float, default=0.1, help="proportion of updates to trim for trimmed mean")
parser.add_argument('--multi_krum', type=int, default=5, help="number of clients to pick after krumming")

args = parser.parse_args()

print(args)

if not(os.path.exists(args.dir)):
    os.makedirs(os.path.join(args.dir, args.exp_name), exist_ok=True)

with open(os.path.join(args.dir, 'command.sh'), "w") as f:
    f.write(" ".join(sys.argv))
    f.write("\n")
		
K=args.rank_param

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.criterion == "NLL":
  criterion = torch.nn.NLLLoss() # Default criterion set to NLL loss function
elif args.criterion == "crossentropy":    
  criterion = torch.nn.CrossEntropyLoss()

############################### Loading Dataset ###############################
if args.data_source == 'MNIST':
  data_dir = 'data/'
  transformation = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))
  ])
  train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transformation)
  test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transformation)
  
elif args.data_source == 'CIFAR10':
  #CIFAR10 dataset
  data_dir = 'data/'
  transformation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
  ])
  train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transformation)
  test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transformation)

print("Train and Test Sizes for %s - (%d, %d)"%(args.data_source, len(train_dataset), len(test_dataset))) 
################################ Sampling Data ################################
if args.sampling == 'iid':
  user_groups = iid(train_dataset, args.num_users, args.seed)
else:
  user_groups = non_iid(train_dataset, args.num_users, args.num_shards_user, args.seed)

################################ Defining Model ################################
if args.model == 'LR':
  global_model = LR(dim_in=28*28, dim_out=10, seed=args.seed)
elif args.model == 'MLP':
  global_model = MLP(dim_in=28*28, dim_hidden=200, dim_out=10, seed=args.seed)
elif args.model == 'CNN' and args.data_source == 'MNIST':
  global_model = CNNMnist(args.seed)
elif args.model == 'VGG16':
  global_model = models.vgg16(num_classes=10)

else:
  raise ValueError('Check the model and data source provided in the arguments.')

if args.data_parallel:
  global_model=nn.DataParallel(global_model)
print("Number of parameters in %s - %d."%(args.model, network_parameters(global_model)))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
global_model.to(device)
global_model.train()

global_weights = global_model.state_dict() # Setting the initial global weights

############################ Initializing Placeholder ############################

# Momentum parameter 'v' for FedAvgM & `m` for FedAdam & FedYogi
# Control variates for SCAFFOLD (Last one corresponds to the server variate)
v = OrderedDict()
m = OrderedDict()
c = [OrderedDict() for i in range(len(user_groups) + 1)]

for k in global_weights.keys():
  v[k] = torch.zeros(global_weights[k].shape, dtype=global_weights[k].dtype)
  m[k] = torch.zeros(global_weights[k].shape, dtype=global_weights[k].dtype)
  for idx, i in enumerate(c):
    c[idx][k] = torch.zeros(global_weights[k].shape, dtype=global_weights[k].dtype)

################################ Defining Model ################################

train_loss_updated = []
train_loss_all = []
test_loss = []
train_accuracy = []
test_accuracy = []
mus = [args.mu for i in range(args.num_users)]

num_classes = 10 # MNIST

# Picking byzantine users (they would remain constant throughout the training procedure)
if args.is_attack == 1:
  idxs_byz_users = np.random.choice(range(args.num_users), max(int(args.frac_byz_clients*args.num_users), 1), replace=False)
is_phase1=True
ep_times = []
for epoch in range(args.global_epochs):
  ################################# Client Sampling & Local Training #################################
  global_model.train()
  
  np.random.seed(epoch) # Picking a fraction of users to choose for training
  idxs_users = np.random.choice(range(args.num_users), max(int(args.frac_clients*args.num_users), 1), replace=False)
  
  local_updates, local_losses, local_sizes, control_updates = [], [], [], []
  train_time_per_process = []
  for idx in idxs_users: # Training the local models
    # start training timer for process
    pr_time = time.time()
    local_model = LocalUpdate(train_dataset, user_groups[idx], device, criterion,
        args.train_test_split, args.train_batch_size, args.test_batch_size)
    w, c_update, c_new, loss, local_size,_ = local_model.local_opt(args.local_optimizer, args.local_lr, 
                        args.local_epochs, global_model,is_phase1, args.momentum, mus[idx], c[idx], c[-1], 
                        epoch+1, idx+1, args.batch_print_frequency)

    c[idx] = c_new # Updating the control variates in the main list for that client
        
    local_updates.append(copy.deepcopy(w))
    control_updates.append(c_update)
    local_losses.append(loss)
    local_sizes.append(local_size)
    train_time_per_process.append(time.time()-pr_time)

  # start aggregation timer
  agg_time = time.time() 
  train_loss_updated.append(sum(local_losses)/len(local_losses)) # Appending global training loss
  #gw = copy.deepcopy(global_weights)
  #global_model.load_state_dict(gw) # [i for idx, i in enumerate(local_updates) if idx in idxs_to_use]
  global_model.load_state_dict(global_weights) 
  global_weights, v, m = global_aggregate(args.global_optimizer, global_weights, local_updates, 
                    local_sizes, args.global_lr, args.beta1, args.beta2,
                    v, m, args.eps, epoch+1)
  global_model.load_state_dict(global_weights)

  # compute time cost of epoch
  ep_time = (time.time() - agg_time) + max(train_time_per_process)
  ep_times.append(ep_time)

  ######################################### Model Evaluation #########################################
  global_model.eval()
  
  if args.train_test_split != 1.0:
    list_acc = []
    list_loss = []
    for idx in range(args.num_users):

      local_model = LocalUpdate(train_dataset, user_groups[idx], device,criterion, args.train_test_split, 
                  args.train_batch_size, args.test_batch_size)
      acc, loss = local_model.inference(global_model)
      list_acc.append(acc)
      list_loss.append(loss)

    train_loss_all.append(sum(list_loss)/len(list_loss))
    train_accuracy.append(sum(list_acc)/len(list_acc))
  
  # Evaluation on the hold-out test set at central server
  test_acc, test_loss_value = test_inference(global_model, test_dataset, device,criterion, args.test_batch_size)
  test_accuracy.append(test_acc)
  test_loss.append(test_loss_value)

  if (epoch+1) % args.global_print_frequency == 0 or (epoch+1) == args.global_epochs:
    msg = '| Global Round : {0:>4} | TeLoss - {1:>6.4f}, TeAcc - {2:>6.2f} %, TrLoss (U) - {3:>6.4f}'

    if args.train_test_split != 1.0:
      msg = 'TrLoss (A) - {4:>6.4f} % , TrAcc - {5:>6.2f} %'
      print(msg.format(epoch+1, test_loss[-1], test_accuracy[-1]*100.0, train_loss_updated[-1], 
              train_loss_all[-1], train_accuracy[-1]*100.0))
    else:
      print(msg.format(epoch+1, test_loss[-1], test_accuracy[-1]*100.0, train_loss_updated[-1]))

  if (epoch+1) % args.global_store_frequency == 0  or (epoch+1) == args.global_epochs or test_accuracy[-1] >= args.threshold_test_metric:
    if args.train_test_split != 1.0:
      out_arr = pd.DataFrame(np.array([list(range(epoch+1)), train_accuracy, test_accuracy, train_loss_updated, train_loss_all, test_loss]).T,
                columns=['epoch', 'train_acc', 'test_acc', 'train_loss_updated', 'train_loss_all', 'test_loss'])
    else:
      out_arr = pd.DataFrame(np.array([list(range(epoch+1)), test_accuracy, train_loss_updated, test_loss]).T,
        columns=['epoch', 'test_acc', 'train_loss_updated', 'test_loss'])

    out_arr.to_csv(f'{args.dir}/{args.exp_name}.csv', index=False)

  if test_accuracy[-1] >= args.threshold_test_metric:
    print("Terminating as desired threshold for test metric reached...")
    break
 
is_phase1=False
#for idx in idxs_users: # Training the local models
idxs_users = np.random.choice(range(args.num_users), max(int(args.frac_users_phase2*args.num_users), 1), replace=False)
final_updates=[]
test_acc_phase2=[]
test_loss_value_phase2=[]
local_model_phase2=copy.deepcopy(global_model)
test_acc, test_loss_value = test_inference(local_model_phase2, test_dataset, device, criterion,args.test_batch_size)
print('Initial Test accuracy is',test_acc)
print('Starting Phase2')
j=1
for idx in idxs_users: # Training the local models
  local_model = LocalUpdate(train_dataset, user_groups[idx], device, criterion,
      args.train_test_split, args.train_batch_size, args.test_batch_size)

  w, c_update, c_new, loss, local_size,_ = local_model.local_opt(args.local_optimizer, args.local_lr, 
                      args.local_epochs_sampling, global_model,is_phase1, args.momentum, mus[idx], c[idx], c[-1], 
                      epoch+1, idx+1, args.batch_print_frequency)
  local_model_phase2.load_state_dict(w)
  test_acc, test_loss_value = test_inference(local_model_phase2, test_dataset, device,criterion, args.test_batch_size)
  test_acc_phase2.append(test_acc)
  test_loss_value_phase2.append(test_loss_value)

  if j % args.global_print_frequency == 0 or (epoch+1) == args.global_epochs:
    msg = '| Global Round : {0:>4} | TeLoss - {1:>6.4f}, TeAcc - {2:>6.2f} %'

    if args.train_test_split != 1.0:
      msg = 'TrLoss (A) - {4:>6.4f} % , TrAcc - {5:>6.2f} %'
      print(msg.format(
              train_loss_all[-1], train_accuracy[-1]*100.0))
    else:
      print(msg.format(j, test_loss_value_phase2[-1], test_acc_phase2[-1]*100.0))
  j+=1

  final_updates.append(copy.deepcopy(w))
w_final = OrderedDict()
mean = OrderedDict()
std = OrderedDict()

for k in global_weights.keys():
  w_final[k] = torch.zeros(global_weights[k].shape, dtype=global_weights[k].dtype).to(device)
  mean[k] = torch.zeros(global_weights[k].shape, dtype=global_weights[k].dtype).to(device)
  std[k] = torch.zeros(global_weights[k].shape, dtype=global_weights[k].dtype).to(device)
total_size=sum(local_sizes)
print('will start calculating mean and variance')
mom_time = time.time()
for k in global_weights.keys():
  for i in range(len(final_updates)):
    mean[k] = ((i)*mean[k]+final_updates[i][k])/(i+1)
    std[k] = (i*std[k]+torch.mul(final_updates[i][k], final_updates[i][k]))/(i+1)
  std[k] = (torch.abs(std[k] - torch.mul(mean[k], mean[k]))) ** 0.5
mom_time = time.time()-mom_time

print('done!!')

swag_model=copy.deepcopy(global_weights)
swag_model1=copy.deepcopy(global_model)
mean_model_test=copy.deepcopy(global_model)
mean_model_test.load_state_dict(mean)

test_acc, test_loss_value = test_inference(mean_model_test, test_dataset, device,criterion, args.test_batch_size)
print('Initial Test accuracy is',test_acc)
print('Starting testing phase')

loader_train=torch.utils.data.DataLoader(train_dataset,batch_size=128,shuffle=True)
loader_test=torch.utils.data.DataLoader(test_dataset,batch_size=128,shuffle=False)
eps=1e-12
num_classes=10
predictions = np.zeros((len(loader_test.dataset), num_classes))

num_samples=args.num_samples
for j in range(num_samples):
	ep = torch.distributions.MultivariateNormal(torch.zeros(len(final_updates)), torch.diag(torch.ones(len(final_updates)))).rsample()
	print('Sampling diagonal matrix')
	for k in swag_model.keys():

		swag_model[k] = torch.normal(mean[k],std[k])
		
	if K != 0 and K < len(final_updates):
		print('Sampling low-rank matrix')
		for i in range(len(final_updates)-K,len(final_updates),1):
			for k in swag_model.keys():
				swag_model[k] += (final_updates[i][k]-mean[k])*ep[i]/np.float(K-1)/np.sqrt(2)

	swag_model1.load_state_dict(swag_model)
	test_acc, test_loss_value = test_inference(swag_model1, test_dataset, device,criterion, args.test_batch_size)
	print('Sampled model Test accuracy is',test_acc)
	print('Updating batch norm')
	#swag_model.sample(scale=0.5, cov=args.cov_mat)
	bn_update(loader_train, swag_model1)
	sample_res = predict(loader_test, swag_model1, device)
	predictions += sample_res["predictions"]
	targets = sample_res["targets"]
	predictions /= num_samples

swag_accuracies = np.mean(np.argmax(predictions, axis=1) == targets)*100
swag_nlls = -np.mean(np.log(predictions[np.arange(predictions.shape[0]), targets] + eps))
print('Final Accurcay is',swag_accuracies)
print('Final Negative log likelihood is',swag_nlls)

torch.save(mean, os.path.join(args.dir, f'{args.exp_name}_mean_model.pt'))
torch.save(std, os.path.join(args.dir, f'{args.exp_name}_std_model.pt'))

np.savez(
		os.path.join(args.dir, args.fname), 
        mom_time=mom_time,
        ep_times=ep_times,
        predictions=predictions,
        targets=targets,
		nnl=swag_nlls,
    )
RUN_TIME = (time.time() - START_TIME) / 60
print(f'total runtime: {RUN_TIME} minutes.')
