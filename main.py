import numpy as np
import pandas as pd
import copy
import argparse
import json

import torch
from torchvision import datasets, transforms

from src.sampling import iid, non_iid
from src.models import LR, MLP, CNNMnist
from src.utils import global_aggregate, network_parameters, test_inference
from src.local_train import LocalUpdate
from src.attacks import attack_updates
from src.defense import defend_updates

from collections import OrderedDict, Counter

import warnings
warnings.filterwarnings("ignore")

############################## Reading Arguments ##############################

parser = argparse.ArgumentParser()

parser.add_argument('--exp_name', type=str, default='', help="name of the experiment")
parser.add_argument('--seed', type=int, default=0, help="seed for running the experiments")
parser.add_argument('--data_source', type=str, default="MNIST", help="dataset to be used", choices=['MNIST'])
parser.add_argument('--sampling', type=str, default="iid", help="sampling technique for client data", choices=['iid', 'non_iid'])
parser.add_argument('--num_users', type=int, default=100, help="number of clients to create")
parser.add_argument('--num_shards_user', type=int, default=2, help="number of classes to give to the user")
parser.add_argument('--train_test_split', type=float, default=1.0, help="train test split at the client end")
parser.add_argument('--train_batch_size', type=int, default=32, help="batch size for client training")
parser.add_argument('--test_batch_size', type=int, default=32, help="batch size for testing data")

parser.add_argument('--model', type=str, default="MLP", help="network structure to be used for training", choices=['LR', 'MLP', 'CNN'])
parser.add_argument('--device', type=str, default="cpu", help="device for Torch", choices=['cpu', 'gpu'])
parser.add_argument('--frac_clients', type=float, default=0.1, help="proportion of clients to use for local updates")
parser.add_argument('--global_optimizer', type=str, default='fedavg', help="global optimizer to be used", choices=['fedavg', 'fedavgm', 'scaffold', 'fedadam', 'fedyogi'])
parser.add_argument('--global_epochs', type=int, default=100, help="number of global federated rounds")
parser.add_argument('--global_lr', type=float, default=1, help="learning rate for global steps")
parser.add_argument('--local_optimizer', type=str, default='sgd', help="local optimizer to be used", choices=['sgd', 'adam', 'pgd', 'scaffold'])
parser.add_argument('--local_epochs', type=int, default=20, help="number of local client training steps")
parser.add_argument('--local_lr', type=float, default=1e-4, help="learning rate for local updates")
parser.add_argument('--momentum', type=float, default=0.5, help="momentum value for SGD")
parser.add_argument('--mu', type=float, default=0.1, help="proximal coefficient for FedProx")
parser.add_argument('--beta1', type=float, default=0.9, help="parameter for FedAvgM and FedAdam")
parser.add_argument('--beta2', type=float, default=0.999, help="parameter for FedAdam")
parser.add_argument('--eps', type=float, default=1e-4, help="epsilon for adaptive methods")
parser.add_argument('--frac_byz_clients', type=float, default=0.0, help="proportion of clients that are picked in a round")
parser.add_argument('--is_attack', type=int, default=0, help="whether to attack or not")
parser.add_argument('--attack_type', type=str, default='label_flip', help="attack to be used", choices=['fall', 'label_flip', 'little', 'gaussian'])
parser.add_argument('--fall_eps', type=float, default=-5.0, help="epsilon value to be used for the Fall Attack")
parser.add_argument('--little_std', type=float, default=1.5, help="standard deviation to be used for the Little Attack")
parser.add_argument('--is_defense', type=int, default=0, help="whether to defend or not")
parser.add_argument('--defense_type', type=str, default='median', help="aggregation to be used", choices=['median', 'krum', 'trimmed_mean'])
parser.add_argument('--trim_ratio', type=float, default=0.1, help="proportion of updates to trim for trimmed mean")
parser.add_argument('--multi_krum', type=int, default=5, help="number of clients to pick after krumming")
parser.add_argument('--local_epochs_sampling', type=int, default=20, help="Number of local epochs without global aggregation (Phase2)")

parser.add_argument('--batch_print_frequency', type=int, default=100, help="frequency after which batch results need to be printed to the console")
parser.add_argument('--global_print_frequency', type=int, default=1, help="frequency after which global results need to be printed to the console")
parser.add_argument('--global_store_frequency', type=int, default=100, help="frequency after which global results should be written to CSV")
parser.add_argument('--threshold_test_metric', type=float, default=0.9, help="threshold after which the code should end")

obj = parser.parse_args()

with open('config.json') as f:
	json_vars = json.load(f)

obj = vars(obj)
obj.update(json_vars)
print(obj)

np.random.seed(obj['seed'])
torch.manual_seed(obj['seed'])

############################### Loading Dataset ###############################
if obj['data_source'] == 'MNIST':
	data_dir = 'data/'
	transformation = transforms.Compose([
		transforms.ToTensor(), 
		transforms.Normalize((0.1307,), (0.3081,))
	])
	train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transformation)
	test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transformation)
	print("Train and Test Sizes for %s - (%d, %d)"%(obj['data_source'], len(train_dataset), len(test_dataset)))
	
################################ Sampling Data ################################
if obj['sampling'] == 'iid':
	user_groups = iid(train_dataset, obj['num_users'], obj['seed'])
else:
	user_groups = non_iid(train_dataset, obj['num_users'], obj['num_shards_user'], obj['seed'])

################################ Defining Model ################################
if obj['model'] == 'LR':
	global_model = LR(dim_in=28*28, dim_out=10, seed=obj['seed'])
elif obj['model'] == 'MLP':
	global_model = MLP(dim_in=28*28, dim_hidden=200, dim_out=10, seed=obj['seed'])
elif obj['model'] == 'CNN' and obj['data_source'] == 'MNIST':
	global_model = CNNMnist(obj['seed'])
else:
	raise ValueError('Check the model and data source provided in the arguments.')

print("Number of parameters in %s - %d."%(obj['model'], network_parameters(global_model)))

global_model.to(obj['device'])
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

with open('results/%s_input.json'%(obj['exp_name']), 'w') as f:
	json.dump(obj, f, indent=4)

train_loss_updated = []
train_loss_all = []
test_loss = []
train_accuracy = []
test_accuracy = []
mus = [obj['mu'] for i in range(obj['num_users'])]

num_classes = 10 # MNIST

# Picking byzantine users (they would remain constant throughout the training procedure)
if obj['is_attack'] == 1:
	idxs_byz_users = np.random.choice(range(obj['num_users']), max(int(obj['frac_byz_clients']*obj['num_users']), 1), replace=False)
is_phase1=True
for epoch in range(obj['global_epochs']):

	################################# Client Sampling & Local Training #################################
	global_model.train()
	
	np.random.seed(epoch) # Picking a fraction of users to choose for training
	idxs_users = np.random.choice(range(obj['num_users']), max(int(obj['frac_clients']*obj['num_users']), 1), replace=False)
	
	local_updates, local_losses, local_sizes, control_updates = [], [], [], []

	for idx in idxs_users: # Training the local models

		if obj['is_attack'] == 1 and obj['attack_type'] == 'label_flip' and idx in idxs_byz_users:
			local_model = LocalUpdate(train_dataset, user_groups[idx], obj['device'], 
					obj['train_test_split'], obj['train_batch_size'], obj['test_batch_size'], obj['attack_type'], num_classes)
		else:
			local_model = LocalUpdate(train_dataset, user_groups[idx], obj['device'], 
					obj['train_test_split'], obj['train_batch_size'], obj['test_batch_size'])

		w, c_update, c_new, loss, local_size = local_model.local_opt(obj['local_optimizer'], obj['local_lr'], 
												obj['local_epochs'], global_model,is_phase1, obj['momentum'], mus[idx], c[idx], c[-1], 
												epoch+1, idx+1, obj['batch_print_frequency'])

		c[idx] = c_new # Updating the control variates in the main list for that client
		
		local_updates.append(copy.deepcopy(w))
		control_updates.append(c_update)
		local_losses.append(loss)
		local_sizes.append(local_size)
		#print(idx, np.unique(np.array([train_dataset.targets.numpy()[i] for i in user_groups[idx]])))

	train_loss_updated.append(sum(local_losses)/len(local_losses)) # Appending global training loss

	################################# Attack on the local weights #################################

	if obj['is_attack'] == 1:

		local_benign_updates = [i for idx, i in enumerate(local_updates) if idxs_users[idx] not in idxs_byz_users]
		local_byz_updates = [i for idx, i in enumerate(local_updates) if idxs_users[idx] in idxs_byz_users]
		local_byz_sizes = [i for idx, i in enumerate(local_sizes) if idxs_users[idx] in idxs_byz_users]

		if len(local_byz_updates) > 0:

			if obj['attack_type'] in ['fall', 'little']:

				byz_update, _, _ = attack_updates(global_weights, obj['defense_type'], obj['attack_type'], local_byz_updates, 
													local_byz_sizes, obj['little_std'], obj['fall_eps'])
				for i in range(len(local_byz_updates)):
					local_benign_updates.append(copy.deepcopy(byz_update)) # Setting same update for all byzantine workers

			elif obj['attack_type'] == 'gaussian':

				byz_update, m, s = attack_updates(global_weights, obj['defense_type'], obj['attack_type'], 
													local_byz_updates, local_byz_sizes, obj['little_std'], obj['fall_eps'])
				for i in range(len(local_byz_updates)):
					for k in byz_update.keys():
						local_byz_updates[i][k] = torch.normal(m[k], s[k])
					local_benign_updates.append(copy.deepcopy(local_byz_updates[i]))

			elif obj['attack_type'] == 'label_flip':
				pass

			else:
				raise ValueError("Please specify a valid attack_type from ['fall' ,'little', 'gaussian'].")

		local_updates = local_benign_updates

	#################################### Defense BEFORE Aggregation ####################################

	if obj['is_defense'] == 1:

		local_updates, local_sizes = defend_updates(local_updates, local_sizes, obj['defense_type'], obj['trim_ratio'], obj['multi_krum'])

	################################# Aggregation of the local weights #################################

	# Marginal Computation
	gw = copy.deepcopy(global_weights)
	# temp_g_weights1, v, m = global_aggregate(obj['global_optimizer'], gw, local_updates, 
	# 									local_sizes, obj['global_lr'], obj['beta1'], obj['beta2'],
	# 									v, m, obj['eps'], epoch+1)
	# global_model.load_state_dict(temp_g_weights1)
	# global_model.eval()
	# test_acc1, test_loss_value1 = test_inference(global_model, test_dataset, obj['device'], obj['test_batch_size'])

	# idxs_to_use = []
	# for idx in range(len(local_updates)):

	# 	curr_dist = [torch.norm(local_updates[idx][k]).numpy().reshape(-1, 1)[0][0] for k in gw.keys()]
	# 	#print(idxs_users[idx], curr_dist)

	# 	ee = mus[idxs_users[idx]]
	# 	mus[idxs_users[idx]] += (obj['mu'])*np.linalg.norm(curr_dist)
	# 	print("Mu for the client %d changed from %f to %f."%(idxs_users[idx], ee, mus[idxs_users[idx]]))

		# temp_g_weights2, v, m = global_aggregate(obj['global_optimizer'], gw, local_updates[:idx] + local_updates[idx+1:],
		# 								local_sizes, obj['global_lr'], obj['beta1'], obj['beta2'],
		# 								v, m, obj['eps'], epoch+1)
		# global_model.load_state_dict(temp_g_weights2)
		# global_model.eval()
		# test_acc2, test_loss_value2 = test_inference(global_model, test_dataset, obj['device'], obj['test_batch_size'])

		# if test_acc2 < test_acc1: # Without is less than with
		# 	idxs_to_use.append(idx)
		# else:
		# 	ee = mus[idxs_users[idx]]
		# 	mus[idxs_users[idx]] += (obj['mu']*10.0)*(test_acc2 - test_acc1)*100.0
		# 	print("Mu for the client %d changed from %f to %f."%(idxs_users[idx], ee, mus[idxs_users[idx]]))
		#print("Marginal with and without client %d - %.2f %% and %.2f %%."%(idxs_users[idx], test_acc1*100.0, test_acc2*100.0))

	global_model.load_state_dict(gw) # [i for idx, i in enumerate(local_updates) if idx in idxs_to_use]
	global_weights, v, m = global_aggregate(obj['global_optimizer'], global_weights, local_updates, 
										local_sizes, obj['global_lr'], obj['beta1'], obj['beta2'],
										v, m, obj['eps'], epoch+1)
	global_model.load_state_dict(global_weights)

	if obj['local_optimizer'] == 'scaffold': # Need to update the server control variate
		for key in c[-1].keys():
			for i in control_updates:
				c[-1][key] += torch.div(i[key], len(user_groups))

	######################################### Model Evaluation #########################################
	global_model.eval()
	
	if obj['train_test_split'] != 1.0:
		list_acc = []
		list_loss = []
		for idx in range(obj['num_users']):

			local_model = LocalUpdate(train_dataset, user_groups[idx], obj['device'], obj['train_test_split'], 
								  obj['train_batch_size'], obj['test_batch_size'])
			acc, loss = local_model.inference(global_model)
			list_acc.append(acc)
			list_loss.append(loss)

		train_loss_all.append(sum(list_loss)/len(list_loss))
		train_accuracy.append(sum(list_acc)/len(list_acc))
	
	# Evaluation on the hold-out test set at central server
	test_acc, test_loss_value = test_inference(global_model, test_dataset, obj['device'], obj['test_batch_size'])
	test_accuracy.append(test_acc)
	test_loss.append(test_loss_value)

	if (epoch+1) % obj['global_print_frequency'] == 0 or (epoch+1) == obj['global_epochs']:
		msg = '| Global Round : {0:>4} | TeLoss - {1:>6.4f}, TeAcc - {2:>6.2f} %, TrLoss (U) - {3:>6.4f}'

		if obj['train_test_split'] != 1.0:
			msg = 'TrLoss (A) - {4:>6.4f} % , TrAcc - {5:>6.2f} %'
			print(msg.format(epoch+1, test_loss[-1], test_accuracy[-1]*100.0, train_loss_updated[-1], 
							train_loss_all[-1], train_accuracy[-1]*100.0))
		else:
			print(msg.format(epoch+1, test_loss[-1], test_accuracy[-1]*100.0, train_loss_updated[-1]))

	if (epoch+1) % obj['global_store_frequency'] == 0  or (epoch+1) == obj['global_epochs'] or test_accuracy[-1] >= obj['threshold_test_metric']:
		if obj['train_test_split'] != 1.0:
			out_arr = pd.DataFrame(np.array([list(range(epoch+1)), train_accuracy, test_accuracy, train_loss_updated, train_loss_all, test_loss]).T,
								columns=['epoch', 'train_acc', 'test_acc', 'train_loss_updated', 'train_loss_all', 'test_loss'])
		else:
			out_arr = pd.DataFrame(np.array([list(range(epoch+1)), test_accuracy, train_loss_updated, test_loss]).T,
				columns=['epoch', 'test_acc', 'train_loss_updated', 'test_loss'])
		out_arr.to_csv('results/%s_output.csv'%(obj['exp_name']), index=False)

	if test_accuracy[-1] >= obj['threshold_test_metric']:
		print("Terminating as desired threshold for test metric reached...")
		break

 
is_phase1=False
#for idx in idxs_users: # Training the local models
final_updates=[]
for idx in idxs_users: # Training the local models

	if obj['is_attack'] == 1 and obj['attack_type'] == 'label_flip' and idx in idxs_byz_users:
		local_model = LocalUpdate(train_dataset, user_groups[idx], obj['device'], 
				obj['train_test_split'], obj['train_batch_size'], obj['test_batch_size'], obj['attack_type'], num_classes)
	else:
		local_model = LocalUpdate(train_dataset, user_groups[idx], obj['device'], 
				obj['train_test_split'], obj['train_batch_size'], obj['test_batch_size'])

		w, c_update, c_new, loss, local_size = local_model.local_opt(obj['local_optimizer'], obj['local_lr'], 
												obj['local_epochs_sampling'], global_model,is_phase1, obj['momentum'], mus[idx], c[idx], c[-1], 
												epoch+1, idx+1, obj['batch_print_frequency'])
	final_updates.append(copy.deepcopy(w))
w_final = OrderedDict()
mean = OrderedDict()
std = OrderedDict()

for k in global_weights.keys():
	w_final[k] = torch.zeros(global_weights[k].shape, dtype=global_weights[k].dtype)
	mean[k] = torch.zeros(global_weights[k].shape, dtype=global_weights[k].dtype)
	std[k] = torch.zeros(global_weights[k].shape, dtype=global_weights[k].dtype)
total_size=sum(local_sizes)
print('will start calculating mean and variance')
for k in global_weights.keys():
	for i in range(len(final_updates)):
		a = torch.mul(final_updates[i][k], local_sizes[i]/total_size)
		mean[k] += a
		std[k] += torch.mul(a, a)
		std[k]= (torch.abs(std[k] - torch.mul(mean[k], mean[k]))) ** 0.5
print('done!!')
#	for epoch in range(obj['local_epoch'])
#		if obj['is_attack'] == 1 and obj['attack_type'] == 'label_flip' and idx in idxs_byz_users:
#			local_model = LocalUpdate(train_dataset, user_groups[idx], obj['device'], 
#					obj['train_test_split'], obj['train_batch_size'], obj['test_batch_size'], obj['attack_type'], num_classes)
#		else:
#			local_model = LocalUpdate(train_dataset, user_groups[idx], obj['device'], 
#					obj['train_test_split'], obj['train_batch_size'], obj['test_batch_size'])
#
#		w, c_update, c_new, loss, local_size = local_model.local_opt(obj['local_optimizer'], obj['local_lr'], 
#												epoch+1, idx+1, obj['batch_print_frequency'])
#
#		c[idx] = c_new # Updating the control variates in the main list for that client
##	local_updates.append(copy.deepcopy(w))
#	control_updates.append(c_update)
#	local_losses.append(loss)
##		#print(idx, np.unique(np.array([train_dataset.targets.numpy()[i] for i in user_groups[idx]])))
#
#	train_loss_updated.append(sum(local_losses)/len(local_losses)) # Appending global training loss