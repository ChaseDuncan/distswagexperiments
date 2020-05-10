import copy
import math
from collections import OrderedDict
import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
def global_aggregate(global_optimizer, global_weights, local_updates, local_sizes, 
					global_lr=1., beta1=0.9, beta2=0.999, v=None, m=None, eps=1e-4, step=None):
	"""
	Aggregates the local client updates to find a focused global update.

	Args:
		global_optimizer (str) : Optimizer to be used for the steps
		global_weights (OrderedDict) : Initial state of the global model (which needs to be updated here)
		local_updates (list of OrderedDict) : Contains the update differences (delta) between global and local models
		local_sizes (list) : Sizes of local datasets for proper normalization
		global_lr (float) : Stepsize for global step
		beta1 (float) : Role of ``beta`` in FedAvgM, otheriwse analogous to beta_1 and beta_2 famous in literature for Adaptive methods
		beta2 (float) : Same as above
		v (OrderedDict) : Role of ``momentum`` in FedAvgM, else Adaptive methods
		m (OrderedDict) : Common in ADAM and YOGI.
		step (int) : Current epoch number to configure ADAM and YOGI properly
	"""
	
	total_size = sum(local_sizes)

	################################ FedAvg | SCAFFOLD ################################
	# Good illustration provided in SCAFFOLD paper - Equations (1). (https://arxiv.org/pdf/1910.06378.pdf)
	if global_optimizer in ['fedavg', 'scaffold']:
		
		w = copy.deepcopy(global_weights)
		for k in global_weights.keys():
			w[k] = torch.zeros(global_weights[k].shape, dtype=global_weights[k].dtype)

		for key in w.keys():
			for i in range(len(local_updates)):
				if global_optimizer == 'scaffold':
					w[key] += torch.mul(torch.div(local_updates[i][key], len(local_sizes)), global_lr)
				else:
					w[key] += torch.mul(local_updates[i][key].to('cpu'), local_sizes[i]/total_size)

		return w, v, m
	
	################################ FedAvgM ################################
	# Implementation similar to in (https://arxiv.org/pdf/1909.06335.pdf).
	elif global_optimizer == 'fedavgm':
		
		w = copy.deepcopy(global_weights)
		temp_v = copy.deepcopy(v)
		
		for key in w.keys():
			temp_v[key] = beta1*temp_v[key]
			for i in range(len(local_updates)):
				temp_v[key] -= torch.mul(torch.mul(local_updates[i][key], local_sizes[i]/total_size), global_lr)
			w[key] -= temp_v[key]
			
		return w, temp_v, m

	################################ FedAdam ################################
	# Adam from here : https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization.pdf
	elif global_optimizer in ['fedadam', 'fedyogi']:

		w = copy.deepcopy(global_weights)
		temp_v = copy.deepcopy(v)
		temp_m = copy.deepcopy(m)
		effective_lr = global_lr*math.sqrt(1 - beta2**step)/(1 - beta1**step)

		averaged_w = OrderedDict()
		for key in w.keys():
			averaged_w[key] = torch.zeros(w[key].shape, dtype=w[key].dtype)
			for i in range(len(local_updates)):
				averaged_w[key] += torch.mul(local_updates[i][key], local_sizes[i]/total_size)

		for key in w.keys():
			temp_m[key] = beta1*temp_m[key] + (1. - beta1)*averaged_w[key]

			if global_optimizer == 'fedadam':
				temp_v[key] = temp_v[key] - (1 - beta2)*(temp_v[key] - torch.pow(averaged_w[key], 2))
			else: #FedYogi
				temp_v[key] = temp_v[key] - (1 - beta2)*torch.mul(torch.sign(temp_v[key] - torch.pow(averaged_w[key], 2)), 
																			torch.pow(averaged_w[key], 2))

			w[key] += torch.mul(effective_lr, torch.div(temp_m[key], torch.add(eps, torch.pow(temp_v[key], 0.5))))

		return w, temp_v, temp_m

	else:

		raise ValueError('Check the global optimizer for a valid value.')
	
def network_parameters(model):
	"""
	Calculates the number of parameters in the model.

	Args:
		model : PyTorch model used after intial weight initialization
	"""
	total_params = 0
	
	for param in list(model.parameters()):
		curr_params = 1
		for p in list(param.size()):
			curr_params *= p
		total_params += curr_params
		
	return total_params

class DatasetSplit(Dataset):
	"""
	An abstract dataset class wrapped around Pytorch Dataset class.
	"""

	def __init__(self, dataset, idxs):

		self.dataset = dataset
		self.idxs = [int(i) for i in idxs]

	def __len__(self):

		return len(self.idxs)

	def __getitem__(self, item):
		
		image, label = self.dataset[self.idxs[item]]

		return torch.tensor(image), torch.tensor(label)

def test_inference(global_model, test_dataset, device, criterion, test_batch_size=128):
	"""
	Evaluates the performance of the global model on hold-out dataset.

	Args:
		global_model (model state) : Global model for evaluation
		test_dataset (tensor) : Hold-out data available at the server
		device (str) : One from ['cpu', 'cuda'].
		test_batch_size (int) : Batch size of the testing samples
	"""

	test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
	#criterion = nn.NLLLoss().to(device)

	global_model.eval()

	loss, total, correct = 0.0, 0.0, 0.0

	for batch_idx, (images, labels) in enumerate(test_loader):

		images, labels = images.to(device), labels.to(device)

		outputs = global_model(images)
		batch_loss = criterion(outputs, labels)
		loss += batch_loss.item()

		# Prediction
		_, pred_labels = torch.max(outputs, 1)
		pred_labels = pred_labels.view(-1)
		correct += torch.sum(torch.eq(pred_labels, labels)).item()
		total += len(labels)
	
	return correct/total, loss/total


def predict(loader, model, device, verbose=False):
    predictions = list()
    targets = list()

    model.eval()

    if verbose:
        loader = tqdm.tqdm(loader)

    offset = 0
    with torch.no_grad():
        for input, target in loader:
            #input = input.cuda(non_blocking=True)
            output = model(input.to(device))

            batch_size = input.size(0)
            predictions.append(F.softmax(output, dim=1).cpu().numpy())
            targets.append(target.numpy())
            offset += batch_size

    return {"predictions": np.vstack(predictions), "targets": np.concatenate(targets)}



def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def apply(self, *args, **kwargs):
        self.net.apply(*args, **kwargs)




def bn_update(loader, model, verbose=False, subset=None, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)
        if verbose:

            loader = tqdm.tqdm(loader, total=num_batches)
        for input, _ in loader:
            input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))

import argparse

def pop_irrelevant_args(parser):
  parser.add_argument('--seed', type=int, default=0, help="seed for running the experiments")
  parser.add_argument('--sampling', type=str, default="iid", help="sampling technique for client data", choices=['iid', 'non_iid'])
  parser.add_argument('--num_shards_user', type=int, default=2, help="number of classes to give to the user")
  parser.add_argument('--train_test_split', type=float, default=1.0, help="train test split at the client end")
  parser.add_argument('--train_batch_size', type=int, default=32, help="batch size for client training")
  parser.add_argument('--test_batch_size', type=int, default=32, help="batch size for testing data")
  parser.add_argument('--local_epochs_sampling', type=int, default=20, help="Number of local epochs without global aggregation (Phase2)")
  parser.add_argument('--rank_param', type=int, default=4, help="Low rank approxmation parameter")
  parser.add_argument('--num_samples', type=int, default=20, help="Number of samples in testing phase")
  parser.add_argument('--frac_users_phase2', type=float, default=1.0, help="Number of clients to run phase 2")
  parser.add_argument('--device', type=str, default="gpu", help="device for Torch", choices=['cpu', 'gpu'])
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

