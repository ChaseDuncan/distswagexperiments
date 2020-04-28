import os
import argparse

import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np

from utils import *
from tqdm import tqdm

device = torch.device(f'cuda:{str(get_free_gpu())}')
# The order of the epochs doesn't matter for SWAG-diagonal
# since the mean is not weighted.
parser = argparse.ArgumentParser(description='D-SWAG experiemnts training script.') 
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
parser.add_argument('--data_path', type=str, default='/shared/mrfil-data/cddunca2/cifar10/', metavar='PATH',
                    help='path to datasets location (default: /shared/mrfil-data/cddunca2/cifar10/)')
args = parser.parse_args()

N = 30
num_classes = 10
var_clamp = 1e-6 
eps = 1e-16 
checkpoints = []
save_path = f'{args.dir}evaluation'


transform_test = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )

cifar10_test_dataset = datasets.CIFAR10(args.data_path, 
                      train=False, download=True, transform=transform_test)

cifar10_test_loader = DataLoader(cifar10_test_dataset, shuffle=False, batch_size=128)


# Gather model checkpoints
for root, dirs, filenames in os.walk(f'{args.dir}/checkpoints/'):
  if filenames:
    worker = []
    for name in filenames:
      worker.extend([os.path.join(root, name)])
    checkpoints.append(worker)

assert len(checkpoints) > 0

model = models.vgg16(num_classes=num_classes)
mean = torch.zeros(sum(param.numel() for param in model.parameters()))
sq_mean = mean.clone() 

# compute moments
posterior_by_epoch = []
for n, checkpoint in enumerate(zip(*checkpoints)):
  print(f'Loading: {n} of {len(checkpoints[0])}')
  chkpts = [torch.load(ckpt) for ckpt in checkpoint]
  for chkpt in chkpts:
    model.load_state_dict(chkpt['state_dict'])
    mean, sq_mean = collect_model(model, mean, sq_mean, n)
  posterior_by_epoch.append((mean, sq_mean))


accuracy_per_epoch = []
nll_per_epoch = []
for ep, (mean, sq_mean) in enumerate(posterior_by_epoch):
  predictions = np.zeros((len(cifar10_test_loader.dataset), num_classes))
  targets = np.zeros(len(cifar10_test_loader.dataset))
  for i in range(N):
    print("Epoch: %d, %d/%d" % (ep, i + 1, N))
    sample = sample_post(mean, sq_mean)  
    set_weights(model, sample, device) 
    
    k = 0 
    # Have to move data to device after setting the weights
    model.to(device)
    model.eval()
    for input, target in tqdm(cifar10_test_loader):
      input = input.to(device)
      output = model(input)
      with torch.no_grad():
        predictions[k : k + input.size()[0]] += (
          F.softmax(output, dim=1).cpu().numpy()
        )
      targets[k : (k + target.size(0))] = target.numpy()
      k += input.size()[0]    
   
  accuracy =  np.mean(np.argmax(predictions, axis=1)  == targets)
  accuracy_per_epoch.append(accuracy)
  print("Accuracy:", accuracy)
  #nll is sum over entire dataset
  nll_eval = nll(predictions / N, targets)
  print("NLL:", nll_eval)
  nll_per_epoch.append(nll_eval)
  print(accuracy_per_epoch)
  print(nll_per_epoch)

predictions /= N
accuracy_per_epoch = np.array(accuracy_per_epoch)
entropies = -np.sum(np.log(predictions + eps) * predictions, axis=1)
np.savez(save_path, entropies=entropies, predictions=predictions, 
    targets=targets, accuracy_per_epoch=accuracy_per_epoch,
    nll_per_epoch=nll_per_epoch)
