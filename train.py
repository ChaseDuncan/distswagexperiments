# python modules
import os

# random modules
from tqdm import tqdm

# pytorch modules
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import utils
from utils import *
# Directory to retrieve or store CIFAR10 dataset. This directory will
# be created if it doesn't exist already.
model_name = 'baseline'
ckpt_dir = 'checkpoints/'+ model_name + '/'
cifar10_path = '/shared/mrfil-data/cddunca2/cifar10/'
epochs = 150
swa_start = 125
swa_lr = 0.05
lr_init = 0.1
momentum = 0.0

verbose = True

lr_decay = lr_init / epochs

device = torch.device(f'cuda:{str(get_free_gpu())}')
# Create data directories as necessary.
if not os.path.exists(ckpt_dir):
  print('[INFO] Make dir %s' % ckpt_dir)
  os.makedirs(ckpt_dir)

if not os.path.exists(cifar10_path):
  print('[INFO] Make dir %s' % cifar10_path)
  os.makedirs(cifar10_path)

# Taken from swag experiments
# https://github.com/wjmaddox/drbayes/blob/master/subspace_inference/models/vgg.py
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

cifar10_train_dataset = datasets.CIFAR10(cifar10_path, 
                      train=True, download=True, transform=transform_train)

cifar10_train_loader = DataLoader(cifar10_train_dataset, shuffle=True, batch_size=128)

model = models.vgg16(num_classes=10)
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=lr_init, 
    momentum=momentum, weight_decay=1e-5)

# time-based lr decay
#lr_lambda = lambda epoch: (1. / lr_decay*epoch)

# SWA lr dacay
lr_lambda = lambda epoch: utils.schedule(epoch, swa_start, swa_lr, lr_init)
# learning rate scheduler
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
# L2 regularized cross entropy loss
loss = torch.nn.CrossEntropyLoss()

sample = False
for epoch in range(epochs):
  total_loss = 0
  model.train()

  if epoch >= swa_start:
    sample = True 

  if verbose:
    cifar10_train_loader = tqdm(cifar10_train_loader)

  for src, target in cifar10_train_loader:
    optimizer.zero_grad()
    src, target = src.to(device), target.to(device)
    output = model(src)
    cur_loss = loss(output, target)
    total_loss += cur_loss
    cur_loss.backward()
    optimizer.step()
  if sample:
    utils.save_checkpoint(ckpt_dir, epoch=epoch, name=model_name+f'{epoch:03}',
                          state_dict=model.state_dict(), optimizer=optimizer.state_dict())
    
