# python modules
import os
import copy
import argparse
import sys

# random modules
from tqdm import tqdm

# pytorch modules
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import utils
from utils import *
# Directory to retrieve or store CIFAR10 dataset. This directory will
# be created if it doesn't exist already.

parser = argparse.ArgumentParser(description='D-SWAG experiemnts') 
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
parser.add_argument('--data_path', type=str, default='/shared/mrfil-data/cddunca2/cifar10/', metavar='PATH',
                    help='path to datasets location (default: /shared/mrfil-data/cddunca2/cifar10/)')
parser.add_argument('--num_workers', type=int, default=1, metavar='N', help='number of workers (default: 1)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--epochs', type=int, default=150, metavar='N', help='number of epochs (default: 150)')
parser.add_argument('--swa_start', type=int, default=125, metavar='N', help='epoch at which to begin phase 2(default: 125)')
parser.add_argument('--swa_lr', type=float, default=0.05, metavar='learning rate', help='Learning rate to use in phase 2 (default: 0.05)')
parser.add_argument('--lr_init', type=float, default=0.05, metavar='learning rate', help='Initial learning rate (default: 0.1)')
parser.add_argument('--weight_decay', type=float, default=1e-5, metavar='weight decay rate', help='Initial learning rate (default: 0.1e-5)')
parser.add_argument('--momentum', type=float, default=0.0, metavar='momentum', help='Momentum for SGD (default: 0.0)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')
args = parser.parse_args()

print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

# TODO: figure out how to use momentum in the scheduler
verbose = True

lr_decay = args.lr_init / args.epochs
device = torch.device(f'cuda:{str(get_free_gpu())}')

# Create data directories as necessary.
if not os.path.exists(args.data_path):
  print('[INFO] Make dir %s' % args.data_path)
  os.makedirs(args.data_path)

# Taken from swag experiments
# https://github.com/wjmaddox/drbayes/blob/master/subspace_inference/models/vgg.py
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

cifar10_train_dataset = datasets.CIFAR10(args.data_path, 
                      train=True, download=True, transform=transform_train)
lengths = [len(cifar10_train_dataset) // args.num_workers] * args.num_workers
# if there's a difference in the sum of the lengths of the split
# and the length of the dataset, add it to the last length.
lengths[-1]+=len(cifar10_train_dataset)-sum(lengths)
cifar10_train_dataset_split = random_split(cifar10_train_dataset, lengths)

# SWA lr dacay
lr_lambda = lambda epoch: utils.schedule(epoch, args.swa_start, args.swa_lr, args.lr_init)

def train(model, split, dataset, model_name='model', ckpt_dir='.'):
  cifar10_train_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)

  optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, 
      momentum=args.momentum, weight_decay=args.weight_decay)

  # learning rate scheduler
  scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

  # L2 regularized cross entropy loss
  loss = torch.nn.CrossEntropyLoss()

  sample = False
  for epoch in range(args.epochs):
    total_loss = 0
    model.train()

    if epoch >= args.swa_start:
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
    
base_model = models.vgg16(num_classes=10)

# Train a model for each split of data. Not sure how parameter initialization affects things.
for split, cifar10_train_dataset in enumerate(cifar10_train_dataset_split):
  model = copy.deepcopy(base_model)
  model = model.to(device)
  model_name = f'{split:03}'
  ckpt_dir = f'{args.dir}/{model_name}/'


  if not os.path.exists(ckpt_dir):
    print('[INFO] Make dir %s' % ckpt_dir)
    os.makedirs(ckpt_dir)
 
  train(model, split, cifar10_train_dataset, model_name, ckpt_dir)

