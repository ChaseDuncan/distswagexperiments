import os
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
num_classes = 10
eps = 1e-32
#model_name = 'baseline'
#model_name = 'dist-bayesian'
model_name = '1worker'
checkpoints = []
cifar10_path = '/shared/mrfil-data/cddunca2/cifar10/'
save_path = 'results/'+model_name

transform_test = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )

cifar10_test_dataset = datasets.CIFAR10(cifar10_path, 
                      train=False, download=True, transform=transform_test)

cifar10_test_loader = DataLoader(cifar10_test_dataset, shuffle=False, batch_size=128)


# Gather model checkpoints
for root, dirs, filenames in os.walk('checkpoints/'+ model_name):
  for name in filenames:
    checkpoints.extend([os.path.join(root, name)])

model = models.vgg16(num_classes=num_classes)

predictions = np.zeros((len(cifar10_test_loader.dataset), num_classes))
targets = np.zeros(len(cifar10_test_loader.dataset))

accuracy_per_epoch = []
N = len(checkpoints)
for i in range(N):
  print("%d/%d" % (i + 1, N))
  k = 0 

  chkpt = torch.load(checkpoints[i])

  model.load_state_dict(chkpt['state_dict'])
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
  
  accuracy =  np.mean(np.argmax(predictions, axis=1) == targets)
  accuracy_per_epoch.append(accuracy)
  print(f'Model: {checkpoints[i]}')
  print("Accuracy:", accuracy)
  #nll is sum over entire dataset
  print("NLL:", nll(predictions / (i + 1), targets))

predictions /= N
accuracy_per_epoch = np.array(accuracy_per_epoch)
entropies = -np.sum(np.log(predictions + eps) * predictions, axis=1)
