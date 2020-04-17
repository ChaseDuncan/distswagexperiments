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
parser = argparse.ArgumentParser(description='D-SWAG experiemnts training script.') 
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
parser.add_argument('--data_path', type=str, default='/shared/mrfil-data/cddunca2/cifar10/', metavar='PATH',
args = parser.parse_args()

N = 30
num_classes = 10
var_clamp = 1e-6 
eps = 1e-16 
checkpoints = []
num_checkpoints = None
#num_checkpoints = 5
save_path = f'{args}/evaluation/'


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
for root, dirs, filenames in os.walk(f'/shared/mrfil-data/cddunca2/pgmproject/{model_name}'):
  for name in filenames:
    checkpoints.extend([os.path.join(root, name)])

assert len(checkpoints) > 0

model = models.vgg16(num_classes=num_classes)
mean = torch.zeros(sum(param.numel() for param in model.parameters()))
sq_mean = mean.clone() 

# compute moments
for n, checkpoint in enumerate(checkpoints):
  print(f'Loading: {n} of {len(checkpoints)}')
  chkpt = torch.load(checkpoint)
  model.load_state_dict(chkpt['state_dict'])
  mean, sq_mean = collect_model(model, mean, sq_mean, n)

predictions = np.zeros((len(cifar10_test_loader.dataset), num_classes))
targets = np.zeros(len(cifar10_test_loader.dataset))

accuracy_per_epoch = []
for i in range(N):
  print("%d/%d" % (i + 1, N))
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
  
  accuracy =  np.mean(np.argmax(predictions, axis=1) == targets)
  accuracy_per_epoch.append(accuracy)
  print("Accuracy:", accuracy)
  #nll is sum over entire dataset
  print("NLL:", nll(predictions / (i + 1), targets))

predictions /= N
accuracy_per_epoch = np.array(accuracy_per_epoch)
entropies = -np.sum(np.log(predictions + eps) * predictions, axis=1)
np.savez(save_path, entropies=entropies, predictions=predictions, 
    targets=targets, accuracy_per_epoch=accuracy_per_epoch)
