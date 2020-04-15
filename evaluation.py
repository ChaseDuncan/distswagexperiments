import os
import torch
import torchvision.models as models

from utils import collect_model

# The order of the epochs doesn't matter for SWAG-diagonal
# since the mean is not weighted.

var_clamp = 1e-6 
model_name = 'baseline'
checkpoints = []

# Gather model checkpoints
for root, dirs, filenames in os.walk('checkpoints/'+ model_name):
  for name in filenames:
    checkpoints.extend([os.path.join(root, name)])

model = models.vgg16()
mean = torch.zeros(sum(param.numel() for param in model.parameters()))
sq_mean = mean.clone() 

# compute moments
for n, checkpoint in enumerate(checkpoints):
  chkpt = torch.load(checkpoints[1])
  model.load_state_dict(chkpt['state_dict'])
  mean, sq_mean = collect_model(model, mean, sq_mean, n)

variance = torch.clamp(sq_mean - mean ** 2, var_clamp)

