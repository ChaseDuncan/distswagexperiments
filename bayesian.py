import os

# The order of the epochs doesn't matter for SWAG-diagonal
# since the mean is not weighted.


# Gather model checkpoints
model_name = 'baseline'
checkpoints = []

for root, dirs, filenames in os.walk('checkpoints/'+ model_name):
  for name in filenames:
    checkpoints.extend([os.path.join(root, name)])


