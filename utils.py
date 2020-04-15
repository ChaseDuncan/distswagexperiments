import os
import torch

def moving_average(net1, net2, alpha=1):
  for param1, param2 in zip(net1.parameters(), net2.parameters()):
    param1.data *= (1.0 - alpha)
    param1.data += param2.data * alpha

def swa_moment1(moment1, obs, n):
  for param1, param2 in zip(moment1.parameters(), obs.parameters()):
    param1.data *= n
    param1.data += param2.data 
  param1.data /= 1. / (n + 1)

def swa_moment2(moment2, net2, n):
  for param1, param2 in zip(moment2.parameters(), net2.parameters()):
    param1.data *= n
    param1.data += param2.data * param2.data
  param1.data /= 1. / (n + 1)


def flatten(lst):
  tmp = [i.contiguous().view(-1, 1) for i in lst]
  return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
  # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
  #    shaped like likeTensorList
  outList = []
  i = 0
  for tensor in likeTensorList:
    # n = module._parameters[name].numel()
    n = tensor.numel()
    outList.append(vector[:, i : i + n].view(tensor.shape))
    i += n
  return outList

def save_checkpoint(dir, epoch=None, name='checkpoint', **kwargs):
  state = {
      'epoch': epoch,
      }
  if epoch is not None:
    name = '%s-%d.pt' % (name, epoch)
  else:
    name = '%s.pt' % (name)
    state.update(kwargs)
    filepath = os.path.join(dir, name)
    torch.save(state, filepath)


def adjust_learning_rate(optimizer, lr):
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
    return lr


def collect_model(model, mean, sq_mean, n):
  w = flatten([param.detach().cpu() for param in model.parameters()])
  # first moment
  mean.mul_(n / (n + 1.0))
  mean.add_(w / (n + 1.0))

  # second moment
  sq_mean.mul_(n / (n + 1.0))
  sq_mean.add_(w ** 2 / (n + 1.0))
  return mean, sq_mean

# args.swa_start <- th to begin swa sampling of parameters
# args.swa_lr <- the constant learning rate at which the swa sampling is performed.
# args.lr_init <- the learning rate for the constant lr period
def schedule(epoch, swa_start, swa_lr, lr_init):
  t = epoch / swa_start
  lr_ratio = swa_lr / lr_init
  # TODO: replace this with something more sophisticated?
  if t <= 0.5:
    factor = 1.0
  elif t <= 0.9:
    factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
  else:
    factor = lr_ratio
  return lr_init * factor

#def train_epoch(loader, model, criterion, optimizer):
#    loss_sum = 0.0
#    correct = 0.0
#
#    model.train()
#
#    for i, (input, target) in enumerate(loader):
#        input = input.cuda(async=True)
#        target = target.cuda(async=True)
#        input_var = torch.autograd.Variable(input)
#        target_var = torch.autograd.Variable(target)
#
#        output = model(input_var)
#        loss = criterion(output, target_var)
#
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#
#        loss_sum += loss.item() * input.size(0)
#        pred = output.data.max(1, keepdim=True)[1]
#        correct += pred.eq(target_var.data.view_as(pred)).sum().item()
#
#    return {
#        'loss': loss_sum / len(loader.dataset),
#        'accuracy': correct / len(loader.dataset) * 100.0,
#    }
