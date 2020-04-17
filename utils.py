import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

def moving_average(net1, net2, alpha=1):
  for param1, param2 in zip(net1.parameters(), net2.parameters()):
    param1.data *= (1.0 - alpha)
    param1.data += param2.data * alpha


def flatten(lst):
  tmp = [i.contiguous().view(-1, 1) for i in lst]
  return torch.cat(tmp).view(-1)


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

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


def set_weights(model, vector, device=None):
  param_list = [ param for param in model.parameters()]
  sample_list = unflatten_like(vector, param_list)
  for param, sample in zip(model.parameters(), sample_list):
    param.data = sample.data

def train(model, split, dataset, args, model_name='model', ckpt_dir='.'):
  train_loss = []
  train_eval = []
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
  
    train_loss.append(total_loss)

    model.eval()
    k=0 
    for input, target in cifar10_test_loader:
      input = input.to(device)
      output = model(input)
      with torch.no_grad():
        predictions[k : k + input.size()[0]] += (
          F.softmax(output, dim=1).cpu().numpy()
        )
      targets[k : (k + target.size(0))] = target.numpy()
      k += input.size()[0]    
    
    accuracy =  np.mean(np.argmax(predictions, axis=1) == targets)
    train_eval.append(accuracy)
    print("Accuracy:", accuracy)
    #nll is sum over entire dataset
    print("NLL:", nll(predictions / (i + 1), targets))

    if sample:
      utils.save_checkpoint(ckpt_dir, epoch=epoch, name=model_name+f'{epoch:03}',
                            state_dict=model.state_dict(), optimizer=optimizer.state_dict())
    return train_loss, train_eval


def nll(outputs, labels):
  labels = labels.astype(int)
  idx = (np.arange(labels.size), labels)
  ps = outputs[idx]
  nll = -np.sum(np.log(ps))
  return nll

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


def sample_post(mean, sq_mean, var_clamp=1e-6):
  variance = torch.clamp(sq_mean - mean ** 2, var_clamp)
  var_sample = variance.sqrt()*torch.randn_like(variance, requires_grad=False)
  sample = mean + var_sample
  sample = sample.unsqueeze(0)
  return sample


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
