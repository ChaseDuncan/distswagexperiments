import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as transforms


class LR(nn.Module):

	"""
	Logistic Regression Model. (Convex objective)
	"""

	def __init__(self, dim_in, dim_out, seed):
		"""
		Args:
			dim_in (int) : Input dimension
			dim_out (int) : Output dimension
			seed (int) : Random seed value
		"""

		super(LR, self).__init__()

		torch.manual_seed(seed)

		self.dim_in = dim_in
		self.linear = nn.Linear(dim_in, dim_out)

	def forward(self, x):

		x = x.view(-1, self.dim_in) # Flattening the input
		x = self.linear(x)

		return F.log_softmax(x, dim=1)
	
class MLP(nn.Module):

	"""
	Multi Layer Perceptron with a single hidden layer.
	"""
	
	def __init__(self, dim_in, dim_hidden, dim_out, seed):
		"""
		Args:
			dim_in (int) : Input dimension
			dim_hidden (int) : # units in the hidden layer
			dim_out (int) : Output dimension
			seed (int) : Random seed value
		"""
		
		super(MLP, self).__init__()
		
		torch.manual_seed(seed)
		
		self.input = nn.Linear(dim_in, dim_hidden)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout()
		self.layer_hidden = nn.Linear(dim_hidden, dim_hidden)
		self.output = nn.Linear(dim_hidden, dim_out)

	def forward(self, x):
		
		x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
		x = self.input(x)
		x = self.relu(x)
		x = self.layer_hidden(x)
		x = self.relu(x)
		x = self.output(x)

		return F.log_softmax(x, dim=1)
	
class CNNMnist(nn.Module):

	"""
	2-layer CNN as used in (http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf).
		
	Note: TF code doesn't use dropout. (https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/emnist_fedavg_main.py#L82)
	"""
	
	def __init__(self, seed):
		"""
		Args:
			seed (int) : Random seed value
		"""
		
		super(CNNMnist, self).__init__()
		
		torch.manual_seed(seed)
		
		self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
		self.fc1 = nn.Linear(7*7*64, 512)
		self.fc2 = nn.Linear(512, 10)

	def forward(self, x):
		
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2)
		x = x.view(-1, 7*7*64)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		
		return F.log_softmax(x, dim=1)


"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""


__all__ = ["VGG16", "VGG16BN", "VGG19", "VGG19BN"]


def make_layers(cfg, batch_norm=False):
    layers = list()
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    16: [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    19: [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(self, num_classes=10, depth=16, batch_norm=False):
        super(VGG, self).__init__()
        self.features = make_layers(cfg[depth], batch_norm)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Base:
    base = VGG
    args = list()
    kwargs = dict()
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            # transforms.Normalize((0.4376821 , 0.4437697 , 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # transforms.Normalize((0.45242316, 0.45249584, 0.46897713), (0.21943445, 0.22656967, 0.22850613))
        ]
    )


class VGG16(Base):
    pass


class VGG16BN(Base):
    kwargs = {"batch_norm": True}


class VGG19(Base):
    kwargs = {"depth": 19}


class VGG19BN(Base):
    kwargs = {"depth": 19, "batch_norm": True}