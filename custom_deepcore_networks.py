# this file needs to be placed in DeepCore/deepcore/nets/
from torch import nn
from torch import set_grad_enabled, flatten

from models.convolutional_network import MnistCnn, Cifar10Cnn9l
from torchvision.models import resnet18
from .nets_utils import EmbeddingRecorder

class MnistCnnAdapter(nn.Module):
    def __init__(self, channel, n_classes, im_size, pretrained=False, record_embedding=False, no_grad=False):
        super().__init__()
        self.model = MnistCnn(n_classes=n_classes, dropout_probability=0)
        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    def get_last_layer(self):
        return self.model.linear_layers[-1]

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            out = self.model.cnn_layers(x)
            out = self.model.linear_layers[:-1]((out.flatten(1)))
            out = self.embedding_recorder(out)
            out = self.model.linear_layers[-1](out)
        return out
    
class Cifar10CnnAdapter(nn.Module):
    def __init__(self, channel, n_classes, im_size, pretrained=False):
        super().__init__()
        self.model = Cifar10Cnn9l(n_classes=n_classes, dropout_probability=0)
        self.no_grad = False
    
    def get_last_layer(self):
        return self.model.linear_layers[-1]

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            out = self.model(x)
        return out
    
class Resnet18Adapter(nn.Module):
    def __init__(self, channel=3, n_classes=10, num_classes=10, im_size=32, pretrained=False, record_embedding=False, no_grad=False):
        super().__init__()
        self.n_classes = n_classes
        self.resnet = resnet18(weights="DEFAULT")
        self.resnet.maxpool = nn.Identity()
        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        if channel != 3:
            self.resnet.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        if self.n_classes != 1000:
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.n_classes)
        self.no_grad = False
    
    def get_last_layer(self):
        return self.resnet.fc

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            x = self.resnet.conv1(x)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)

            x = self.resnet.layer1(x)
            x = self.resnet.layer2(x)
            x = self.resnet.layer3(x)
            x = self.resnet.layer4(x)

            x = self.resnet.avgpool(x)
            x = flatten(x, 1)
            x = self.embedding_recorder(x)
            x = self.resnet.fc(x)

        return x