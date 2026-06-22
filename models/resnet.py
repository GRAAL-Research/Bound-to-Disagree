from torch import nn, flatten
from torchvision.models import resnet18, resnet
import loralib as lora
from PBB.pbb.models import ProbLinear, ProbConv2d
from torch import zeros
from copy import deepcopy

def freeze_params(layer):
    for param in layer.parameters():
        param.requires_grad = False

class Cifar10Resnet(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.n_classes = n_classes
        self.resnet = resnet18(pretrained=False, num_classes=self.n_classes)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=False)
        self.resnet.maxpool = nn.Identity()

    def forward(self, input):
        return self.resnet(input)

class Cifar10LoraResnet(nn.Module):
    def __init__(self, n_classes=10, rank=16, lora_alpha=1.0, init_model=None):
        super().__init__()
        self.n_classes = n_classes

        if init_model is not None:
            self.resnet = deepcopy(init_model.resnet)
        else: 
            self.resnet = resnet18(pretrained=False, num_classes=self.n_classes)
            self.resnet.maxpool = nn.Identity()

        self.resnet.conv1 = lora.Conv2d(in_channels=3, out_channels=64, kernel_size=3, 
                                        stride=(1, 1), padding=(1, 1), bias=False,
                                        r=rank, lora_alpha=lora_alpha, merge_weights=False)
        if init_model is not None:
            self.resnet.conv1.conv.weight.data.copy_(init_model.resnet.conv1.weight.data)
            # self.resnet.conv1.conv.bias.data.copy_(init_model.resnet.conv1.bias.data)

        for i in range(1,5):
            for key, vals in getattr(self.resnet, f"layer{i}").named_modules():
                if isinstance(vals, nn.Conv2d):
                    temp = lora.Conv2d(in_channels=vals.in_channels, 
                                    out_channels=vals.out_channels,
                                    kernel_size=vals.kernel_size[0],
                                    stride=vals.stride,
                                    padding=vals.padding,
                                    bias=(vals.bias is not None),
                                    r=rank,
                                    lora_alpha=lora_alpha,
                                    merge_weights=False)
                    if init_model is not None:
                        if "." in key[2:]:
                            layer = getattr(getattr(init_model.resnet, f"layer{i}")[int(key[0])], key[2:-2])[int(key[-1])]
                        else:
                            layer = getattr(getattr(init_model.resnet, f"layer{i}")[int(key[0])], key[2:])
                        temp.conv.weight.data.copy_(layer.weight.data)
                        # temp.conv.bias.data.copy_(layer.bias.data)
                    if "." in key[2:]:
                        setattr(getattr(getattr(self.resnet, f"layer{i}")[int(key[0])], key[2:-2]), key[-1], temp) 
                    else:
                        setattr(getattr(self.resnet, f"layer{i}")[int(key[0])], key[2:], temp)
        self.resnet.fc = lora.Linear(in_features=self.resnet.fc.in_features,
                                     out_features=self.resnet.fc.out_features,
                                     r=rank,
                                     lora_alpha=lora_alpha,
                                     bias=(self.resnet.fc.bias is not None),
                                     merge_weights=False)
        if init_model is not None:
            self.resnet.fc.weight.data.copy_(init_model.resnet.fc.weight.data)
            self.resnet.fc.bias.data.copy_(init_model.resnet.fc.bias.data)

    def forward(self, input):
        return self.resnet(input)
    

class Cifar10ProbResnet(nn.Module):
    def __init__(self, n_classes=10, rho_prior=1e-2, device='cuda', init_net=None):
        super().__init__()
        self.n_classes = n_classes
        if init_net is None:
            self.resnet = resnet18(pretrained=False, num_classes=self.n_classes)
            self.resnet.maxpool = nn.Identity()
        else:
            self.resnet = deepcopy(init_net.resnet)
        freeze_params(self.resnet)

        init_net.resnet.conv1.bias = nn.Parameter(zeros(init_net.resnet.conv1.out_channels), requires_grad=False)
        self.resnet.conv1 = ProbConv2d(in_channels=3, out_channels=64, kernel_size=3, 
                                        stride=(1, 1), padding=(1, 1), rho_prior=rho_prior,
                                    device=device, dilation=init_net.resnet.conv1.dilation,
                                    init_prior='weights', init_layer_prior=init_net.resnet.conv1 if init_net else None,
                                    init_layer=init_net.resnet.conv1 if init_net else None
                                        )
        for i in range(1,5):
            for key, vals in getattr(self.resnet, f"layer{i}").named_modules():
                if isinstance(vals, nn.Conv2d):
                    vals.bias = nn.Parameter(zeros(vals.out_channels), requires_grad=False)
                    temp = ProbConv2d(in_channels=vals.in_channels, 
                                    out_channels=vals.out_channels,
                                    kernel_size=vals.kernel_size[0],
                                    stride=vals.stride,
                                    padding=vals.padding,
                                    rho_prior=rho_prior,
                                    device=device, 
                                    dilation=vals.dilation,
                                    init_prior="weights",
                                    init_layer_prior=vals if init_net else None,
                                    init_layer=vals if init_net else None
                                    )
                    if "." in key[2:]:
                        setattr(getattr(getattr(getattr(self.resnet, f"layer{i}"), key[0]), key[2:-2]), key[-1], temp)
                    else:
                        setattr(getattr(getattr(self.resnet, f"layer{i}"), key[0]), key[2:], temp)
                
        self.resnet.fc = ProbLinear(in_features=self.resnet.fc.in_features,
                                     out_features=self.resnet.fc.out_features,
                                    rho_prior=rho_prior,
                                    device=device, 
                                    init_prior="weights",
                                    init_layer=init_net.resnet.fc if init_net else None,
                                    init_layer_prior=init_net.resnet.fc if init_net else None)

    def set_sampling_mode(self, sample=True):
        self.sample = sample
    
    def layer_forward(self, layer, input, sample):
        out = input
        for j in range(len(layer)):
            identity = out

            out = layer[j].conv1(out, sample=sample)
            out = layer[j].bn1(out)
            out = layer[j].relu(out)

            out = layer[j].conv2(out, sample=sample)
            out = layer[j].bn2(out)

            if layer[j].downsample is not None:
                identity = layer[j].downsample[1](layer[j].downsample[0](identity, sample=sample))

            out += identity
            out = layer[j].relu(out)

        return out

    def forward(self, input):
        out = self.resnet.conv1(input, sample=self.sample)
        out = self.resnet.bn1(out)
        out = self.resnet.relu(out)
        out = self.resnet.maxpool(out)

        out = self.layer_forward(self.resnet.layer1, out, self.sample)
        out = self.layer_forward(self.resnet.layer2, out, self.sample)
        out = self.layer_forward(self.resnet.layer3, out, self.sample)
        out = self.layer_forward(self.resnet.layer4, out, self.sample)

        out = self.resnet.avgpool(out)
        out = flatten(out, 1)
        out = self.resnet.fc(out, sample=self.sample)
        return out
    
    def compute_kl(self):
        kl_div = self.resnet.conv1.kl_div + self.resnet.fc.kl_div
        for i in range(1, 5):
            for j in range(2):
                for _, vals in getattr(self.resnet, f"layer{i}")[j].named_modules():
                    if getattr(vals, "kl_div", None) is not None:
                        kl_div += vals.kl_div
        return kl_div
