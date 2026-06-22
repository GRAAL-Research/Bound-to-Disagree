from torch import nn
import loralib as lora
from PBB.pbb.models import ProbLinear, ProbConv2d

class MnistCnn(nn.Module):
    def __init__(self, n_classes=2, dropout_probability=0.2):
        super().__init__()
        self.n_classes = n_classes
        self.dropout_probability = dropout_probability
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            nn.MaxPool2d(2)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(128, self.n_classes),
        )

    def forward(self, input):
        out = self.cnn_layers(input)
        return self.linear_layers(out.flatten(1))

class MnistLoraCnn(nn.Module):
    def __init__(self, n_classes=2, dropout_probability=0.2, rank=16, lora_alpha=1.0, init_model=None):
        super().__init__()
        self.n_classes = n_classes
        self.dropout_probability = dropout_probability
        self.cnn_layers = nn.Sequential(
            lora.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, r=rank, lora_alpha=lora_alpha, merge_weights=False),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            lora.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, r=rank, lora_alpha=lora_alpha, merge_weights=False),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            nn.MaxPool2d(2)
        )
        self.linear_layers = nn.Sequential(
            lora.Linear(9216, 128, r=rank, lora_alpha=lora_alpha, merge_weights=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            lora.Linear(128, self.n_classes, r=rank, lora_alpha=lora_alpha, merge_weights=False),
        )

        if init_model is None:
            return self
        
        for i in range(len(self.cnn_layers)):
            if isinstance(self.cnn_layers[i], lora.Conv2d):
                self.cnn_layers[i].conv.weight.data.copy_(init_model.cnn_layers[i].weight.data)
                self.cnn_layers[i].conv.bias.data.copy_(init_model.cnn_layers[i].bias.data)

        for i in range(len(self.linear_layers)):
            if isinstance(self.linear_layers[i], lora.Linear):
                self.linear_layers[i].weight.data.copy_(init_model.linear_layers[i].weight.data)
                self.linear_layers[i].bias.data.copy_(init_model.linear_layers[i].bias.data)

    def forward(self, input):
        out = self.cnn_layers(input)
        return self.linear_layers(out.flatten(1))
    
class MnistProbCnn(nn.Module):
    def __init__(self, n_classes=2, rho_prior=1e-2, device='cuda', init_net=None):
        super().__init__()
        self.n_classes = n_classes
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.sample = False
        
        self.conv1 = ProbConv2d(in_channels=1,
                        out_channels=32,
                        kernel_size=3, 
                        stride=1,
                        rho_prior=rho_prior,
                        device=device, 
                        init_layer=init_net.cnn_layers[0] if init_net else None)
        self.conv2 = ProbConv2d(in_channels=32,
                        out_channels=64, 
                        kernel_size=3,
                        stride=1,
                        rho_prior=rho_prior, 
                        device=device,
                        init_layer=init_net.cnn_layers[3] if init_net else None)
        
        self.linear1 = ProbLinear(in_features=9216,
                        out_features=128,
                        rho_prior=rho_prior,
                        device=device,
                        init_layer=init_net.linear_layers[0] if init_net else None)
        self.linear2 = ProbLinear(in_features=128,
                        out_features=self.n_classes,
                        rho_prior=rho_prior,
                        device=device, 
                        init_layer=init_net.linear_layers[3] if init_net else None)

    def set_sampling_mode(self, sample=True):
        self.sample = sample

    def forward(self, input):
        out = self.relu(self.conv1(input, sample=self.sample))
        out = self.maxpool(self.relu(self.conv2(out, sample=self.sample)))

        out = self.relu(self.linear1(out.flatten(1), sample=self.sample))
        out = self.linear2(out, sample=self.sample)
        return out
    
    def compute_kl(self):
        kl_div = 0.0
        for _, vals in self.named_children():
            if getattr(vals, "kl_div", None) is not None:
                kl_div += vals.kl_div
        return kl_div


class Cifar10Cnn9l(nn.Module):
    def __init__(self, n_classes=2, dropout_probability=0.2):
        super().__init__()
        self.n_classes = n_classes
        self.dropout_probability = dropout_probability
        self.cnn_layers = nn.Sequential(
            # layer 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            # layer 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #layer 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            # layer 4
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # layer 5
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            # layer 6
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(512, n_classes)
        )

    def forward(self, input):
        out = self.cnn_layers(input)
        return self.linear_layers(out.flatten(1))

class Cifar10LoraCnn9l(nn.Module):
    def __init__(self, n_classes=2, dropout_probability=0.2, rank=16, lora_alpha=1.0, init_model=None):
        super().__init__()
        self.n_classes = n_classes
        self.dropout_probability = dropout_probability
        self.cnn_layers = nn.Sequential(
            # layer 1
            lora.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, r=rank, lora_alpha=lora_alpha, merge_weights=False),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            # layer 2
            lora.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, r=rank, lora_alpha=lora_alpha, merge_weights=False),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #layer 3
            lora.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, r=rank, lora_alpha=lora_alpha, merge_weights=False),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            # layer 4
            lora.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, r=rank, lora_alpha=lora_alpha, merge_weights=False),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # layer 5
            lora.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, r=rank, lora_alpha=lora_alpha, merge_weights=False),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            # layer 6
            lora.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, r=rank, lora_alpha=lora_alpha, merge_weights=False),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_probability),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.linear_layers = nn.Sequential(
            lora.Linear(4096, 1024, r=rank, lora_alpha=lora_alpha, merge_weights=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            lora.Linear(1024, 512, r=rank, lora_alpha=lora_alpha, merge_weights=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            lora.Linear(512, n_classes, r=rank, lora_alpha=lora_alpha, merge_weights=False)
        )

        if init_model is None:
            return self
        
        for i in range(len(self.cnn_layers)):
            if isinstance(self.cnn_layers[i], lora.Conv2d):
                self.cnn_layers[i].conv.weight.data.copy_(init_model.cnn_layers[i].weight.data)
                self.cnn_layers[i].conv.bias.data.copy_(init_model.cnn_layers[i].bias.data)

        for i in range(len(self.linear_layers)):
            if isinstance(self.linear_layers[i], lora.Linear):
                self.linear_layers[i].weight.data.copy_(init_model.linear_layers[i].weight.data)
                self.linear_layers[i].bias.data.copy_(init_model.linear_layers[i].bias.data)

    def forward(self, input):
        out = self.cnn_layers(input)
        return self.linear_layers(out.flatten(1))

class Cifar10ProbCnn9l(nn.Module):
    def __init__(self, n_classes=2, rho_prior=1e-2, device='cuda', init_net=None):
        super().__init__()
        self.n_classes = n_classes
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ProbConv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1,
                                rho_prior=rho_prior, device=device, 
                                init_layer=init_net.cnn_layers[0] if init_net else None)
        self.conv2 = ProbConv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1,
                                rho_prior=rho_prior, device=device, 
                                init_layer=init_net.cnn_layers[3] if init_net else None)
        self.conv3 = ProbConv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1,
                                rho_prior=rho_prior, device=device, 
                                init_layer=init_net.cnn_layers[7] if init_net else None)
        self.conv4 = ProbConv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1,
                                rho_prior=rho_prior, device=device, 
                                init_layer=init_net.cnn_layers[10] if init_net else None)
        self.conv5 = ProbConv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1,
                                rho_prior=rho_prior, device=device, 
                                init_layer=init_net.cnn_layers[14] if init_net else None)
        self.conv6 = ProbConv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1,
                                rho_prior=rho_prior, device=device, 
                                init_layer=init_net.cnn_layers[17] if init_net else None)

        self.linear1 = ProbLinear(in_features=4096, out_features=1024,
                                  rho_prior=rho_prior, device=device, 
                                init_layer=init_net.linear_layers[0] if init_net else None)
        self.linear2 = ProbLinear(in_features=1024, out_features=512,
                                  rho_prior=rho_prior, device=device, 
                                init_layer=init_net.linear_layers[3] if init_net else None)
        self.linear3 = ProbLinear(in_features=512, out_features=n_classes,
                                  rho_prior=rho_prior, device=device, 
                                init_layer=init_net.linear_layers[6] if init_net else None)

    def set_sampling_mode(self, sample=True):
        self.sample = sample

    def forward(self, input):
        out = self.relu(self.conv1(input, sample=self.sample))
        out = self.maxpool(self.relu(self.conv2(out, sample=self.sample)))
        out = self.relu(self.conv3(out, sample=self.sample))
        out = self.maxpool(self.relu(self.conv4(out, sample=self.sample)))
        out = self.relu(self.conv5(out, sample=self.sample))
        out = self.maxpool(self.relu(self.conv6(out, sample=self.sample)))

        out = self.relu(self.linear1(out.flatten(1), sample=self.sample))
        out = self.relu(self.linear2(out, sample=self.sample))
        out = self.linear3(out, sample=self.sample)

        return out
    
    def compute_kl(self):
        kl_div = 0.0
        for _, vals in self.named_children():
            if getattr(vals, "kl_div", None) is not None:
                kl_div += vals.kl_div
        return kl_div