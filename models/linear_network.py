from torch import nn

class MnistMlp(nn.Module):
    def __init__(self, dataset_shape=784, n_classes=2, dropout_probability=0.2):
        super().__init__()
        self.dataset_shape = dataset_shape
        self.n_classes = n_classes
        self.dropout_probability = dropout_probability
        self.layers = nn.Sequential(
            nn.Linear(self.dataset_shape, 600),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(600, self.n_classes)
        )

    def forward(self, input):
        x = input.flatten(1)
        return self.layers(x)

