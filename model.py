from torch import nn
import torch.nn.functional as F
import torchvision
import config
from torch.nn import init

class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.model = self.get_model()
    def get_model(self):
        model = torchvision.models.resnet18()
        # for param in model.parameters():
        #     param.requires_grad = False
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(model.fc.in_features, config.N_CLASSES, bias=True)
        return model
    def forward(self, x):
        return self.model(x)

class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.model = self.get_model()
    def get_model(self):
        model = torchvision.models.resnet50()
        # for param in model.parameters():
        #     param.requires_grad = False
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(model.fc.in_features, config.N_CLASSES, bias=True)
        return model
    def forward(self, x):
        return self.model(x)

class MobilenetV3(nn.Module):
    def __init__(self):
        super(MobilenetV3, self).__init__()
        self.model = self.get_model()
    def get_model(self):
        model = torchvision.models.mobilenet_v3_small()
        model.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[3] = nn.Linear(in_features=model.classifier[3].in_features, out_features=config.N_CLASSES, bias=True)
        return model
    def forward(self, x):
        return self.model(x)

class Squeezenet(nn.Module):
    def __init__(self):
        super(Squeezenet, self).__init__()
        self.model = self.get_model()
    def get_model(self):
        model = torchvision.models.squeezenet1_1()
        # print(model)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2))
        model.classifier[1] = nn.Conv2d(512, config.N_CLASSES, kernel_size=(1, 1), stride=(1, 1))
        return model
    def forward(self, x):
        return self.model(x)

class AudioClassifier(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=config.N_CLASSES)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
 
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x

if __name__ == '__main__':
    model = Squeezenet()
    print(model)