from torch import nn
from torch.nn import functional as F


class AlexNet(nn.Module):
    """ AlexNet class implements the 2012 AlexNet Architecture
    input size is 256x256x3
    """

    def __init__(self):
        super(AlexNet, self).__init__()
        # NB: 224x224 as per the paper doesn't add up to 55 output
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)  # 227x227 > 55x55
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)  # 27x27 > 27x27
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)  # 13x13 > 13x13
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)  # 13x13 > 13x13
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)  # 13x13 > 13x13

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.lrn = nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.75, k=2)
        self.dropout = nn.Dropout()

        self.fc1 = nn.Linear(6 * 6 * 256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        nn.init.normal_(self.conv3.weight, 0, 0.01)
        nn.init.normal_(self.conv4.weight, 0, 0.01)
        nn.init.normal_(self.conv5.weight, 0, 0.01)
        nn.init.normal_(self.fc1.weight, 0, 0.01)
        nn.init.normal_(self.fc2.weight, 0, 0.01)
        nn.init.normal_(self.fc3.weight, 0, 0.01)

        nn.init.constant_(self.conv2.bias, 1)
        nn.init.constant_(self.conv4.bias, 1)
        nn.init.constant_(self.conv5.bias, 1)
        nn.init.constant_(self.fc1.bias, 1)
        nn.init.constant_(self.fc2.bias, 1)
        nn.init.constant_(self.fc3.bias, 1)

        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv3.bias, 0)

    def forward(self, x):
        x = F.relu(self.pool(self.lrn(self.conv1(x))))
        x = F.relu(self.pool(self.lrn(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.pool(self.lrn(self.conv5(x))))

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
