from torch import nn
from torch.nn import functional as F


class AlexNet(nn.Module):
    """ AlexNet class implements the 2012 AlexNet Architecture
    input size is 256x256x3
    """

    def __init__(self):
        super(AlexNet, self).__init__()
        # (224 x 224 x 3) > (55 x 55 x 96)
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        # (27 x 27 x 96) > (27 x 27 x 256)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        #  (13 x 13 x 256) > (13 x 13 x 384)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        #  (13 x 13 x 384) > (13 x 13 x 384)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        #  (13 x 13 x 384) > (13 x 13 x 256)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.lrn = nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.75, k=2)

        self.fc1 = nn.Linear(5 * 5 * 20, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = F.relu(self.pool(self.lrn(self.conv1(x))))
        x = F.relu(self.pool(self.lrn(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.pool(self.lrn(self.conv5(x))))

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
