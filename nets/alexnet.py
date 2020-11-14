from torch import nn
from torch.nn import functional as F

class AlexNet(nn.Module):
    """ AlexNet class implements the 2012 AlexNet Architecture
    input size is 256x256x3
    """

    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv4 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv5 = nn.Conv2d(10, 20, kernel_size=3)
#         self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(5*5*20, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(50, 1000)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
            
        return num_features


class AlexNetCIFAR10(nn.Module):
    """ AlexNet class implements the 2012 AlexNet Architecture
    input size is 256x256x3
    """

    def __init__(self):
        super(AlexNetCIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv4 = nn.Conv2d(10, 20, kernel_size=3)
#         self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(5*5*20, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(50, 1000)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
            
        return num_features
