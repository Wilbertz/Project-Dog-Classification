import torch.nn as nn
import torch.nn.functional as f


# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define layers of a CNN
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_dim = 512*7*7  # depth channels * (image h/w (256) / 2**(# of max pool layers))**2
        self.fc1 = nn.Linear(self.fc_dim, 1024)
        self.output = nn.Linear(1024, 133)  # 133 dog breed classes
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Define forward behavior
        x = f.relu(self.conv1(x))
        x = self.pool(x)
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = f.relu(self.conv3(x))
        x = self.pool(x)
        x = f.relu(self.conv4(x))
        x = self.pool(x)
        x = f.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(-1, self.fc_dim)  # Flatten Image
        x = f.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.output(x)

        return x
