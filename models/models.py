import torch.nn as nn
import torch.nn.functional as F


class GraphCNN(nn.Module):

    def __init__(self, input_w, input_h, num_classes):
        super(GraphCNN, self).__init__()
        # add the reshape operation
        self.reshape_layer = lambda x: x.view(-1, 1, input_w, input_h)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(64 * (input_w // 2) *
                             (input_h // 2), 128)  # 注意这里的形状计算
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # [batch, input_w*input_h] -> [batch, 1, input_w, input_h]
        x = self.reshape_layer(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(-1, 64 * (x.shape[2]) * (x.shape[3]))  # 调整形状以匹配全连接层的输入
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        # output = F.softmax(x, dim=1)
        return x
