import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# take [batch, seq_len] as input, [batch, num_classes] as output


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


class MLP(nn.Module):
    def __init__(self, input_w, input_h, num_classes=3, transformer_layers=1, transformer_dim=64, heads=2):
        super(MLP, self).__init__()
        # [input_w*input_h] -> [1, input_w, input_h]
        self.reshape_layer = lambda x: x.view(-1, 1, input_w, input_h)
        self.fc1 = nn.Linear(1 * (input_w) * (input_h), 256)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(256, 128)  # [transformer_dim]到[num_classes]
        self.dropout2 = nn.Dropout(p=0.5)
        # [transformer_dim]到[num_classes]
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # [batch, input_w*input_h] -> [batch, 1, input_w, input_h]
        x = self.reshape_layer(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


class TECEC_1(nn.Module):
    def __init__(self, input_w, input_h, num_classes, transformer_layers=1, transformer_dim=64, heads=2):
        super(TECEC_1, self).__init__()
        # [input_w*input_h] -> [1, input_w, input_h]
        self.reshape_layer = lambda x: x.view(-1, 1, input_w, input_h)
        # [1, input_w, input_h] -> [32, input_w, input_h]
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # [32, input_w, input_h] -> [64, input_w, input_h]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 下采样，例如：[64, input_w, input_h] -> [64, input_w/2, input_h/2]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=0.5)
        # 全连接调整形状，例如：[64, input_w/2, input_h/2]扁平化后到[transformer_dim]
        self.fc1 = nn.Linear(64 * (input_w // 2) *
                             (input_h // 2), transformer_dim)
        self.dropout2 = nn.Dropout(p=0.75)

        # 定义Transformer Encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=transformer_dim, nhead=heads, dim_feedforward=256)
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_layers=transformer_layers)

        # [transformer_dim]到[num_classes]
        self.fc2 = nn.Linear(transformer_dim, num_classes)

    def forward(self, x):
        # [batch, input_w*input_h] -> [batch, 1, input_w, input_h]
        x = self.reshape_layer(x)
        # [batch, 1, input_w, input_h] -> [batch, 32, input_w, input_h]
        x = F.relu(self.conv1(x))
        # [batch, 32, input_w, input_h] -> [batch, 64, input_w, input_h]
        x = F.relu(self.conv2(x))
        # [batch, 64, input_w, input_h] -> [batch, 64, input_w/2, input_h/2]
        x = self.pool(x)
        x = self.dropout1(x)

        # 调整形状以匹配全连接层，例如：[batch, 64, input_w/2, input_h/2] -> [batch, 64*(input_w/2)*(input_h/2)]
        x = x.view(x.size(0), -1)
        # [batch, 64*(input_w/2)*(input_h/2)] -> [batch, transformer_dim]
        x = F.relu(self.fc1(x))
        # 为序列维度添加unsqueeze，得到：[batch, 1, transformer_dim]
        x = x.unsqueeze(1)

        # 通过Transformer Encoder
        # [batch, 1, transformer_dim] -> [batch, 1, transformer_dim]
        x = self.transformer_encoder(x)
        x = x.squeeze(1)                # 移除序列维度，得到：[batch, transformer_dim]

        x = self.dropout2(x)
        # [batch, transformer_dim] -> [batch, num_classes]
        x = self.fc2(x)
        return x


class TECEC_2(nn.Module):
    def __init__(self, input_w, input_h, num_classes, transformer_layers=1, transformer_dim=128, heads=2):
        super(TECEC_2, self).__init__()
        self.reshape_layer = lambda x: x.view(-1, 1, input_w, input_h)  # [input_w*input_h] -> [1, input_w, input_h]
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # [1, input_w, input_h] -> [32, input_w, input_h]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # [32, input_w, input_h] -> [64, input_w, input_h]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样，例如：[64, input_w, input_h] -> [64, input_w/2, input_h/2]
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(64 * (input_w // 2) * (input_h // 2), transformer_dim)  # 全连接调整形状，例如：[64, input_w/2, input_h/2]扁平化后到[transformer_dim]
        self.dropout2 = nn.Dropout(p=0.75)
        
        # 定义Transformer Encoder
        encoder_layers = TransformerEncoderLayer(d_model=transformer_dim, nhead=heads, dim_feedforward=256)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=transformer_layers)
        self.dropout3 = nn.Dropout(p=0.5)
        encoder_layers_2 = TransformerEncoderLayer(d_model=transformer_dim, nhead=heads, dim_feedforward=256)
        self.transformer_encoder_2 = TransformerEncoder(encoder_layers_2, num_layers=transformer_layers)
        self.fc2 = nn.Linear(transformer_dim, num_classes)  # [transformer_dim]到[num_classes]

    def forward(self, x):
        x = self.reshape_layer(x)  # [batch, input_w*input_h] -> [batch, 1, input_w, input_h]
        x = F.relu(self.conv1(x))  # [batch, 1, input_w, input_h] -> [batch, 32, input_w, input_h]
        x = F.relu(self.conv2(x))  # [batch, 32, input_w, input_h] -> [batch, 64, input_w, input_h]
        x = self.pool(x)          # [batch, 64, input_w, input_h] -> [batch, 64, input_w/2, input_h/2]
        x = self.dropout1(x)
        
        x = x.view(x.size(0), -1)  # 调整形状以匹配全连接层，例如：[batch, 64, input_w/2, input_h/2] -> [batch, 64*(input_w/2)*(input_h/2)]
        x = F.relu(self.fc1(x))    # [batch, 64*(input_w/2)*(input_h/2)] -> [batch, transformer_dim]
        x = x.unsqueeze(1)         # 为序列维度添加unsqueeze，得到：[batch, 1, transformer_dim]
        
        # 通过Transformer Encoder
        x = self.transformer_encoder(x)  # [batch, 1, transformer_dim] -> [batch, 1, transformer_dim]
        x = x.squeeze(1)                # 移除序列维度，得到：[batch, transformer_dim]
        
        x = self.dropout2(x)
        x = x.unsqueeze(1)         # 为序列维度添加unsqueeze，得到：[batch, 1, transformer_dim]
        x = self.transformer_encoder_2(x)  # [batch, 1, transformer_dim] -> [batch, 1, transformer_dim]
        x = self.dropout3(x)
        x = x.squeeze(1)   
        x = self.fc2(x)                 # [batch, transformer_dim] -> [batch, num_classes]
        return x