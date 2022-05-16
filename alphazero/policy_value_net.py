import torch
from torch import nn
from torch.nn import functional as F
from alphazero.chess_board import ChessBoard
import alphazero.common as common


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channel: int, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channel,
                              kernel_size=kernel_size, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        return F.relu(self.batch_norm(self.conv(x)))


class ResidueBlock(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        out = F.relu(self.batch_norm1(self.conv1(x)))
        out = self.batch_norm2(self.conv2(out))
        return F.relu(out + x)


class PolicyHead(nn.Module):
    def __init__(self, in_channels=128, board_len=9):
        super().__init__()
        self.board_len = board_len
        self.in_channels = in_channels
        self.conv = ConvBlock(in_channels, 2, 1)
        self.fc = nn.Linear(2*board_len**2, board_len**2)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        return F.log_softmax(x, dim=1)


class ValueHead(nn.Module):
    def __init__(self, in_channels=128, board_len=9):
        super().__init__()
        self.in_channels = in_channels
        self.board_len = board_len
        self.conv = ConvBlock(in_channels, 1, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Linear(board_len**2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        return x


class PolicyValueNet(nn.Module):
    def __init__(self, board_len=common.size, n_feature_planes=common.feature_planes, is_use_gpu=True):
        super().__init__()
        self.board_len = board_len
        self.is_use_gpu = is_use_gpu
        self.n_feature_planes = n_feature_planes
        self.device = torch.device('cuda:0' if is_use_gpu else 'cpu')
        self.conv = ConvBlock(n_feature_planes, 128, 3, padding=1)
        self.residues = nn.Sequential(*[ResidueBlock(128, 128) for i in range(4)])
        self.policy_head = PolicyHead(128, board_len)
        self.value_head = ValueHead(128, board_len)

    def forward(self, x):
        x = self.conv(x)
        x = self.residues(x)
        p_hat = self.policy_head(x)
        value = self.value_head(x)
        return p_hat, value

    def predict(self, chess_board: ChessBoard):
        feature_planes = chess_board.get_feature_planes().to(self.device)
        feature_planes.unsqueeze_(0)
        p_hat, value = self(feature_planes)
        # 将对数概率转换为概率
        p = torch.exp(p_hat).flatten()
        # 只取可行的落点
        if self.is_use_gpu:
            p = p[chess_board.available_actions].cpu().detach().numpy()
        else:
            p = p[chess_board.available_actions].detach().numpy()
        return p, value[0].item()

    def set_device(self, is_use_gpu: bool):
        self.is_use_gpu = is_use_gpu
        self.device = torch.device('cuda:0' if is_use_gpu else 'cpu')
