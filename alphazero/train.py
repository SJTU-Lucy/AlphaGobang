import os
import torch
import torch.nn.functional as F
from torch import nn, optim, cuda
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from alphazero.alpha_zero_mcts import AlphaZeroMCTS
from alphazero.chess_board import ChessBoard
from alphazero.policy_value_net import PolicyValueNet
from alphazero.self_play_dataset import SelfPlayData, SelfPlayDataSet
import alphazero.common as common


class PolicyValueLoss(nn.Module):
    def __init__(self):
        super().__init__()

    # p_hat: 对数动作概率, pi：mcts产生的动作概率, value: 局面估值, z: 玩家奖励
    def forward(self, p_hat, pi, value, z):
        value_loss = F.mse_loss(value, z)
        policy_loss = -torch.sum(pi*p_hat, dim=1).mean()
        loss = value_loss + policy_loss
        return loss


class TrainModel:
    def __init__(self, board_len=common.size, lr=0.01, n_self_plays=1000, n_mcts_iters=500,
                 n_feature_planes=common.feature_planes, batch_size=500, start_train_size=500, check_frequency=100,
                 n_test_games=10, c_puct=3, is_use_gpu=True, is_save_game=False):
        self.c_puct = c_puct
        self.is_use_gpu = is_use_gpu
        self.batch_size = batch_size
        self.n_self_plays = n_self_plays
        self.n_test_games = n_test_games
        self.n_mcts_iters = n_mcts_iters
        self.is_save_game = is_save_game
        self.check_frequency = check_frequency
        self.start_train_size = start_train_size
        self.device = torch.device('cuda:0' if is_use_gpu and cuda.is_available() else 'cpu')
        self.chess_board = ChessBoard(board_len, n_feature_planes)
        # 创建策略-价值网络和蒙特卡洛搜索树
        self.policy_value_net = self.get_net()
        self.mcts = AlphaZeroMCTS(self.policy_value_net, c_puct=c_puct, n_iters=n_mcts_iters, is_self_play=True)
        # 创建优化器和损失函数
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = PolicyValueLoss()
        self.lr_scheduler = MultiStepLR(self.optimizer, [1500, 2500], gamma=0.1)
        # 创建数据集
        self.dataset = SelfPlayDataSet(board_len)
        # 记录误差
        self.loss_record = []

    # 进行一局自我博弈，产生一组数据，包含pi_list, z_list, feature_planes_list
    # pi_list: mcts的动作概率，z_list: 每个动作玩家的游戏结果，feature_planes_list: 特征平面
    def __self_play(self):
        # 初始化棋盘和数据容器
        self.policy_value_net.eval()
        self.chess_board.clear_board()
        pi_list, feature_planes_list, players = [], [], []
        action_list = []

        # 开始一局游戏
        while True:
            action, pi = self.mcts.get_action(self.chess_board)
            # 保存每一步的数据
            feature_planes_list.append(self.chess_board.get_feature_planes())
            players.append(self.chess_board.current_player)
            action_list.append(action)
            pi_list.append(pi)
            self.chess_board.do_action(action)
            # 判断游戏是否结束
            is_over, winner = self.chess_board.is_game_over()
            if is_over:
                if winner is not None:
                    z_list = [1 if i == winner else -1 for i in players]
                else:
                    z_list = [0]*len(players)
                break
        # 重置根节点
        self.mcts.reset_root()
        self_play_data = SelfPlayData(pi_list=pi_list, z_list=z_list, feature_planes_list=feature_planes_list)
        return self_play_data

    # 进行训练的主函数
    def train(self):
        for i in range(self.n_self_plays):
            print(f'正在进行第 {i+1} 局自我博弈游戏...')
            self.dataset.append(self.__self_play())

            # 如果数据集中的数据量大于 start_train_size 就进行一次训练
            if len(self.dataset) >= self.start_train_size:
                data_loader = iter(DataLoader(self.dataset, self.batch_size, shuffle=True, drop_last=False))
                print('开始训练...')

                self.policy_value_net.train()
                # 随机选出一批数据来训练，防止过拟合
                feature_planes, pi, z = next(data_loader)
                feature_planes = feature_planes.to(self.device)
                pi, z = pi.to(self.device), z.to(self.device)
                for _ in range(5):
                    # 前馈
                    p_hat, value = self.policy_value_net(feature_planes)
                    # 梯度清零
                    self.optimizer.zero_grad()
                    # 计算损失
                    loss = self.criterion(p_hat, pi, value.flatten(), z)
                    # 误差反向传播
                    loss.backward()
                    # 更新参数
                    self.optimizer.step()
                    # 学习率退火
                    self.lr_scheduler.step()
                self.loss_record.append(loss.item())
                print(f"train_loss = {loss.item():<10.5f}\n")

            # 测试模型
            if (i+1) % self.check_frequency == 0:
                self.__test_model()

    # 测试模型胜率
    def __test_model(self):
        model_path = 'model/best_policy_value_net.pth'
        # 如果最佳模型不存在保存当前模型为最佳模型
        if not os.path.exists(model_path):
            torch.save(self.policy_value_net, model_path)
            return
        # 载入历史最优模型
        best_model = torch.load(model_path)
        best_model.eval()
        best_model.set_device(self.is_use_gpu)
        mcts = AlphaZeroMCTS(best_model, self.c_puct, self.n_mcts_iters)
        self.mcts.set_self_play(False)
        self.policy_value_net.eval()

        # 开始比赛
        print('正在测试当前模型...')
        n_wins = 0
        for i in range(self.n_test_games):
            self.chess_board.clear_board()
            self.mcts.reset_root()
            mcts.reset_root()
            while True:
                # 当前模型走一步
                is_over, winner = self.__do_mcts_action(self.mcts)
                if is_over:
                    n_wins += int(winner == common.black)
                    break
                # 历史最优模型走一步
                is_over, winner = self.__do_mcts_action(mcts)
                if is_over:
                    break

        # 如果胜率大于55%，就保存当前模型为最优模型
        win_prob = n_wins / self.n_test_games
        if win_prob > 0.55:
            torch.save(self.mcts.policy_value_net, model_path)
            print(f'保存当前模型为最优模型，当前模型胜率为：{win_prob:.1%}\n')
        else:
            print(f'保持历史最优模型不变，当前模型胜率为：{win_prob:.1%}\n')
        self.mcts.set_self_play(True)

    # 保存模型
    def save_model(self, model_name: str, loss_name: str, game_name: str):
        os.makedirs('model', exist_ok=True)
        path = f'model/{model_name}.pth'
        self.policy_value_net.eval()
        torch.save(self.policy_value_net, path)
        print(f'已将当前模型保存到 {os.path.join(os.getcwd(), path)}')

    # 用mcts移动一步
    def __do_mcts_action(self, mcts):
        action = mcts.get_action(self.chess_board)
        self.chess_board.do_action(action)
        is_over, winner = self.chess_board.is_game_over()
        return is_over, winner

    def get_net(self):
        model = f'model/best_policy_value_net.pth'
        if os.path.exists(model):
            print(f'载入已有模型 {model} ...\n')
            net = torch.load(model).to(self.device)  # type:PolicyValueNet
            net.set_device(self.is_use_gpu)
        else:
            net = PolicyValueNet().to(self.device)
        return net