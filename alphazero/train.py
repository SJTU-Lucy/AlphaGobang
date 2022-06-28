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

    def forward(self, p_hat, pi, value, z):
        value_loss = F.mse_loss(value, z)
        policy_loss = -torch.sum(pi*p_hat, dim=1).mean()
        loss = value_loss + policy_loss
        return loss


class TrainModel:
    def __init__(self, board_len=common.size, lr=0.01, n_self_plays=2000, n_mcts_iters=500,
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
        self.policy_value_net = self.get_net()
        self.mcts = AlphaZeroMCTS(self.policy_value_net, c_puct=c_puct, n_iters=n_mcts_iters, is_self_play=True)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = PolicyValueLoss()
        self.lr_scheduler = MultiStepLR(self.optimizer, [1500, 2500], gamma=0.1)
        self.dataset = SelfPlayDataSet(board_len)
        self.loss_record = []
        self.updatecount = 0

    def __self_play(self):
        # 初始化棋盘和数据容器
        self.policy_value_net.eval()
        self.chess_board.clear_board()
        pi_list, feature_planes_list, players = [], [], []
        action_list = []

        while True:
            action, pi = self.mcts.get_action(self.chess_board)
            feature_planes_list.append(self.chess_board.get_feature_planes())
            players.append(self.chess_board.current_player)
            action_list.append(action)
            pi_list.append(pi)
            self.chess_board.do_action(action)
            is_over, winner = self.chess_board.is_game_over()
            if is_over:
                if winner is not None:
                    z_list = [1 if i == winner else -1 for i in players]
                else:
                    z_list = [0]*len(players)
                break
        self.mcts.reset_root()
        self_play_data = SelfPlayData(pi_list=pi_list, z_list=z_list, feature_planes_list=feature_planes_list)
        return self_play_data

    def train(self):
        for i in range(self.n_self_plays):
            print(f'开始第 {i+1} 轮训练')
            self.dataset.append(self.__self_play())

            if len(self.dataset) >= self.start_train_size:
                data_loader = iter(DataLoader(self.dataset, self.batch_size, shuffle=True, drop_last=False))

                self.policy_value_net.train()
                feature_planes, pi, z = next(data_loader)
                feature_planes = feature_planes.to(self.device)
                pi, z = pi.to(self.device), z.to(self.device)
                for _ in range(5):
                    p_hat, value = self.policy_value_net(feature_planes)
                    self.optimizer.zero_grad()
                    loss = self.criterion(p_hat, pi, value.flatten(), z)
                    loss.backward()
                    self.optimizer.step()
                    self.lr_scheduler.step()
                self.loss_record.append(loss.item())
                print(f"train_loss = {loss.item():<10.5f}\n")

            if (i+1) % self.check_frequency == 0:
                self.__test_model()

    def __test_model(self):
        model_path = 'model/best_model.pth'
        if not os.path.exists(model_path):
            torch.save(self.policy_value_net, model_path)
            return
        best_model = torch.load(model_path)
        best_model.eval()
        best_model.set_device(self.is_use_gpu)
        mcts = AlphaZeroMCTS(best_model, self.c_puct, self.n_mcts_iters)
        self.mcts.set_self_play(False)
        self.policy_value_net.eval()

        n_wins = 0
        for i in range(self.n_test_games):
            self.chess_board.clear_board()
            self.mcts.reset_root()
            mcts.reset_root()
            while True:
                is_over, winner = self.__do_mcts_action(self.mcts)
                if is_over:
                    n_wins += int(winner == common.black)
                    break
                is_over, winner = self.__do_mcts_action(mcts)
                if is_over:
                    break

        win_prob = n_wins / self.n_test_games
        print(f"获胜概率为: {win_prob:.1%}\n")
        if win_prob > 0.55:
            torch.save(self.mcts.policy_value_net, model_path)
        self.mcts.set_self_play(True)

    # 保存模型
    def save_model(self):
        os.makedirs('model', exist_ok=True)
        path = f'model/best_model.pth'
        self.policy_value_net.eval()
        torch.save(self.policy_value_net, path)
        self.updatecount += 1

    def __do_mcts_action(self, mcts):
        action = mcts.get_action(self.chess_board)
        self.chess_board.do_action(action)
        is_over, winner = self.chess_board.is_game_over()
        return is_over, winner

    def get_net(self):
        model = f'model/best_model.pth'
        if os.path.exists(model):
            print("载入现有模型")
            net = torch.load(model).to(self.device)
            net.set_device(self.is_use_gpu)
        else:
            net = PolicyValueNet().to(self.device)
        return net
