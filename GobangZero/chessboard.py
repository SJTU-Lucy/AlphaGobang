from copy import deepcopy
from typing import Tuple
from collections import OrderedDict
import torch
import numpy as np
import common

SIZE = common.size
EMPTY = common.empty
BLACK = common.black
WHITE = common.white


class ChessBoard(object):
    def __init__(self):
        self.__board = [[EMPTY for n in range(SIZE)] for m in range(SIZE)]
        self.__dir = [[(-1, 0), (1, 0)], [(0, -1), (0, 1)], [(-1, 1), (1, -1)], [(-1, -1), (1, 1)]]
        self.board_len = SIZE
        self.current_player = BLACK
        self.n_feature_planes = common.feature_planes
        self.available_actions = list(range(self.board_len**2))
        self.state = OrderedDict()
        self.previous_action = None

    # 返回数组对象
    def board(self):
        return self.__board

    # 修改落子点坐标的状态
    def draw(self, x, y, state=None):
        if state is None:
            self.__board[x][y] = self.current_player
            self.state[x * SIZE + y] = self.current_player
        else:
            self.__board[x][y] = state
            self.state[x * SIZE + y] = state
        self.previous_action = x * SIZE + y
        self.current_player = BLACK + WHITE - self.current_player
        self.available_actions.remove(x*SIZE+y)

    # 获取指定点坐标的状态
    def get(self, x, y):
        return self.__board[x][y]

    # 获得指定位置的指定方向坐标
    def getPos(self, point, direction):
        x = point[0] + direction[0]
        y = point[1] + direction[1]
        if x < 0 or x >= SIZE or y < 0 or y >= SIZE:
            return False
        else:
            return x, y

    # 获得指定位置的指定方向状态
    def getState(self, point, direction):
        if point is not False:
            xy = self.getPos(point, direction)
            if xy is not False:
                x, y = xy
                return self.__board[x][y]
        return False

    # 检测是否存在五子相连
    def judge(self):
        x = self.previous_action / common.size
        y = self.previous_action % common.size
        state = self.get(x, y)
        for directions in self.__dir:
            count = 1
            for direction in directions:
                point = (x, y)
                while True:
                    if self.getState(point, direction) == state:
                        count += 1
                        point = self.getPos(point, direction)
                    else:
                        break
            if count >= 5:
                return state
        return EMPTY

    # 重置棋盘
    def reset(self):
        self.__board = [[EMPTY for n in range(SIZE)] for m in range(SIZE)]
        self.state.clear()
        self.previous_action = None
        self.current_player = BLACK
        self.available_actions = list(range(self.board_len ** 2))

    # 复制棋盘
    def copy(self):
        """ 复制棋盘 """
        return deepcopy(self)

    # 棋盘状态特征张量
    def get_feature_planes(self) -> torch.Tensor:
        """ 棋盘状态特征张量，维度为 `(n_feature_planes, board_len, board_len)`

        Returns
        -------
        feature_planes: Tensor of shape `(n_feature_planes, board_len, board_len)`
            特征平面图像
        """
        n = self.board_len
        feature_planes = torch.zeros((self.n_feature_planes, n**2))
        # 最后一张图像代表当前玩家颜色
        # feature_planes[-1] = self.current_player
        # 添加历史信息
        if self.state:
            actions = np.array(list(self.state.keys()))[::-1]
            players = np.array(list(self.state.values()))[::-1]
            Xt = actions[players == self.current_player]
            Yt = actions[players != self.current_player]
            for i in range((self.n_feature_planes-1)//2):
                if i < len(Xt):
                    feature_planes[2*i, Xt[i:]] = 1
                if i < len(Yt):
                    feature_planes[2*i+1, Yt[i:]] = 1

        return feature_planes.view(self.n_feature_planes, n, n)
