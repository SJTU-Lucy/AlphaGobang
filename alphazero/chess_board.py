# coding: utf-8
from typing import Tuple
from copy import deepcopy
from collections import OrderedDict
from alphazero import common

import torch
import numpy as np

SIZE = common.size
EMPTY = common.empty
BLACK = common.black
WHITE = common.white
TRANSEMPTY = -1
TRANSWHITE = 0
TRANSBLACK = 1


class ChessBoard:
    """ 棋盘类 """

    def __init__(self, board_len=SIZE, n_feature_planes=7):
        """
        Parameters
        ----------
        board_len: int
            棋盘边长

        n_feature_planes: int
            特征平面的个数，必须为偶数
        """
        self.__board = [[EMPTY for n in range(SIZE)] for m in range(SIZE)]
        self.__dir = [[(-1, 0), (1, 0)], [(0, -1), (0, 1)], [(-1, 1), (1, -1)], [(-1, -1), (1, 1)]]
        self.board_len = SIZE
        self.current_player = BLACK
        self.n_feature_planes = common.feature_planes
        self.available_actions = list(range(self.board_len**2))
        self.state = OrderedDict()
        self.previous_action = None

    def copy(self):
        """ 复制棋盘 """
        return deepcopy(self)

    def clear_board(self):
        """ 清空棋盘 """
        self.__board = [[EMPTY for n in range(SIZE)] for m in range(SIZE)]
        self.state.clear()
        self.previous_action = None
        self.current_player = BLACK
        self.available_actions = list(range(self.board_len**2))

    # 重置棋盘
    def reset(self):
        self.__board = [[EMPTY for n in range(SIZE)] for m in range(SIZE)]
        self.state.clear()
        self.previous_action = None
        self.current_player = BLACK
        self.available_actions = list(range(self.board_len ** 2))

    # 修改落子点坐标的状态
    def draw(self, x, y, player):
        self.__board[x][y] = player
        self.state[x * SIZE + y] = self.trans(player)
        self.previous_action = x * SIZE + y
        self.current_player = BLACK + WHITE - self.current_player
        self.available_actions.remove(x * SIZE + y)

    def do_action(self, action: int):
        self.__board[action//SIZE][action%SIZE] = self.current_player
        self.state[action] = self.trans(self.current_player)
        self.previous_action = action
        self.current_player = WHITE + BLACK - self.current_player
        self.available_actions.remove(action)

    def do_action_(self, pos: tuple) -> bool:
        """ 落子并更新棋盘，只提供给 app 使用

        Parameters
        ----------
        pos: Tuple[int, int]
            落子在棋盘上的位置，范围为 `(0, 0) ~ (board_len-1, board_len-1)`

        Returns
        -------
        update_ok: bool
            是否成功落子
        """
        action = pos[0]*self.board_len + pos[1]
        if action in self.available_actions:
            self.do_action(action)
            return True
        return False

    def is_game_over(self) -> Tuple[bool, int]:
        """ 判断游戏是否结束

        Returns
        -------
        is_over: bool
            游戏是否结束，分出胜负或者平局则为 `True`, 否则为 `False`

        winner: int
            游戏赢家，有以下几种:
            * 如果游戏分出胜负，则为 `ChessBoard.BLACK` 或 `ChessBoard.WHITE`
            * 如果还有分出胜负或者平局，则为 `None`
        """
        # 如果下的棋子不到 9 个，就直接判断游戏还没结束
        if len(self.state) < 9:
            return False, None

        n = self.board_len
        act = self.previous_action
        player = self.state[act]
        row, col = act//n, act % n

        # 搜索方向
        directions = [[(0, -1),  (0, 1)],   # 水平搜索
                      [(-1, 0),  (1, 0)],   # 竖直搜索
                      [(-1, -1), (1, 1)],   # 主对角线搜索
                      [(1, -1),  (-1, 1)]]  # 副对角线搜索

        for i in range(4):
            count = 1
            for j in range(2):
                flag = True
                row_t, col_t = row, col
                while flag:
                    row_t = row_t + directions[i][j][0]
                    col_t = col_t + directions[i][j][1]
                    if 0 <= row_t < n and 0 <= col_t < n and self.state.get(row_t*n+col_t, TRANSEMPTY) == player:
                        # 遇到相同颜色时 count+1
                        count += 1
                    else:
                        flag = False
            # 分出胜负
            if count >= 5:
                return True, player

        # 平局
        if not self.available_actions:
            return True, None

        return False, None

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

    # 返回数组对象
    def board(self):
        return self.__board

    def trans(self, player):
        if player == BLACK:
            return TRANSBLACK
        if player == WHITE:
            return TRANSWHITE
        if player == EMPTY:
            return TRANSEMPTY

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
        x = self.previous_action // SIZE
        y = self.previous_action % SIZE
        state = self.get(x, y)
        # 在四个方向上分别检查
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


class ColorError(ValueError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
