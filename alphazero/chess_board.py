from typing import Tuple
from copy import deepcopy
from collections import OrderedDict
from alphazero import common
import torch
import numpy as np

SIZE = common.size
EMPTY = common.empty
WHITE = common.white
BLACK = common.black


class ChessBoard:
    def __init__(self, board_len=common.size, n_feature_planes=common.feature_planes):
        self.__board = [[EMPTY for n in range(SIZE)] for m in range(SIZE)]
        self.board_len = board_len
        self.current_player = BLACK
        self.n_feature_planes = n_feature_planes
        self.available_actions = list(range(self.board_len**2))
        self.state = OrderedDict()
        self.previous_action = None

    def copy(self):
        return deepcopy(self)

    def get(self, x, y):
        if x < 0 or x >= SIZE or y < 0 or y >= SIZE:
            return None
        return self.__board[x][y]

    def clear_board(self):
        self.__board = [[EMPTY for n in range(SIZE)] for m in range(SIZE)]
        self.state.clear()
        self.previous_action = None
        self.current_player = BLACK
        self.available_actions = list(range(self.board_len**2))

    def do_action(self, action: int):
        self.__board[action // SIZE][action % SIZE] = self.current_player
        self.state[action] = self.current_player
        self.previous_action = action
        self.current_player = WHITE + BLACK - self.current_player
        self.available_actions.remove(action)

    def do_action_(self, pos: tuple) -> bool:
        action = pos[0]*self.board_len + pos[1]
        if action in self.available_actions:
            self.do_action(action)
            return True
        return False

    def is_game_over(self) -> Tuple[bool, int]:
        if len(self.state) < 9:
            return False, None

        n = self.board_len
        act = self.previous_action
        player = self.state[act]
        row, col = act//n, act % n

        directions = [[(0, -1),  (0, 1)],
                      [(-1, 0),  (1, 0)],
                      [(-1, -1), (1, 1)],
                      [(1, -1),  (-1, 1)]]

        for i in range(4):
            count = 1
            for j in range(2):
                flag = True
                row_t, col_t = row, col
                while flag:
                    row_t = row_t + directions[i][j][0]
                    col_t = col_t + directions[i][j][1]
                    if 0 <= row_t < n and 0 <= col_t < n and self.state.get(row_t*n+col_t, EMPTY) == player:
                        count += 1
                    else:
                        flag = False
            if count == 5:
                return True, player
            if count > 5 and player == BLACK:
                return True, WHITE

        if player == BLACK:
            threeline = 0
            fourline = 0
            for i in range(4):
                foundthree = False
                foundfour = False
                for j in range(2):
                    down = directions[i][j]
                    down1 = self.get(row + down[0], col + down[1])
                    down2 = self.get(row + down[0] * 2, col + down[1] * 2)
                    up = directions[i][1 - j]
                    up1 = self.get(row + up[0], col + up[1])
                    up2 = self.get(row + up[0] * 2, col + up[1] * 2)
                    up3 = self.get(row + up[0] * 3, col + up[1] * 3)
                    up4 = self.get(row + up[0] * 4, col + up[1] * 4)
                    if down1 == EMPTY and down2 == EMPTY and up1 == BLACK and up2 == BLACK and up3 == EMPTY:
                        foundthree = True
                    if down1 == EMPTY and up1 == EMPTY and up2 == BLACK and up3 == BLACK and up4 == EMPTY:
                        foundthree = True
                    if down1 == EMPTY and up1 == BLACK and up2 == BLACK and up3 == EMPTY and up4 == EMPTY:
                        foundthree = True
                    if down1 == BLACK and down2 == EMPTY and up1 == BLACK and up2 == EMPTY and up3 == EMPTY:
                        foundthree = True
                    if down1 == EMPTY and up1 == BLACK and up2 == BLACK and up3 == BLACK and up4 == EMPTY:
                        foundfour = True
                    if down1 == BLACK and down2 == EMPTY and up1 == BLACK and up2 == BLACK and up3 == EMPTY:
                        foundfour = True
                if foundthree:
                    threeline += 1
                if foundfour:
                    fourline += 1
            if threeline >= 2 or fourline >= 2:
                return True, WHITE
            if threeline >= 2:
                return True, WHITE

        if not self.available_actions:
            return True, None

        return False, None

    def get_feature_planes(self) -> torch.Tensor:
        n = self.board_len
        feature_planes = torch.zeros((self.n_feature_planes, n**2))
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
