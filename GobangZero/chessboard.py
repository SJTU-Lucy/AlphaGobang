import common

SIZE = common.size
EMPTY = common.empty
BLACK = common.black
WHITE = common.white


class ChessBoard(object):
    def __init__(self):
        self.__board = [[EMPTY for n in range(SIZE)] for m in range(15)]
        self.__dir = [[(-1, 0), (1, 0)], [(0, -1), (0, 1)], [(-1, 1), (1, -1)], [(-1, -1), (1, 1)]]

    def board(self):  # 返回数组对象
        return self.__board

    def draw_xy(self, x, y, state):  # 获取落子点坐标的状态
        self.__board[x][y] = state

    def get_xy_on_logic_state(self, x, y):  # 获取指定点坐标的状态
        return self.__board[x][y]

    def get_next_xy(self, point, direction):  # 获取指定点的指定方向的坐标
        x = point[0] + direction[0]
        y = point[1] + direction[1]
        if x < 0 or x >= SIZE or y < 0 or y >= SIZE:
            return False
        else:
            return x, y

    def get_xy_on_direction_state(self, point, direction):  # 获取指定点的指定方向的状态
        if point is not False:
            xy = self.get_next_xy(point, direction)
            if xy is not False:
                x, y = xy
                return self.__board[x][y]
        return False

    def anyone_win(self, x, y):
        state = self.get_xy_on_logic_state(x, y)
        for directions in self.__dir:  # 对米字的4个方向分别检测是否有5子相连的棋
            count = 1
            for direction in directions:  # 对落下的棋子的同一条线的两侧都要检测，结果累积
                point = (x, y)
                while True:
                    if self.get_xy_on_direction_state(point, direction) == state:
                        count += 1
                        point = self.get_next_xy(point, direction)
                    else:
                        break
            if count >= 5:
                return state
        return EMPTY

    def reset(self):  # 重置
        self.__board = [[EMPTY for n in range(SIZE)] for m in range(15)]
