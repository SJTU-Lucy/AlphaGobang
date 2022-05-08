import common

SIZE = common.size
EMPTY = common.empty
BLACK = common.black
WHITE = common.white


class ChessBoard(object):
    def __init__(self):
        self.__board = [[EMPTY for n in range(SIZE)] for m in range(SIZE)]
        self.__dir = [[(-1, 0), (1, 0)], [(0, -1), (0, 1)], [(-1, 1), (1, -1)], [(-1, -1), (1, 1)]]

    # 返回数组对象
    def board(self):
        return self.__board

    # 修改落子点坐标的状态
    def draw(self, x, y, state):
        self.__board[x][y] = state

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
    def judge(self, x, y):
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

    def reset(self):
        self.__board = [[EMPTY for n in range(SIZE)] for m in range(SIZE)]
