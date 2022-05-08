import sys
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QIcon, QPalette, QPainter
from PyQt5.QtMultimedia import QSound
from chessboard import ChessBoard
from ai import searcher
import common

EMPTY = common.empty
BLACK = common.black
WHITE = common.white
SIZE = common.size
WIDTH = common.width
HEIGHT = common.height
MARGIN = common.margin
GRID = common.grid
PIECE = common.piece


class AI(QtCore.QThread):
    finishSignal = QtCore.pyqtSignal(int, int)

    def __init__(self, board, parent=None):
        super(AI, self).__init__(parent)
        self.board = board

    def run(self):
        self.ai = searcher()
        self.ai.board = self.board
        score, x, y = self.ai.search(2, 2)
        self.finishSignal.emit(x, y)


class GoBang(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 棋盘
        self.chessboard = ChessBoard()
        # 设置背景
        palette = QPalette()
        palette.setBrush(self.backgroundRole(), QtGui.QBrush(QtGui.QPixmap('img/chessboard.jpg')))
        self.setPalette(palette)
        # 鼠标变成手指形状
        self.setCursor(Qt.PointingHandCursor)
        self.sound_piece = QSound("sound/luozi.wav")
        self.sound_win = QSound("sound/win.wav")
        self.sound_defeated = QSound("sound/defeated.wav")
        # 设置尺寸，不可设置大小
        self.resize(WIDTH, HEIGHT)
        self.setMinimumSize(QtCore.QSize(WIDTH, HEIGHT))
        self.setMaximumSize(QtCore.QSize(WIDTH, HEIGHT))

        self.setWindowTitle("GoBangZero")  # 窗口名称
        self.setWindowIcon(QIcon('img/black.png'))  # 窗口图标

        self.black = QPixmap('img/black.png')
        self.white = QPixmap('img/white.png')

        self.piece_now = BLACK  # 先手
        self.my_turn = True     # 是否玩家落子
        self.step = 0           # 步数
        self.x, self.y = 1000, 1000     # 上一个落子位置的像素

        # 设置鼠标位置，实时显示棋子
        self.mouse_point = QLabel(self)
        self.mouse_point.setMouseTracking(True)
        self.mouse_point.setScaledContents(True)
        self.mouse_point.setPixmap(self.black)  # 加载黑棋
        self.mouse_point.setGeometry(WIDTH/2, HEIGHT/2, PIECE, PIECE)
        # 提前建好棋子，后续设置位置显示
        self.pieces = [QLabel(self) for i in range(SIZE*SIZE)]
        for piece in self.pieces:
            piece.setMouseTracking(True)
            piece.setVisible(True)  # 图片可视
            piece.setScaledContents(True)  # 图片大小根据标签大小可变

        self.mouse_point.raise_()  # 鼠标始终在最上层
        self.ai_down = True  # AI已下棋，主要是为了加锁，当值是False的时候说明AI正在思考，这时候玩家鼠标点击失效，要忽略掉 mousePressEvent

        self.setMouseTracking(True)
        self.show()

    def paintEvent(self, event):  # 画出指示箭头
        qp = QPainter()
        qp.begin(self)
        self.drawLines(qp)
        qp.end()

    # 重载移动函数，黑色棋子随鼠标移动
    def mouseMoveEvent(self, e):
        self.mouse_point.move(e.x() - 16, e.y() - 16)

    # 重载点击函数，当符合条件时落子，AI同样下一步
    def mousePressEvent(self, e):  # 玩家下棋
        if e.button() == Qt.LeftButton and self.ai_down:
            x, y = e.x(), e.y()  # 鼠标坐标
            i, j = self.coordinate_transform_pixel2map(x, y)  # 对应棋盘坐标
            if not i is None and not j is None:  # 棋子落在棋盘上，排除边缘
                if self.chessboard.get_xy_on_logic_state(i, j) == EMPTY:
                    # 黑子下一步
                    self.draw(i, j)
                    self.ai_down = False
                    board = self.chessboard.board()
                    # 白子下一步
                    self.AI = AI(board)  # 新建线程对象，传入棋盘参数
                    self.AI.finishSignal.connect(self.AI_draw)  # 结束线程，传出参数
                    self.AI.start()

    def AI_draw(self, i, j):
        self.draw(i, j)  # AI
        self.x, self.y = self.coordinate_transform_map2pixel(i, j)
        self.ai_down = True
        self.update()

    def draw(self, i, j):
        x, y = self.coordinate_transform_map2pixel(i, j)
        if self.piece_now == BLACK:
            self.pieces[self.step].setPixmap(self.black)  # 放置黑色棋子
            self.piece_now = WHITE
            self.chessboard.draw_xy(i, j, BLACK)
        else:
            self.pieces[self.step].setPixmap(self.white)  # 放置白色棋子
            self.piece_now = BLACK
            self.chessboard.draw_xy(i, j, WHITE)

        self.pieces[self.step].setGeometry(x, y, PIECE, PIECE)  # 画出棋子
        self.sound_piece.play()  # 落子音效
        self.step += 1  # 步数+1

        winner = self.chessboard.anyone_win(i, j)  # 判断输赢
        if winner != EMPTY:
            self.mouse_point.clear()
            self.gameover(winner)

    def drawLines(self, qp):  # 指示AI当前下的棋子
        if self.step != 0:
            pen = QtGui.QPen(QtCore.Qt.black, 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(self.x - 5, self.y - 5, self.x + 3, self.y + 3)
            qp.drawLine(self.x + 3, self.y, self.x + 3, self.y + 3)
            qp.drawLine(self.x, self.y + 3, self.x + 3, self.y + 3)

    def coordinate_transform_map2pixel(self, i, j):
        # 从 chessMap 里的逻辑坐标到 UI 上的绘制坐标的转换
        return MARGIN + j * GRID - PIECE / 2, MARGIN + i * GRID - PIECE / 2

    def coordinate_transform_pixel2map(self, x, y):
        # 从 UI 上的绘制坐标到 chessMap 里的逻辑坐标的转换
        i, j = int(round((y - MARGIN) / GRID)), int(round((x - MARGIN) / GRID))
        # 有MAGIN, 排除边缘位置导致 i,j 越界
        if i < 0 or i >= SIZE or j < 0 or j >= SIZE:
            return None, None
        else:
            return i, j

    def gameover(self, winner):
        if winner == BLACK:
            self.sound_win.play()
            reply = QMessageBox.question(self, 'You Win!', 'Continue?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        else:
            self.sound_defeated.play()
            reply = QMessageBox.question(self, 'You Lost!', 'Continue?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:  # 复位
            self.piece_now = BLACK
            self.mouse_point.setPixmap(self.black)
            self.step = 0
            for piece in self.pieces:
                piece.clear()
            self.chessboard.reset()
            self.update()
        else:
            self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GoBang()
    sys.exit(app.exec_())
