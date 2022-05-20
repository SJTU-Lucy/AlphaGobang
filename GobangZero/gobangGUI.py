import sys
sys.path.append("../")
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMessageBox, QProgressBar
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QIcon, QPalette, QPainter
from alphazero.chess_board import ChessBoard
from GobangZero.aiThread import AIThread
import alphazero.common as common

EMPTY = common.empty
BLACK = common.black
WHITE = common.white
SIZE = common.size
WIDTH = common.width
HEIGHT = common.height
MARGIN = common.margin
GRID = common.grid
PIECE = common.piece
PATH = common.modelpath
BOARDPATH = common.boardpath


class GoBang(QWidget):
    def __init__(self):
        super().__init__()
        # 棋盘
        self.chessboard = ChessBoard(board_len=SIZE)

        # 设置背景
        palette = QPalette()
        palette.setBrush(self.backgroundRole(), QtGui.QBrush(QtGui.QPixmap(BOARDPATH)))
        self.setPalette(palette)

        # 鼠标变成手指形状
        self.setCursor(Qt.PointingHandCursor)

        # 设置尺寸，不可设置大小
        self.resize(WIDTH, HEIGHT)
        self.setMinimumSize(QtCore.QSize(WIDTH, HEIGHT))
        self.setMaximumSize(QtCore.QSize(WIDTH, HEIGHT))
        self.setWindowTitle("GoBangZero")
        self.setWindowIcon(QIcon('img/black.png'))

        # 棋子图像
        self.black = QPixmap('img/black.png')
        self.white = QPixmap('img/white.png')

        # 对局参数
        self.piece_now = BLACK
        self.ai_down = True
        self.step = 0
        self.x, self.y = 1000, 1000

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
            piece.setVisible(True)
            piece.setScaledContents(True)
        # 鼠标始终在最上层
        self.mouse_point.raise_()
        self.setMouseTracking(True)
        # 进度条，一开始不显示
        self.pgb = QProgressBar(self)
        self.pgb.move(0, self.width()-20)
        self.pgb.resize(250, 20)
        self.pgb.setMinimum(0)
        self.pgb.setMaximum(common.n_iters)
        self.pgb.setVisible(False)
        self.pgb.setVisible(0)
        self.show()

    # 在update时自动调用
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
            i, j = self.pixel_map(x, y)  # 对应棋盘坐标
            if not i is None and not j is None:  # 棋子落在棋盘上，排除边缘
                if self.chessboard.get(i, j) == EMPTY:
                    # 黑子下一步
                    self.ai_down = False
                    self.draw(i, j)
                    # 白子下一步
                    # 加载模型+进度条
                    self.AI = AIThread(self.chessboard, PATH)
                    self.pgb.setVisible(True)
                    self.pgb.setValue(0)
                    self.AI.finishSignal.connect(self.AI_draw)
                    self.AI.progresssignal.connect(self.AI_progress)
                    self.AI.start()

    def AI_progress(self, current):
        self.pgb.setValue(current)

    # AI线程结束信号的对应槽
    def AI_draw(self, i, j):
        if self.step != 0:
            self.pgb.setVisible(False)
            self.draw(i, j)  # AI
            self.x, self.y = self.map_pixel(i, j)
            self.ai_down = True
            self.update()

    # 在指定位置落子
    def draw(self, i, j):
        x, y = self.map_pixel(i, j)
        if self.piece_now == BLACK:
            self.pieces[self.step].setPixmap(self.black)
            self.piece_now = WHITE
            self.chessboard.do_action(i * SIZE + j)
        else:
            self.pieces[self.step].setPixmap(self.white)
            self.piece_now = BLACK
            self.chessboard.do_action(i * SIZE + j)
        self.pieces[self.step].setGeometry(x, y, PIECE, PIECE)
        self.step += 1

        is_end, winner = self.chessboard.is_game_over()  # 判断输赢
        if is_end:
            self.mouse_point.clear()
            self.gameover(winner)

    # 指示AI当前下的棋子
    def drawLines(self, qp):
        if self.step != 0:
            pen = QtGui.QPen(QtCore.Qt.black, 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(self.x - 5, self.y - 5, self.x + 3, self.y + 3)
            qp.drawLine(self.x + 3, self.y, self.x + 3, self.y + 3)
            qp.drawLine(self.x, self.y + 3, self.x + 3, self.y + 3)

    # 棋盘的逻辑坐标->像素坐标
    def map_pixel(self, i, j):
        return MARGIN + j * GRID - PIECE / 2, MARGIN + i * GRID - PIECE / 2

    # 像素坐标->逻辑坐标
    def pixel_map(self, x, y):
        i, j = int(round((y - MARGIN) / GRID)), int(round((x - MARGIN) / GRID))
        if i < 0 or i >= SIZE or j < 0 or j >= SIZE:
            return None, None
        else:
            return i, j

    # 游戏结束，判定胜负关系
    def gameover(self, winner):
        if winner == BLACK:
            reply = QMessageBox.question(self, 'You Win!', 'Continue?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        else:
            reply = QMessageBox.question(self, 'You Lost!', 'Continue?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.piece_now = BLACK
            self.ai_down = True
            self.step = 0
            for piece in self.pieces:
                piece.clear()
            self.mouse_point.setPixmap(self.black)
            self.chessboard.clear_board()
            self.update()
        else:
            self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GoBang()
    sys.exit(app.exec_())
