# coding: utf-8
from app.common import resource
from app.components.framelesswindow import FramelessWindow
from app.components.widgets.pop_up_ani_stacked_widget import PopUpAniStackedWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWinExtras import QtWin
from PyQt5.QtWidgets import QApplication, qApp


class MainWindow(FramelessWindow):
    def __init__(self, board_len=11):
        super().__init__(parent=None)
        self.boardLen = board_len
        self.stackedWidget = PopUpAniStackedWidget(self)
        # 初始化
        self.initWidget()

    def initWidget(self):
        """ 初始化界面 """
        self.resize(1250, 1350)
        self.setObjectName("mainWindow")
        self.setWindowTitle('Alpha Gobang Zero')
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowMinMaxButtonsHint)
        self.windowEffect.addWindowAnimation(self.winId())
        # 在去除任务栏的显示区域居中显示
        desktop = QApplication.desktop().availableGeometry()
        self.move(
            int(desktop.width() / 2 - self.width() / 2),
            int(desktop.height() / 2 - self.height() / 2),
        )
