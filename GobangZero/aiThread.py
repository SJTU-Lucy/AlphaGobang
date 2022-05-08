import torch
from alpha_zero_mcts import AlphaZeroMCTS
from PyQt5.QtCore import QThread, pyqtSignal
from ai import searcher
import common

SIZE = common.size


class AIThread(QThread):
    finishSignal = pyqtSignal(int, int)

    def __init__(self, chessBoard, model=None, c_puct=5, n_iters=2000, is_use_gpu=True, parent=None):
        super().__init__(parent=parent)
        self.chessBoard = chessBoard
        self.c_puct = c_puct
        self.n_iters = n_iters
        self.isUseGPU = is_use_gpu
        self.device = torch.device('cuda:0' if self.isUseGPU else 'cpu')
        if model is not None:
            self.model = torch.load(model).to(self.device)
            self.model.set_device(is_use_gpu=self.isUseGPU)
            self.model.eval()
            self.mcts = AlphaZeroMCTS(self.model, c_puct, n_iters)
        else:
            self.model = None
            self.mcts = searcher()
            self.mcts.board = self.chessBoard

    def run(self):
        action = self.mcts.get_action(self.chessBoard)
        x = action // SIZE
        y = action % SIZE
        self.finishSignal.emit(x, y)
