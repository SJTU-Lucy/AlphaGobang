import sys
sys.path.append("../")
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from alphazero.chess_board import ChessBoard
from alphazero.alpha_zero_mcts import AlphaZeroMCTS
import alphazero.common as common

SIZE = common.size


class AIThread(QThread):
    finishSignal = pyqtSignal(int, int)
    progresssignal = pyqtSignal(int)

    def __init__(self, chessBoard: ChessBoard, model=None, c_puct=1, n_iters=2000, is_use_gpu=True, parent=None):
        super().__init__(parent=parent)
        self.chessBoard = chessBoard
        self.c_puct = c_puct
        self.n_iters = n_iters
        self.isUseGPU = is_use_gpu
        self.device = torch.device('cuda:0' if self.isUseGPU else 'cpu')
        self.model = torch.load(model).to(self.device)
        self.model.set_device(is_use_gpu=self.isUseGPU)
        self.model.eval()
        self.mcts = AlphaZeroMCTS(self.model, c_puct=c_puct, n_iters=n_iters, progesssignal=self.progresssignal)

    def run(self):
        action = self.mcts.get_action(self.chessBoard)
        x = action // SIZE
        y = action % SIZE
        self.finishSignal.emit(x, y)
