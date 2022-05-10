from alphazero import PolicyValueNet, ChessBoard
import torch

device = torch.device('cuda:0')
net = PolicyValueNet().to(device)
chessboard = ChessBoard()
p, value = net.predict(chessboard)
torch.save(net, 'model/testnet.pth')