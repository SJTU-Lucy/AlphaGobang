from alphazero.train import TrainModel
import matplotlib.pyplot as plt

train_model = TrainModel()
train_model.train()
loss_record = train_model.loss_record
plt.plot(loss_record)
plt.show()
print("总更新次数=", train_model.updatecount)
