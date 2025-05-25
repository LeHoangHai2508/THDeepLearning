#cnn/chart.py
import matplotlib.pyplot as plt

# Giả lập dữ liệu 50 epoch
epochs = list(range(1, 51))
train_losses = [2.0 / (epoch ** 0.5) for epoch in epochs]  # Loss giảm dần
train_accuracies = [0.5 + 0.5 * (epoch / 50) for epoch in epochs]  # Accuracy tăng dần

# Vẽ biểu đồ
fig, ax1 = plt.subplots()

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='red')
ax1.plot(epochs, train_losses, color='red', label='Training Loss')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='blue')
ax2.plot(epochs, train_accuracies, color='blue', label='Training Accuracy')
ax2.tick_params(axis='y', labelcolor='blue')

plt.title('Training Loss and Accuracy over Epochs')
plt.tight_layout()
plt.savefig('training_chart.png')  # Lưu biểu đồ thành file ảnh
plt.show()
