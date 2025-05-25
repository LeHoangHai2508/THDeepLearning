#cnn/train.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from cnn.cnn_model import CNN
from tqdm import tqdm

if __name__ == '__main__':
    # ==== Cấu hình ====
    data_dir = './fingerprint_dataset'
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    num_classes = 2
    batch_size = 16
    num_epochs = 30
    learning_rate = 0.0005
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    # ==== Tiền xử lý ====
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # ==== Dataset & DataLoader ====
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    print(f"📂 Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # ==== Model ====
    model = CNN(num_classes=num_classes).to(device)

    # ==== Loss & Optimizer ====
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # ==== Huấn luyện ====
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = correct / len(train_dataset)
        train_losses.append(total_loss)
        train_accuracies.append(train_acc)

        # ==== Đánh giá ====
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                test_correct += (outputs.argmax(1) == labels).sum().item()
        test_acc = test_correct / len(test_dataset)
        test_accuracies.append(test_acc)

        scheduler.step()

        print(f"[{epoch+1:>2}/{num_epochs}] "
              f"Train Loss: {total_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    # ==== Lưu mô hình ====
    torch.save(model.state_dict(), 'fingerprint_model.pth')
    print("✅ Mô hình đã được lưu vào: fingerprint_model.pth")

    # ==== Biểu đồ ====
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss', color='red', marker='o')
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='blue', marker='x')
    plt.plot(epochs, test_accuracies, label='Test Accuracy', color='green', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Loss, Training Accuracy & Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_test_chart.png')
    plt.show()
