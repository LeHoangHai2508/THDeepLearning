import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from cnn.cnn_model import CNN

# ==== CẤU HÌNH ====
model_path = 'fingerprint_model.pth'
image_path = 'cnn/testimages/3__M_Right_ring_finger.BMP'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== TIỀN XỬ LÝ ====
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ==== LOAD CLASS ====
train_dataset = ImageFolder('./fingerprint_dataset/train')
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

# ==== LOAD MODEL ====
model = CNN(num_classes=len(idx_to_class))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ==== DỰ ĐOÁN ====
def predict_fingerprint(image_path):
    try:
        image = Image.open(image_path).convert('L')
    except Exception as e:
        print(f"Lỗi ảnh: {e}")
        return "Không thể mở ảnh"

    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(1).item()
        class_name = idx_to_class[pred]
        return class_name

# ==== GIAO DIỆN TKINTER ====
def browse_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if file_path:
        img = Image.open(file_path).convert("L").resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk

        result = predict_fingerprint(file_path)
        result_label.config(text=f"→ Đây là vân tay của: {result}")

# ==== CỬA SỔ ====
root = tk.Tk()
root.title("Nhận dạng vân tay")
root.geometry("400x450")

tk.Label(root, text="Hệ thống nhận dạng vân tay", font=("Arial", 16)).pack(pady=10)
btn = tk.Button(root, text="Chọn ảnh vân tay", command=browse_image, font=("Arial", 12))
btn.pack(pady=10)
panel = tk.Label(root)
panel.pack()
result_label = tk.Label(root, text="", font=("Arial", 14), fg="blue")
result_label.pack(pady=20)

root.mainloop()