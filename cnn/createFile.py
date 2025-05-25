import os
import shutil
import random
from pathlib import Path

# Đường dẫn đến SOCOFing/Real sau khi giải nén
DATASET_DIR = "data/SOCOFing/Real"
OUTPUT_DIR = "fingerprint_dataset"
SPLIT_RATIO = [0.7, 0.15, 0.15]  # train, val, test

# Lấy danh sách ảnh
images = list(Path(DATASET_DIR).glob("*.bmp"))
random.shuffle(images)

# Tách dữ liệu
n_total = len(images)
n_train = int(n_total * SPLIT_RATIO[0])
n_val = int(n_total * SPLIT_RATIO[1])

splits = {
    'train': images[:n_train],
    'val': images[n_train:n_train + n_val],
    'test': images[n_train + n_val:]
}

# Tạo thư mục đích
for split in splits:
    for img_path in splits[split]:
        # Tên ảnh: 101_1_LF.bmp → lấy nhãn = LF
        label = img_path.stem.split('_')[2]
        dest_dir = Path(OUTPUT_DIR) / split / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(img_path, dest_dir / img_path.name)

print("✅ Done splitting and organizing the dataset.")