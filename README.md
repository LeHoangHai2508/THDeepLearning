📋 Mục lục

✨ Tính năng

🗂️ Cấu trúc dự án

⚙️ Yêu cầu hệ thống

🚀 Hướng dẫn cài đặt

🗃️ Chuẩn bị dữ liệu

🎬 Cách sử dụng

🏋️ Huấn luyện

🔍 Đánh giá

🤖 Dự đoán (Inference)

📦 Mô hình đã huấn luyện sẵn

📈 Kết quả

🤝 Đóng góp

📜 Giấy phép

✨ Tính năng

🎯 Kiến trúc CNN tùy chỉnh dành riêng cho nhận diện vân tay

🔄 Pipeline huấn luyện và đánh giá tập trung trong cnn/

🗂️ Mô hình được huấn luyện sẵn (fingerprint_model.pth)

📊 Đồ thị so sánh Accuracy & Loss giữa huấn luyện và kiểm thử (training_test_chart.png)

🗂️ Cấu trúc dự án

THDeepLearning/
├── cnn/                   # Mã nguồn mô hình, huấn luyện, đánh giá
│   ├── model.py           # Định nghĩa kiến trúc CNN
│   ├── train.py           # Script huấn luyện
│   ├── test.py            # Script đánh giá
│   ├── predict.py         # Script dự đoán ảnh đơn
│   └── utils.py           # Data loader & hàm hỗ trợ
├── data/
│   └── SOCOFing/          # Nội dung gốc của SOCOFing (tải ngoài)
├── fingerprint_dataset/   # Dữ liệu đã xử lý sẵn cho huấn luyện/kiểm thử
├── fingerprint_model.pth  # Weights mô hình đã huấn luyện
├── training_test_chart.png# Đồ thị Accuracy/Loss
└── README.md              # Hướng dẫn sử dụng

⚙️ Yêu cầu hệ thống

Hệ điều hành: Windows, macOS, hoặc Linux

Python: 3.8+

Thư viện:

torch (>=1.7)

torchvision

numpy

pandas

matplotlib

opencv-python

scikit-learn

TIP: Nên sử dụng virtual environment để cô lập phụ thuộc.

🚀 Hướng dẫn cài đặt

Clone repo

git clone https://github.com/LeHoangHai2508/THDeepLearning.git
cd THDeepLearning

Tạo & kích hoạt môi trường ảo

python3 -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

Cài đặt phụ thuộc

pip install --upgrade pip
pip install -r requirements.txt

Nếu không có requirements.txt, bạn có thể cài thủ công:

pip install torch torchvision numpy pandas matplotlib opencv-python scikit-learn

🗃️ Chuẩn bị dữ liệu

Tải bộ SOCOFing từ trang chính thức:

https://seu-software.github.io/SOCOFING/

Giải nén vào thư mục data/SOCOFing/

Xử lý (nếu có):

python cnn/utils.py --input_dir data/SOCOFing --output_dir fingerprint_dataset

Nếu không sử dụng script, đảm bảo thư mục fingerprint_dataset/ chứa ảnh phân theo thư mục lớp.

🎬 Cách sử dụng

🏋️ Huấn luyện

python cnn/train.py \
  --data_dir fingerprint_dataset \
  --epochs 50 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --save_path fingerprint_model.pth

--epochs: Số vòng lặp

--batch_size: Kích thước batch

--learning_rate: Tốc độ học

--save_path: Đường dẫn lưu weights

🔍 Đánh giá

python cnn/test.py \
  --model_path fingerprint_model.pth \
  --data_dir fingerprint_dataset

🤖 Dự đoán (Inference)

python cnn/predict.py \
  --model_path fingerprint_model.pth \
  --image_path path/to/fingerprint.png

📦 Mô hình đã huấn luyện sẵn

Bạn có thể tải fingerprint_model.pth từ Releases để sử dụng ngay.
