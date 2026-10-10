# Nhận dạng người nói bằng đặc trưng FIR

Project nhận dạng người nói (Speaker Identification) sử dụng bộ lọc FIR
trích xuất từ tín hiệu âm thanh, kết hợp khoảng cách Euclidean để phân
loại 5 thành viên và phát hiện người lạ.

## Tính năng

- Đọc file âm thanh `.wav`, tự động chuyển về mono và resample về 16 kHz.
- Loại bỏ khoảng lặng đơn giản bằng RMS theo frame 20 ms (VAD).
- Chuẩn hóa âm lượng sau khi tiền xử lý.
- Trích xuất đặc trưng FIR bằng phương pháp bình phương tối thiểu
  (`np.linalg.lstsq`), trả về vector hệ số và MSE.
- Huấn luyện mô hình 5 thành viên bằng vector đặc trưng trung bình.
- Dự đoán người nói dựa trên khoảng cách Euclidean nhỏ nhất.
- Từ chối nhận dạng nếu khoảng cách vượt ngưỡng → gán nhãn "Không nằm
  trong 5 thành viên".
- Trực quan hóa phân bố đặc trưng bằng SVD 2D.
- Đánh giá bằng accuracy và confusion matrix.

## Cấu trúc thư mục
```text
XLTH/
├── main.py                  # Script chính
├── requirements.txt         # Danh sách thư viện
├── README.md                # Tài liệu này
├── .gitignore               # Loại trừ file không cần thiết
├── data/
│   ├── train/               # Dữ liệu huấn luyện
│   │   ├── <tên_người_1>/
│   │   │   ├── file1.wav
│   │   │   └── file2.wav
│   │   ├── <tên_người_2>/
│   │   └── ...
│   └── test/                # Dữ liệu kiểm thử
│       ├── <tên_người_1>/
│       ├── ...
│       └── nguoila/         # Thư mục chứa người lạ
├── svd_plot.png             # (tự sinh) Biểu đồ SVD
└── confusion_matrix.png     # (tự sinh) Ma trận nhầm lẫn
```

### Quy ước đặt tên thư mục

- Mỗi thư mục con trong `data/train/` và `data/test/` tương ứng một
  người nói.
- Tên thư mục được dùng làm nhãn.
- Các thư mục sau được coi là **người lạ**:
  `nguoila`, `nguoi_la`, `nguoi la`, `unknown`, `outsider`.
- Chỉ hỗ trợ file `.wav`.

## Yêu cầu

- Python 3.9+
- Các thư viện trong `requirements.txt`:
  - numpy
  - scipy
  - scikit-learn
  - matplotlib
  - soundfile


