from math import gcd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# ============================================================
# 1. CẤU HÌNH
# ============================================================

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

SAMPLE_RATE = 16000
FIR_ORDER = 40

# Ngưỡng quyết định người lạ.
# Cần điều chỉnh ngưỡng dựa trên dữ liệu validation.
UNKNOWN_THRESHOLD = 1.2

UNKNOWN_LABEL = "Không nằm trong 5 thành viên"

# Các tên thư mục được nhận diện là người lạ
UNKNOWN_FOLDER_NAMES = {
    "nguoila",
    "nguoi_la",
    "nguoi la",
    "unknown",
    "outsider",
}

AUDIO_EXTENSIONS = {".wav"}


# ============================================================
# 2. ĐỌC VÀ TIỀN XỬ LÝ ÂM THANH
# ============================================================

def load_audio(file_path):
    # Đọc file âm thanh
    audio, original_sr = sf.read(file_path)

    # Chuyển âm thanh stereo thành mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Xử lý các giá trị NaN và Inf
    audio = np.nan_to_num(audio.astype(np.float64))

    # Đưa âm thanh về tần số lấy mẫu 16 kHz
    if original_sr != SAMPLE_RATE:
        common = gcd(int(original_sr), SAMPLE_RATE)

        audio = resample_poly(
            audio,
            SAMPLE_RATE // common,
            int(original_sr) // common,
        )

    # Tính RMS để kiểm tra âm thanh im lặng
    rms = np.sqrt(np.mean(audio ** 2))

    # Đơn giản hóa Voice Activity Detection (VAD) để loại bỏ khoảng lặng
    # Chia thành các frame 20ms
    frame_length = int(SAMPLE_RATE * 0.02)
    
    if len(audio) > frame_length:
        # Số lượng frame hoàn chỉnh
        num_frames = len(audio) // frame_length
        audio_truncated = audio[:num_frames * frame_length]
        frames = audio_truncated.reshape(-1, frame_length)
        
        # Tính RMS cho từng frame
        frame_rms = np.sqrt(np.mean(frames**2, axis=1))
        
        # Lọc các frame có năng lượng lớn hơn 20% năng lượng trung bình
        threshold = np.mean(frame_rms) * 0.2
        active_frames = frames[frame_rms > threshold]
        
        if len(active_frames) > 0:
            active_audio = active_frames.flatten()
            # Giữ lại đoạn dư ở cuối
            active_audio = np.concatenate([active_audio, audio[num_frames * frame_length:]])
            
            # Chỉ dùng audio lọc được nếu độ dài đủ lớn
            if len(active_audio) > SAMPLE_RATE * 0.1:
                audio = active_audio

    # Chuẩn hóa âm lượng lại sau khi loại bỏ khoảng lặng
    rms_active = np.sqrt(np.mean(audio ** 2))
    return audio / (rms_active + 1e-10)


# ============================================================
# 3. TRÍCH XUẤT ĐẶC TRƯNG FIR
# ============================================================

def extract_fir_features(audio, order=FIR_ORDER):
    # Tín hiệu mục tiêu
    target = audio[order:]

    # Ma trận đầu vào gồm các mẫu trễ
    X = np.column_stack([
        audio[order - lag - 1: len(audio) - lag - 1]
        for lag in range(order)
    ])

    # Tìm các hệ số FIR bằng phương pháp bình phương tối thiểu
    coefficients = np.linalg.lstsq(
        X,
        target,
        rcond=None
    )[0]

    # Tính sai số bình phương trung bình
    mse = np.mean(
        (target - X @ coefficients) ** 2
    )

    return coefficients, mse


# ============================================================
# 4. ĐỌC DATASET TRAIN / TEST
# ============================================================

def load_dataset(folder):
    if not folder.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục: {folder}"
        )

    features = []
    labels = []
    files = []
    mses = []

    # Mỗi thư mục con tương ứng với một người nói
    for speaker_dir in sorted(
        p for p in folder.iterdir() if p.is_dir()
    ):
        folder_name = speaker_dir.name.strip().lower()

        # Nhận diện thư mục người lạ
        if folder_name in UNKNOWN_FOLDER_NAMES:
            label = UNKNOWN_LABEL
        else:
            label = speaker_dir.name

        for file_path in sorted(speaker_dir.rglob("*")):
            if (
                not file_path.is_file()
                or file_path.suffix.lower() not in AUDIO_EXTENSIONS
            ):
                continue

            audio = load_audio(file_path)
            coefficients, mse = extract_fir_features(audio)
            
            features.append(coefficients)
            labels.append(label)
            files.append(file_path)
            mses.append(mse)

    return (
        np.vstack(features),
        np.array(labels),
        files,
        np.array(mses),
    )


# ============================================================
# 5. HUẤN LUYỆN MÔ HÌNH 5 THÀNH VIÊN
# ============================================================

def train_speaker_model(X_train, y_train):
    # Loại bỏ dữ liệu người lạ khỏi quá trình tạo mô hình
    member_mask = y_train != UNKNOWN_LABEL

    X_members = X_train[member_mask]
    y_members = y_train[member_mask]

    speakers = np.unique(y_members)

    # Mô hình của mỗi người là vector đặc trưng trung bình
    speaker_models = {
        speaker: X_members[y_members == speaker].mean(axis=0)
        for speaker in speakers
    }

    return speaker_models

# ============================================================
# 6. DỰ ĐOÁN NGƯỜI NÓI
# ============================================================

def predict_speaker(
    features,
    speaker_models,
    threshold=UNKNOWN_THRESHOLD
):
    # Tính khoảng cách giữa file cần nhận dạng và từng thành viên
    distances = {
        speaker: np.linalg.norm(
            features - mean_features
        )
        for speaker, mean_features in speaker_models.items()
    }

    # Chọn thành viên có khoảng cách nhỏ nhất
    best_speaker = min(
        distances,
        key=distances.get
    )

    min_distance = distances[best_speaker]

    # Khoảng cách quá lớn thì từ chối nhận dạng
    if min_distance > threshold:
        return UNKNOWN_LABEL, min_distance

    return best_speaker, min_distance


# ============================================================
# 7. VẼ PHÂN BỐ ĐẶC TRƯNG SVD
# ============================================================

def plot_svd(X_train, y_train):
    # Chỉ sử dụng dữ liệu 5 thành viên để trực quan hóa
    member_mask = y_train != UNKNOWN_LABEL

    X_members = X_train[member_mask]
    y_members = y_train[member_mask]

    mean = X_members.mean(axis=0)

    U, S, _ = np.linalg.svd(
        X_members - mean,
        full_matrices=False
    )

    if len(S) < 2:
        print("Không đủ thành phần để vẽ SVD 2D.")
        return

    coordinates = U[:, :2] * S[:2]

    plt.figure(figsize=(8, 6))

    for speaker in np.unique(y_members):
        mask = y_members == speaker

        plt.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            label=speaker,
            alpha=0.8,
        )

    plt.title(
        "Phân bố vector FIR của 5 thành viên trên không gian SVD 2D"
    )
    plt.xlabel("Thành phần SVD 1")
    plt.ylabel("Thành phần SVD 2")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("svd_plot.png", dpi=200)
    # plt.show()


# ============================================================
# 8. ĐÁNH GIÁ MÔ HÌNH
# ============================================================

def evaluate_model(
    X_test,
    y_test,
    test_files,
    speaker_models
):
    y_pred = []
    
    for features in X_test:
        predicted, distance = predict_speaker(
            features,
            speaker_models
        )
        y_pred.append(predicted)

    y_true = y_test
    y_pred = np.array(y_pred)

    # Tính độ chính xác tổng thể
    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print(f"\nĐộ chính xác tổng thể: {accuracy:.2%}")

    # Confusion matrix gồm 5 thành viên và người lạ
    labels = sorted(speaker_models.keys()) + [
        UNKNOWN_LABEL
    ]

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )

    display.plot(
        cmap="Blues",
        values_format="d",
        xticks_rotation=30,
    )

    plt.title(
        "Confusion Matrix - Speaker Identification"
    )
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200)
    # plt.show()

    return accuracy, matrix


# ============================================================
# 9. CHƯƠNG TRÌNH CHÍNH
# ============================================================

def main():
    # Đọc dữ liệu train và test
    X_train, y_train, _, _ = load_dataset(TRAIN_DIR)

    X_test, y_test, test_files, _ = load_dataset(TEST_DIR)

    # Huấn luyện mô hình 5 thành viên
    speaker_models = train_speaker_model(
        X_train,
        y_train
    )

    # Vẽ phân bố đặc trưng của thành viên
    plot_svd(
        X_train,
        y_train
    )

    # Đánh giá bằng tập test
    evaluate_model(
        X_test,
        y_test,
        test_files,
        speaker_models
    )


if __name__ == "__main__":
    main()