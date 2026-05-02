import os
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==================================================

# ROOT_DIR là thư mục gốc project: btapnhom11lab6
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "nam" / "processed_data"
OUTPUT_PATH = ROOT_DIR / "nguyen" / "model_results"

os.makedirs(OUTPUT_PATH, exist_ok=True)


# ==================================================
# 2. CẤU HÌNH GIẢM DỮ LIỆU ĐỂ TRÁNH TRÀN RAM
# ==================================================

# Vì dataset rất lớn, chỉ lấy tối đa từng này mẫu mỗi class để train
# Nếu máy mạnh có thể tăng lên 3000, 5000
MAX_TRAIN_PER_CLASS = 2000

# Số mẫu mỗi class dùng để test
MAX_TEST_PER_CLASS = 500

RANDOM_STATE = 42


# ==================================================
# 3. HÀM GIỚI HẠN SỐ MẪU MỖI CLASS
# ==================================================

def limit_samples_per_class(X, y, max_per_class, random_state=42):
    """
    Giới hạn số lượng mẫu của mỗi class để tránh dataset quá lớn.

    Ví dụ:
    Nếu một class có 1.600.000 mẫu, chỉ lấy 2.000 mẫu.
    Nếu một class có 500 mẫu, giữ nguyên 500 mẫu.
    """

    rng = np.random.default_rng(random_state)
    selected_indices = []

    y_array = np.array(y)

    for label in np.unique(y_array):
        label_indices = np.where(y_array == label)[0]
        n_samples = min(len(label_indices), max_per_class)

        chosen_indices = rng.choice(
            label_indices,
            size=n_samples,
            replace=False
        )

        selected_indices.extend(chosen_indices)

    rng.shuffle(selected_indices)

    X_small = X.iloc[selected_indices].reset_index(drop=True)
    y_small = y_array[selected_indices]

    return X_small, y_small


# ==================================================
# 4. HÀM XỬ LÝ IMBALANCE
# ==================================================

def handle_imbalance(X_train, y_train):
    """
    Xử lý mất cân bằng dữ liệu bằng:
    1. RandomUnderSampler: giảm class quá nhiều mẫu.
    2. SMOTE: tăng class ít mẫu.

    Nếu dữ liệu đã cân bằng thì không sinh thêm mẫu để tránh tốn RAM.
    """

    print("\nPhân bố nhãn trước khi cân bằng:")
    print(pd.Series(y_train).value_counts().sort_index())

    # Bước 1: UnderSampling
    # Sau khi đã giới hạn dữ liệu, bước này vẫn được giữ để đúng yêu cầu bài.
    print("\nĐang áp dụng UnderSampling...")

    under_sampler = RandomUnderSampler(
        random_state=RANDOM_STATE
    )

    X_under, y_under = under_sampler.fit_resample(X_train, y_train)

    print("\nPhân bố nhãn sau UnderSampling:")
    print(pd.Series(y_under).value_counts().sort_index())

    # Bước 2: Kiểm tra có cần SMOTE không
    class_counts = pd.Series(y_under).value_counts()

    if class_counts.max() == class_counts.min():
        print("\nDữ liệu đã cân bằng sau UnderSampling.")
        print("Bỏ qua SMOTE để tránh sinh thêm dữ liệu không cần thiết.")
        return X_under, y_under

    # Nếu còn lệch thì dùng SMOTE để tăng lớp ít mẫu
    print("\nĐang áp dụng SMOTE...")

    counts = Counter(y_under)
    max_count = max(counts.values())

    # Chỉ SMOTE cho class có ít hơn max_count và có ít nhất 2 mẫu
    smote_strategy = {
        label: max_count
        for label, count in counts.items()
        if count < max_count and count >= 2
    }

    if not smote_strategy:
        print("Không thể áp dụng SMOTE vì class thiểu số có quá ít mẫu.")
        return X_under, y_under

    min_class_for_smote = min(counts[label] for label in smote_strategy.keys())
    k_neighbors = min(3, min_class_for_smote - 1)

    smote = SMOTE(
        sampling_strategy=smote_strategy,
        random_state=RANDOM_STATE,
        k_neighbors=k_neighbors
    )

    X_balanced, y_balanced = smote.fit_resample(X_under, y_under)

    print("\nPhân bố nhãn sau SMOTE:")
    print(pd.Series(y_balanced).value_counts().sort_index())

    return X_balanced, y_balanced


# ==================================================
# 5. HÀM TẠO TÊN FILE AN TOÀN
# ==================================================

def safe_filename(model_name):
    return model_name.lower().replace(" ", "_").replace("/", "_")


# ==================================================
# 6. LOAD DỮ LIỆU
# ==================================================

print("Đang đọc dữ liệu train/test...")

x_train_path = DATA_PATH / "X_train.csv"
x_test_path = DATA_PATH / "X_test.csv"
y_train_path = DATA_PATH / "y_train.csv"
y_test_path = DATA_PATH / "y_test.csv"

required_files = [
    x_train_path,
    x_test_path,
    y_train_path,
    y_test_path
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {file_path}\n"
            "Hãy kiểm tra lại thư mục nam/processed_data đã có đủ "
            "X_train.csv, X_test.csv, y_train.csv, y_test.csv chưa."
        )

X_train = pd.read_csv(x_train_path)
X_test = pd.read_csv(x_test_path)
y_train = pd.read_csv(y_train_path).values.ravel()
y_test = pd.read_csv(y_test_path).values.ravel()

print("Dữ liệu ban đầu:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ==================================================
# 7. GIẢM DỮ LIỆU ĐỂ CHẠY ĐƯỢC TRÊN MÁY CÁ NHÂN
# ==================================================

print("\nĐang giảm dữ liệu để tránh tràn RAM...")

# Chuyển float64 sang float32 để giảm RAM
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train, y_train = limit_samples_per_class(
    X_train,
    y_train,
    MAX_TRAIN_PER_CLASS,
    random_state=RANDOM_STATE
)

X_test, y_test = limit_samples_per_class(
    X_test,
    y_test,
    MAX_TEST_PER_CLASS,
    random_state=RANDOM_STATE
)

print("\nDữ liệu sau khi giảm:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ==================================================
# 8. XỬ LÝ IMBALANCE
# ==================================================

X_train_balanced, y_train_balanced = handle_imbalance(X_train, y_train)

print("\nDữ liệu dùng để train model:")
print("X_train_balanced:", X_train_balanced.shape)
print("y_train_balanced:", y_train_balanced.shape)


# ==================================================
# 9. KHAI BÁO 5 MODEL
# ==================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ),

    "SVM": LinearSVC(
        random_state=RANDOM_STATE,
        max_iter=5000
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Naive Bayes": GaussianNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=50,
        max_depth=20,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


# ==================================================
# 10. TRAIN VÀ ĐÁNH GIÁ MODEL
# ==================================================

results = []

report_file = OUTPUT_PATH / "classification_reports.txt"

with open(report_file, "w", encoding="utf-8") as f:
    f.write("CLASSIFICATION REPORTS\n")
    f.write("=" * 80 + "\n\n")


for model_name, model in models.items():
    print("\n===============================")
    print(f"Đang train model: {model_name}")
    print("===============================")

    # Train model
    model.fit(X_train_balanced, y_train_balanced)

    # Dự đoán trên tập test
    y_pred = model.predict(X_test)

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    print(report)

    with open(report_file, "a", encoding="utf-8") as f:
        f.write(f"MODEL: {model_name}\n")
        f.write("-" * 80 + "\n")
        f.write(report)
        f.write("\n\n")

    # Tính chỉ số đánh giá
    accuracy = accuracy_score(y_test, y_pred)

    precision_macro = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro
    })

    # Confusion matrix
    labels = np.unique(np.concatenate([y_test, y_pred]))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    display.plot(
        ax=ax,
        values_format="d",
        colorbar=False,
        xticks_rotation=90
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    file_name = safe_filename(model_name)
    cm_path = OUTPUT_PATH / f"confusion_matrix_{file_name}.png"

    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"Đã lưu confusion matrix: {cm_path}")

    # Lưu từng model
    model_path = OUTPUT_PATH / f"{file_name}.pkl"
    joblib.dump(model, model_path)

    print(f"Đã lưu model: {model_path}")


# ==================================================
# 11. SO SÁNH VÀ CHỌN MODEL TỐT NHẤT
# ==================================================

comparison_df = pd.DataFrame(results)

comparison_df = comparison_df.sort_values(
    by=["f1_macro", "recall_macro"],
    ascending=False
)

comparison_path = OUTPUT_PATH / "model_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)

print("\nBảng so sánh model:")
print(comparison_df)

best_model_name = comparison_df.iloc[0]["model"]

print("\n===============================")
print("MODEL TỐT NHẤT")
print("===============================")
print(f"Model tốt nhất là: {best_model_name}")

# Lưu model tốt nhất thành best_model.pkl
best_model_file_name = safe_filename(best_model_name)
best_model = joblib.load(OUTPUT_PATH / f"{best_model_file_name}.pkl")

best_model_path = OUTPUT_PATH / "best_model.pkl"
joblib.dump(best_model, best_model_path)

print(f"\nĐã lưu model tốt nhất tại: {best_model_path}")
print(f"Đã lưu classification report tại: {report_file}")
print(f"Đã lưu bảng so sánh tại: {comparison_path}")
print("Hoàn thành phần Machine Learning Engineer.")