import os
import joblib
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


# ===============================
# 1. Đường dẫn dữ liệu
# ===============================

DATA_PATH = "nam/processed_data"
OUTPUT_PATH = "nguyen/model_results"

os.makedirs(OUTPUT_PATH, exist_ok=True)


# ===============================
# 2. Load dữ liệu từ phần của Nam
# ===============================

print("Đang đọc dữ liệu train/test...")

X_train = pd.read_csv(f"{DATA_PATH}/X_train.csv")
X_test = pd.read_csv(f"{DATA_PATH}/X_test.csv")
y_train = pd.read_csv(f"{DATA_PATH}/y_train.csv").values.ravel()
y_test = pd.read_csv(f"{DATA_PATH}/y_test.csv").values.ravel()

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ===============================
# 3. Xử lý imbalance
# ===============================

print("\nPhân bố nhãn trước khi cân bằng:")
print(pd.Series(y_train).value_counts())

# SMOTE: tăng mẫu lớp ít dữ liệu
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nPhân bố nhãn sau SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# UnderSampling: giảm mẫu lớp quá nhiều dữ liệu
under_sampler = RandomUnderSampler(random_state=42)

X_train_balanced, y_train_balanced = under_sampler.fit_resample(
    X_train_smote,
    y_train_smote
)

print("\nPhân bố nhãn sau UnderSampling:")
print(pd.Series(y_train_balanced).value_counts())


# ===============================
# 4. Khai báo 5 model
# ===============================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "SVM": LinearSVC(
        random_state=42,
        max_iter=5000
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Naive Bayes": GaussianNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}


# ===============================
# 5. Train + đánh giá model
# ===============================

results = []

report_file = f"{OUTPUT_PATH}/classification_reports.txt"

with open(report_file, "w", encoding="utf-8") as f:
    f.write("CLASSIFICATION REPORTS\n")
    f.write("=" * 80 + "\n\n")


for model_name, model in models.items():
    print(f"\n===============================")
    print(f"Đang train model: {model_name}")
    print(f"===============================")

    # Train model
    model.fit(X_train_balanced, y_train_balanced)

    # Dự đoán
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

    # Tính các chỉ số
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
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot(ax=ax, values_format="d", colorbar=False)

    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    file_name = model_name.lower().replace(" ", "_")
    cm_path = f"{OUTPUT_PATH}/confusion_matrix_{file_name}.png"

    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"Đã lưu confusion matrix: {cm_path}")

    # Lưu từng model
    model_path = f"{OUTPUT_PATH}/{file_name}.pkl"
    joblib.dump(model, model_path)


# ===============================
# 6. So sánh và chọn model tốt nhất
# ===============================

comparison_df = pd.DataFrame(results)

comparison_df = comparison_df.sort_values(
    by=["f1_macro", "recall_macro"],
    ascending=False
)

comparison_path = f"{OUTPUT_PATH}/model_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)

print("\nBảng so sánh model:")
print(comparison_df)

best_model_name = comparison_df.iloc[0]["model"]

print("\n===============================")
print("MODEL TỐT NHẤT")
print("===============================")
print(f"Model tốt nhất là: {best_model_name}")

# Lưu model tốt nhất thành best_model.pkl
best_model_file_name = best_model_name.lower().replace(" ", "_")
best_model = joblib.load(f"{OUTPUT_PATH}/{best_model_file_name}.pkl")

joblib.dump(best_model, f"{OUTPUT_PATH}/best_model.pkl")

print(f"Đã lưu model tốt nhất tại: {OUTPUT_PATH}/best_model.pkl")
print(f"Đã lưu classification report tại: {report_file}")
print(f"Đã lưu bảng so sánh tại: {comparison_path}")