import joblib
import pandas as pd
from datetime import datetime

# 1. Tải model đã được đổi tên thành model.pkl
model = joblib.load('model.pkl')

def simulate_ids(sample):
    # Lấy kết quả dự đoán (là con số)
    prediction = model.predict(sample)[0]
    
    # Bảng tra cứu nhãn (dựa trên LabelEncoder của nhóm)
    # Thông thường: 0 là Benign, các số khác là tấn công
    labels_map = {0: "Benign", 2: "DDoS", 14: "PortScan"} # Thêm các số khác nếu cần
    
    # Chuyển số thành tên để so sánh
    predict = labels_map.get(prediction, str(prediction))

    
    if predict != "Benign":
        print("[ALERT] Attack:", predict)
        
        with open("alerts.log", "a", encoding="utf-8") as f:
            time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{time_stamp}] [ALERT] Attack: {predict}\n")
    else:
        print("Traffic is Normal (Benign).")


if __name__ == "__main__":
    try:
        # Đọc dữ liệu từ thư mục của bạn Nam
        test_data = pd.read_csv('nam/processed_data/X_test.csv')
        
        print("--- Hệ thống IDS đang quét dữ liệu ---")
        # Quét thử 10 dòng đầu
        for i in range(10):
            sample_row = test_data.iloc[[i]]
            simulate_ids(sample_row)
            
    except Exception as e:
        print(f"Lỗi: {e}")