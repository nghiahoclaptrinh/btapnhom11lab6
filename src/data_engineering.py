import pandas as pd
import numpy as np
import glob
import os

def reduce_mem_usage(df):
    """Xử lý downcast các kiểu dữ liệu để giảm RAM."""
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'  > Memory usage ban đầu: {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32) # Giữ float32 để an toàn cho Machine Learning
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'  > Memory usage sau khi tối ưu: {end_mem:.2f} MB')
    print(f'  > Đã tiết kiệm {100 * (start_mem - end_mem) / start_mem:.1f}% RAM!')
    return df

def clean_data(df):
    """Xử lý làm sạch sâu NIDS: Xóa trùng lặp, xóa cột vô dụng, điền giá trị thiếu."""
    print("\n--- QUÁ TRÌNH TIỀN XỬ LÝ (CLEAN DATA) ---")
    
    print("  1. Xóa các dòng trùng lặp (Drop Duplicates)...")
    before_len = len(df)
    df.drop_duplicates(inplace=True)
    print(f"     -> Đã xóa {before_len - len(df)} dòng trùng lặp.")
    
    print("  2. Chuẩn hóa chữ thường và khoảng trắng tên cột...")
    df.columns = df.columns.str.strip().str.lower()
    
    print("  3. Loại bỏ các dòng bị lỗi cấu trúc (âm vô lý)...")
    neg_cols = ['init fwd win byts', 'init bwd win byts']
    for col in neg_cols:
        if col in df.columns:
            before_neg = len(df)
            df = df[df[col] >= 0]
            if before_neg > len(df):
                print(f"     -> Đã xóa {before_neg - len(df)} dòng giá trị rác âm tại nhóm cột {col}.")
    
    print("  4. Xóa các cột định danh hệ thống (Timestamp, Dst Port) không có giá trị học máy...")
    cols_to_drop = ['timestamp', 'dst port']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True, errors="ignore")
    
    print("  5. Xử lý giá trị Infinite (Vô cực) và Missing (Thiếu)...")
    # Thay inf bằng nan rồi điền bằng 0 (giữ nguyên tổng thể mà không mất dòng gói tin)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    print("  6. Xóa các cột Constant (Chỉ mang 1 giá trị duy nhất)...")
    label_col = 'label'
    if label_col in df.columns:
        nunique = df.drop(columns=[label_col]).nunique()
    else:
        nunique = df.nunique()
    
    constant_cols = nunique[nunique == 1].index.tolist()
    if constant_cols:
        print(f"     -> Tìm thấy {len(constant_cols)} cột vô giá trị. Bao gồm: {constant_cols[:3]}...")
        df.drop(columns=constant_cols, inplace=True)
    
    return df

def process_data(input_dir, output_file):
    """Đọc toàn bộ file CSV cung cấp, gộp lại và xử lý."""
    print("=== NGƯỜI 1: BẮT ĐẦU DATA ENGINEERING MODULE ===")
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if len(all_files) == 0:
        print(f"LỖI: Không tìm thấy file CSV nào trong {input_dir}.")
        print("Hãy copy các file CSV dữ liệu NIDS thô vào thư mục này và chạy lại.")
        return
        
    df_list = []
    for f in all_files:
        print(f"  > Đang tải: {os.path.basename(f)}")
        # Cảnh báo dtype có thể xuất hiện, nên dùng low_memory=False
        df_list.append(pd.read_csv(f, low_memory=False))
        
    print(f"\nĐang ráp các phân mảnh với nhau...")
    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"Kích thước tệp tổng hợp: {merged_df.shape}")
    
    merged_df = clean_data(merged_df)
    
    print("\n--- QUÁ TRÌNH TỐI ƯU HÓA TÀI NGUYÊN (DOWNCAST RAM) ---")
    merged_df = reduce_mem_usage(merged_df)
    
    print(f"\n[OK] Lưu Dữ liệu Sạch và Nhẹ vào: {output_file}")
    merged_df.to_csv(output_file, index=False)
    print("=== HOÀN THÀNH NHIỆM VỤ NGƯỜI 1 ===")

if __name__ == "__main__":
    RAW_DATA_DIR = "../data/raw" 
    OUTPUT_FILE = "../data/cleaned_data.csv"
    
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    process_data(RAW_DATA_DIR, OUTPUT_FILE)
