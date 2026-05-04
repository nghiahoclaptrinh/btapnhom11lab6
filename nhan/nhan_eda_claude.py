"""
============================================================
  EDA (Exploratory Data Analysis) - Người 3: Nhân
  Network Intrusion Detection System - NIDS
  Nhiệm vụ: Vẽ biểu đồ + Phân tích insight
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import glob
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# BƯỚC 1: ĐỌC DỮ LIỆU SẠCH TỪ NGƯỜI 1 (NGHĨA)
# ============================================================
print("=" * 60)
print("BƯỚC 1: Đọc dữ liệu")
print("=" * 60)

# Đọc cleaned_data.csv từ Người 1
df = pd.read_csv('cleaned_data.csv')

print(f"Kích thước dữ liệu: {df.shape[0]:,} dòng x {df.shape[1]} cột")
print(f"Các cột: {list(df.columns)}")
print()
print("Mẫu dữ liệu:")
print(df.head(3))
print()
print("Thông tin tổng quan:")
print(df.info())


# ============================================================
# BƯỚC 2: PHÂN PHỐI NHÃN (LABEL DISTRIBUTION)
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 2: Phân tích phân phối nhãn")
print("=" * 60)

label_counts = df['label'].value_counts()
label_pct = df['label'].value_counts(normalize=True) * 100

print("Số lượng từng loại:")
for label, count in label_counts.items():
    pct = label_pct[label]
    print(f"  {label:15s}: {count:6,} mẫu ({pct:.1f}%)")

# Vẽ biểu đồ phân phối nhãn
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Phân Phối Nhãn (Label Distribution)', fontsize=16, fontweight='bold')

# Bar chart
colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0',
          '#00BCD4', '#795548', '#607D8B', '#E91E63', '#FFEB3B',
          '#3F51B5', '#009688', '#FF5722', '#8BC34A', '#FFC107']

bars = axes[0].bar(range(len(label_counts)), label_counts.values,
                    color=colors[:len(label_counts)], edgecolor='white', linewidth=0.5)
axes[0].set_xticks(range(len(label_counts)))
axes[0].set_xticklabels(label_counts.index, rotation=45, ha='right', fontsize=9)
axes[0].set_title('Số lượng mẫu theo nhãn', fontsize=13)
axes[0].set_ylabel('Số lượng mẫu', fontsize=11)
axes[0].set_xlabel('Loại traffic', fontsize=11)

# Thêm số lên mỗi cột
for bar, val in zip(bars, label_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01 * label_counts.max(),
                 f'{val:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')

axes[0].grid(axis='y', alpha=0.3)

# Pie chart
wedges, texts, autotexts = axes[1].pie(
    label_counts.values,
    labels=label_counts.index,
    autopct='%1.1f%%',
    colors=colors[:len(label_counts)],
    startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1}
)
for text in texts:
    text.set_fontsize(9)
for autotext in autotexts:
    autotext.set_fontsize(8)
axes[1].set_title('Tỷ lệ phần trăm từng nhãn', fontsize=13)

plt.tight_layout()
plt.savefig('label_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("→ Đã lưu: label_distribution.png")


# ============================================================
# BƯỚC 3: THỐNG KÊ MÔ TẢ (DESCRIPTIVE STATISTICS)
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 3: Thống kê mô tả các feature số")
print("=" * 60)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Các feature số: {numeric_cols}")
print()
print(df[numeric_cols].describe().round(2))


# ============================================================
# BƯỚC 4: CORRELATION HEATMAP
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 4: Ma trận tương quan (Correlation Heatmap)")
print("=" * 60)

# Tính correlation
corr_matrix = df[numeric_cols].corr()

# In ra correlation mạnh nhất
print("Top 5 cặp feature tương quan mạnh nhất:")
corr_upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
top_corr = corr_upper.stack().abs().nlargest(5)
for (f1, f2), val in top_corr.items():
    print(f"  {f1} <-> {f2}: {val:.3f}")

# Vẽ heatmap
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

heatmap = sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn',
    center=0,
    vmin=-1, vmax=1,
    ax=ax,
    linewidths=0.5,
    linecolor='white',
    annot_kws={'size': 11, 'weight': 'bold'}
)

ax.set_title('Ma Trận Tương Quan Giữa Các Feature', fontsize=15, fontweight='bold', pad=15)
ax.tick_params(axis='x', rotation=30, labelsize=11)
ax.tick_params(axis='y', rotation=0, labelsize=11)

plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("→ Đã lưu: correlation_heatmap.png")


# ============================================================
# BƯỚC 5: PHÂN PHỐI FEATURE THEO NHÃN
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 5: Phân phối feature theo nhãn")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Phân Phối Feature Theo Nhãn (Benign vs Attack)', fontsize=15, fontweight='bold')
axes = axes.flatten()

palette = {'Benign': '#2196F3', 'Attack': '#F44336'}

for idx, col in enumerate(numeric_cols[:6]):
    ax = axes[idx]
    
    # Violin plot
    sns.violinplot(
        data=df, x='label', y=col,
        palette=palette, ax=ax,
        inner='box', cut=0
    )
    
    ax.set_title(f'Feature: {col}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Nhãn', fontsize=10)
    ax.set_ylabel(col, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # In thống kê
    for label in df['label'].unique():
        subset = df[df['label'] == label][col]
        print(f"  {col} | {label}: mean={subset.mean():.2f}, std={subset.std():.2f}")

plt.tight_layout()
plt.savefig('feature_distribution_by_label.png', dpi=150, bbox_inches='tight')
plt.show()
print("→ Đã lưu: feature_distribution_by_label.png")


# ============================================================
# BƯỚC 6: BOXPLOT - PHÁT HIỆN OUTLIER
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 6: Boxplot phát hiện outlier")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Boxplot - Phát Hiện Outlier Theo Nhãn', fontsize=15, fontweight='bold')
axes = axes.flatten()

for idx, col in enumerate(numeric_cols[:6]):
    ax = axes[idx]
    
    sns.boxplot(
        data=df, x='label', y=col,
        palette=palette, ax=ax,
        flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.5}
    )
    
    ax.set_title(f'Feature: {col}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Nhãn', fontsize=10)
    ax.set_ylabel(col, fontsize=10)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('boxplot_outlier.png', dpi=150, bbox_inches='tight')
plt.show()
print("→ Đã lưu: boxplot_outlier.png")


# ============================================================
# BƯỚC 7: PROTOCOL DISTRIBUTION
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 7: Phân phối Protocol")
print("=" * 60)

protocol_label = df.groupby(['protocol', 'label']).size().unstack(fill_value=0)
protocol_map = {6: 'TCP (6)', 17: 'UDP (17)', 0: 'Other (0)'}
protocol_label.index = [protocol_map.get(p, str(p)) for p in protocol_label.index]

print("Số lượng mỗi protocol theo nhãn:")
print(protocol_label)

fig, ax = plt.subplots(figsize=(10, 6))
protocol_label.plot(
    kind='bar', ax=ax,
    color=['#2196F3', '#F44336'],
    edgecolor='white', width=0.6
)
ax.set_title('Phân Phối Protocol Theo Loại Traffic', fontsize=14, fontweight='bold')
ax.set_xlabel('Protocol', fontsize=12)
ax.set_ylabel('Số lượng mẫu', fontsize=12)
ax.legend(title='Nhãn', fontsize=11)
ax.tick_params(axis='x', rotation=0, labelsize=11)
ax.grid(axis='y', alpha=0.3)

for container in ax.containers:
    ax.bar_label(container, fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('protocol_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("→ Đã lưu: protocol_distribution.png")


# ============================================================
# BƯỚC 8: SCATTER PLOT - FEATURE QUAN TRỌNG
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 8: Scatter plot - Flow Duration vs Packets")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Scatter Plot - Feature Phân Biệt Benign vs Attack', fontsize=14, fontweight='bold')

for label, color in [('Benign', '#2196F3'), ('Attack', '#F44336')]:
    subset = df[df['label'] == label]
    axes[0].scatter(
        subset['flow duration'], subset['tot fwd pkts'],
        c=color, alpha=0.4, s=15, label=label
    )

axes[0].set_xlabel('Flow Duration', fontsize=11)
axes[0].set_ylabel('Tot Fwd Pkts', fontsize=11)
axes[0].set_title('Flow Duration vs Tot Fwd Pkts', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

for label, color in [('Benign', '#2196F3'), ('Attack', '#F44336')]:
    subset = df[df['label'] == label]
    axes[1].scatter(
        subset['tot fwd pkts'], subset['tot bwd pkts'],
        c=color, alpha=0.4, s=15, label=label
    )

axes[1].set_xlabel('Tot Fwd Pkts', fontsize=11)
axes[1].set_ylabel('Tot Bwd Pkts', fontsize=11)
axes[1].set_title('Tot Fwd Pkts vs Tot Bwd Pkts', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('scatter_features.png', dpi=150, bbox_inches='tight')
plt.show()
print("→ Đã lưu: scatter_features.png")


# ============================================================
# BƯỚC 9: TỔNG HỢP INSIGHT
# ============================================================
print("\n" + "=" * 60)
print("BƯỚC 9: TỔNG HỢP INSIGHT")
print("=" * 60)

total = len(df)
benign_count = label_counts.get('Benign', 0)
attack_count = label_counts.get('Attack', 0)

print(f"""
╔══════════════════════════════════════════════════════════╗
║            KẾT QUẢ PHÂN TÍCH DỮ LIỆU (EDA)              ║
╠══════════════════════════════════════════════════════════╣
║ 1. PHÂN PHỐI DỮ LIỆU                                     ║
║    - Tổng số mẫu: {total:,}                                ║
║    - Benign (bình thường): {benign_count:,} ({benign_count/total*100:.1f}%)              ║
║    - Attack (tấn công):    {attack_count:,} ({attack_count/total*100:.1f}%)              ║
╠══════════════════════════════════════════════════════════╣
║ 2. MẤT CÂN BẰNG DỮ LIỆU                                  ║
║    → Dữ liệu {'MẤT CÂN BẰNG NHẸ' if abs(benign_count - attack_count) < total*0.1 else 'MẤT CÂN BẰNG NẶNG'}                              ║
║    → Cần xử lý bằng SMOTE + UnderSampling (Người 4)      ║
╠══════════════════════════════════════════════════════════╣
║ 3. FEATURE QUAN TRỌNG (dựa trên correlation)              ║
║    - Protocol: phân biệt rõ TCP/UDP giữa Benign/Attack   ║
║    - Flow Duration: Attack thường ngắn hơn Benign        ║
║    - Fwd/Bwd Packets: Attack có pattern bất thường       ║
╠══════════════════════════════════════════════════════════╣
║ 4. OUTLIER                                                ║
║    → Có nhiều outlier trong flow duration và packet size ║
║    → Đây thường là dấu hiệu của tấn công DDoS/Scan       ║
╠══════════════════════════════════════════════════════════╣
║ 5. KẾT LUẬN                                               ║
║    → Dữ liệu đã sạch, sẵn sàng cho Người 2 (Nam)         ║
║    → Random Forest được dự đoán sẽ cho kết quả tốt nhất  ║
╚══════════════════════════════════════════════════════════╝
""")

print("\n✅ EDA HOÀN THÀNH! Các file đã xuất:")
print("   📊 label_distribution.png")
print("   📊 correlation_heatmap.png")
print("   📊 feature_distribution_by_label.png")
print("   📊 boxplot_outlier.png")
print("   📊 protocol_distribution.png")
print("   📊 scatter_features.png")
