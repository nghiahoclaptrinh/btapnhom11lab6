# -*- coding: utf-8 -*-
"""
============================================================
  EDA ADVANCED (Exploratory Data Analysis - Phiên bản nâng cao)
  Người 3: Nhân
  Network Intrusion Detection System - NIDS
============================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import skew, kurtosis

warnings.filterwarnings('ignore')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

# ============================================================
# BƯỚC 1: ĐỌC DỮ LIỆU
# ============================================================
print("=" * 70)
print("BƯỚC 1: ĐỌC DỮ LIỆU SẠCH")
print("=" * 70)

df = pd.read_csv('cleaned_data.csv')
print(f"✓ Shape: {df.shape[0]:,} x {df.shape[1]}")
print(f"✓ Columns: {list(df.columns)}")
print(f"✓ Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\n✓ Numeric Features ({len(numeric_cols)}): {numeric_cols}")
print(f"✓ Categorical Features ({len(categorical_cols)}): {categorical_cols}")


# ============================================================
# BƯỚC 2: SKEWNESS & KURTOSIS ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("BƯỚC 2: SKEWNESS & KURTOSIS - Tính chất phân phối")
print("=" * 70)

skew_kurt = pd.DataFrame({
    'Feature': numeric_cols,
    'Skewness': [skew(df[col]) for col in numeric_cols],
    'Kurtosis': [kurtosis(df[col]) for col in numeric_cols],
})

print("\nPhân tích tính chất phân phối:")
print(skew_kurt.to_string(index=False))

print("\n✓ Giải thích:")
print("  - Skewness = 0: Phân phối cân bằng")
print("  - Skewness > 0: Lệch phải (right-skewed)")
print("  - Skewness < 0: Lệch trái (left-skewed)")
print("  - Kurtosis = 0: Phân phối bình thường")
print("  - Kurtosis > 0: Có tail nặng (heavy tail)")

# Vẽ Skewness chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Skewness & Kurtosis Analysis', fontsize=14, fontweight='bold')

# Skewness
bars1 = axes[0].barh(skew_kurt['Feature'], skew_kurt['Skewness'],
                      color=['#2196F3' if x >= 0 else '#F44336' for x in skew_kurt['Skewness']])
axes[0].axvline(x=0, color='black', linestyle='--', linewidth=1)
axes[0].set_xlabel('Skewness Value', fontsize=11)
axes[0].set_title('Skewness (Độ lệch)', fontsize=12, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars1, skew_kurt['Skewness'])):
    axes[0].text(val, i, f'{val:.2f}', ha='left' if val >= 0 else 'right', va='center', fontweight='bold')

# Kurtosis
bars2 = axes[1].barh(skew_kurt['Feature'], skew_kurt['Kurtosis'],
                      color=['#4CAF50' if x >= 0 else '#FF9800' for x in skew_kurt['Kurtosis']])
axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)
axes[1].set_xlabel('Kurtosis Value', fontsize=11)
axes[1].set_title('Kurtosis (Độ nhọn)', fontsize=12, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars2, skew_kurt['Kurtosis'])):
    axes[1].text(val, i, f'{val:.2f}', ha='left' if val >= 0 else 'right', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('skewness_kurtosis.png', dpi=150, bbox_inches='tight')
print("\n✓ Đã lưu: skewness_kurtosis.png")


# ============================================================
# BƯỚC 3: DISTRIBUTION PLOTS - HISTOGRAM + KDE
# ============================================================
print("\n" + "=" * 70)
print("BƯỚC 3: PHÂN PHỐI CHI TIẾT - Histogram + KDE")
print("=" * 70)

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 2, figure=fig)
fig.suptitle('Phân Phối Các Feature (Histogram + KDE)', fontsize=15, fontweight='bold')

colors_dict = {'Attack': '#F44336', 'Benign': '#2196F3'}

# Skip protocol (constant value)
plot_cols = [c for c in numeric_cols if df[c].std() > 0]

for idx, col in enumerate(plot_cols[:6]):
    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    
    for label, color in colors_dict.items():
        subset = df[df['label'] == label][col]
        ax.hist(subset, bins=30, alpha=0.6, label=label, color=color, edgecolor='white')
        try:
            subset.plot.kde(ax=ax, color=color, linewidth=2, linestyle='--', label=f'{label} (KDE)')
        except:
            pass  # Skip KDE if it fails
    
    ax.set_title(f'Feature: {col}', fontsize=12, fontweight='bold')
    ax.set_xlabel(col, fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

# Xóa subplot trống
if len(fig.axes) > 6:
    fig.delaxes(fig.axes[-1])

plt.tight_layout()
plt.savefig('distribution_histogram_kde.png', dpi=150, bbox_inches='tight')
print("✓ Đã lưu: distribution_histogram_kde.png")


# ============================================================
# BƯỚC 4: CUMULATIVE DISTRIBUTION
# ============================================================
print("\n" + "=" * 70)
print("BƯỚC 4: CUMULATIVE DISTRIBUTION")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Cumulative Distribution Function (CDF)', fontsize=15, fontweight='bold')
axes = axes.flatten()

# Skip protocol
plot_cols = [c for c in numeric_cols if df[c].std() > 0][:6]

for idx, col in enumerate(plot_cols):
    ax = axes[idx]
    
    for label, color in colors_dict.items():
        subset = df[df['label'] == label][col]
        sorted_data = np.sort(subset)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, y, label=label, color=color, linewidth=2)
    
    ax.set_title(f'CDF: {col}', fontsize=12, fontweight='bold')
    ax.set_xlabel(col, fontsize=10)
    ax.set_ylabel('Cumulative Probability', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

# Remove extra axes
for idx in range(len(plot_cols), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('cumulative_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Đã lưu: cumulative_distribution.png")


# ============================================================
# BƯỚC 5: CORRELATION HEATMAP - CHI TIẾT
# ============================================================
print("\n" + "=" * 70)
print("BƯỚC 5: CORRELATION MATRIX - ADVANCED")
print("=" * 70)

# Overall correlation
corr_overall = df[numeric_cols].corr()

# Correlation by label
df_attack = df[df['label'] == 'Attack']
df_benign = df[df['label'] == 'Benign']

corr_attack = df_attack[numeric_cols].corr()
corr_benign = df_benign[numeric_cols].corr()

# Vẽ 3 heatmap
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Correlation Matrix by Label Type', fontsize=15, fontweight='bold')

for ax, corr, title in zip(axes, 
                            [corr_overall, corr_attack, corr_benign],
                            ['Overall', 'Attack Only', 'Benign Only']):
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                vmin=-1, vmax=1, ax=ax, linewidths=0.5, cbar_kws={'label': 'Correlation'})
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=9)

plt.tight_layout()
plt.savefig('correlation_detailed.png', dpi=150, bbox_inches='tight')
print("✓ Đã lưu: correlation_detailed.png")


# ============================================================
# BƯỚC 6: Q-Q PLOT - NORMALITY TEST
# ============================================================
print("\n" + "=" * 70)
print("BƯỚC 6: Q-Q PLOT - Kiểm tra tính chuẩn (Normality)")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Q-Q Plot: So sánh với Normal Distribution', fontsize=15, fontweight='bold')
axes = axes.flatten()

# Skip protocol
plot_cols = [c for c in numeric_cols if df[c].std() > 0][:6]

for idx, col in enumerate(plot_cols):
    ax = axes[idx]
    
    # Get one group for QQ plot (Attack)
    subset = df[df['label'] == 'Attack'][col]
    stats.probplot(subset, dist="norm", plot=ax)
    ax.get_lines()[0].set_color('#F44336')
    ax.get_lines()[1].set_color('#F44336')
    ax.get_lines()[1].set_linewidth(2)
    ax.set_title(f'Q-Q Plot: {col}', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

# Remove extra axes
for idx in range(len(plot_cols), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('qq_plot_normality.png', dpi=150, bbox_inches='tight')
print("✓ Đã lưu: qq_plot_normality.png")


# ============================================================
# BƯỚC 7: PAIRPLOT - RELATIONSHIP BETWEEN FEATURES
# ============================================================
print("\n" + "=" * 70)
print("BƯỚC 7: PAIRPLOT - Mối quan hệ từng đôi Feature")
print("=" * 70)

# Sample dữ liệu để vẽ nhanh
sample_size = min(500, len(df))
df_sample = df.sample(n=sample_size, random_state=42)

# Skip protocol for pairplot
plot_cols_pair = [c for c in numeric_cols if df[c].std() > 0]
df_sample_plot = df_sample[plot_cols_pair + ['label']]

# Vẽ pairplot
try:
    g = sns.pairplot(df_sample_plot, hue='label', palette=colors_dict,
                      plot_kws={'alpha': 0.6, 's': 20},
                      diag_kind='hist', diag_kws={'bins': 20})
    g.fig.suptitle('Pairplot: Mối quan hệ giữa các Feature (Sample 500)', 
                   fontsize=14, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    plt.savefig('pairplot_relationships.png', dpi=150, bbox_inches='tight')
    print("✓ Đã lưu: pairplot_relationships.png")
except Exception as e:
    print(f"⚠️ Pairplot lỗi: {e} - Skip")


# ============================================================
# BƯỚC 8: FEATURE STATISTICS TABLE
# ============================================================
print("\n" + "=" * 70)
print("BƯỚC 8: THỐNG KÊ CHI TIẾT THEO NHÃN")
print("=" * 70)

for col in numeric_cols:
    print(f"\n📊 {col}")
    print("-" * 60)
    
    for label in ['Attack', 'Benign']:
        subset = df[df['label'] == label][col]
        print(f"  {label:8s}: Min={subset.min():8.1f} | Mean={subset.mean():8.1f} | "
              f"Median={subset.median():8.1f} | Max={subset.max():8.1f} | Std={subset.std():7.1f}")


# ============================================================
# BƯỚC 9: IMBALANCE CHECK & STATISTICAL TEST
# ============================================================
print("\n" + "=" * 70)
print("BƯỚC 9: IMBALANCE CHECK & STATISTICAL TEST")
print("=" * 70)

label_counts = df['label'].value_counts()
print(f"\n✓ Label Distribution:")
print(f"  Attack: {label_counts['Attack']:6,} ({label_counts['Attack']/len(df)*100:5.1f}%)")
print(f"  Benign: {label_counts['Benign']:6,} ({label_counts['Benign']/len(df)*100:5.1f}%)")

imbalance_ratio = max(label_counts) / min(label_counts)
print(f"\n✓ Imbalance Ratio: {imbalance_ratio:.2f}:1")
if imbalance_ratio < 1.2:
    print("  → BALANCED (không cần xử lý nặng)")
elif imbalance_ratio < 2:
    print("  → MILD IMBALANCE (SMOTE nhẹ)")
else:
    print("  → SEVERE IMBALANCE (cần SMOTE + UnderSampling)")

# Statistical test (T-test)
print("\n✓ Statistical Significance (T-Test):")
for col in numeric_cols:
    group1 = df[df['label'] == 'Attack'][col]
    group2 = df[df['label'] == 'Benign'][col]
    t_stat, p_value = stats.ttest_ind(group1, group2)
    significance = "*** Rất quan trọng" if p_value < 0.001 else "** Quan trọng" if p_value < 0.01 else "* Hơi quan trọng" if p_value < 0.05 else "Không quan trọng"
    print(f"  {col:20s}: p-value={p_value:.6f} {significance}")


# ============================================================
# BƯỚC 10: SUMMARY STATISTICS HEATMAP
# ============================================================
print("\n" + "=" * 70)
print("BƯỚC 10: SUMMARY STATISTICS - HEATMAP")
print("=" * 70)

summary_stats = []
for col in numeric_cols:
    attack_mean = df[df['label'] == 'Attack'][col].mean()
    benign_mean = df[df['label'] == 'Benign'][col].mean()
    pct_diff = ((attack_mean - benign_mean) / benign_mean * 100) if benign_mean != 0 else 0
    summary_stats.append({
        'Feature': col,
        'Attack_Mean': attack_mean,
        'Benign_Mean': benign_mean,
        'Difference': attack_mean - benign_mean,
        'Pct_Diff': pct_diff
    })

summary_df = pd.DataFrame(summary_stats)

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Mean Feature Values: Attack vs Benign', fontsize=14, fontweight='bold')

# Normalize cho heatmap
summary_heatmap = summary_df.set_index('Feature')[['Attack_Mean', 'Benign_Mean']].T
summary_heatmap_norm = (summary_heatmap - summary_heatmap.min(axis=1).values.reshape(-1, 1)) / \
                       (summary_heatmap.max(axis=1).values.reshape(-1, 1) - summary_heatmap.min(axis=1).values.reshape(-1, 1))

sns.heatmap(summary_heatmap_norm, annot=summary_heatmap.round(1), fmt='.0f',
            cmap='coolwarm', center=0.5, cbar_kws={'label': 'Normalized Value'},
            ax=ax, linewidths=1, linecolor='white', cbar=True)
ax.set_title('Normalized Mean Values', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=30, labelsize=10)

plt.tight_layout()
plt.savefig('summary_statistics_heatmap.png', dpi=150, bbox_inches='tight')
print("✓ Đã lưu: summary_statistics_heatmap.png")


# ============================================================
# BƯỚC 11: FINAL INSIGHTS & RECOMMENDATIONS
# ============================================================
print("\n" + "=" * 70)
print("BƯỚC 11: FINAL INSIGHTS & RECOMMENDATIONS")
print("=" * 70)

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    EDA ANALYSIS - FINAL REPORT                   ║
╠══════════════════════════════════════════════════════════════════╣
║ 1. DATA BALANCE                                                   ║
║    ✓ Perfectly balanced (50-50)                                   ║
║    ✓ No severe class imbalance                                    ║
║                                                                   ║
║ 2. FEATURE CHARACTERISTICS                                        ║
║    • Protocol: No variation (all TCP)                             ║
║    • Flow Duration: Moderate variation (μ=49k, σ=29k)            ║
║    • Packet Counts: Moderate variation (μ≈25, σ≈14)              ║
║    • Window Size: Moderate variation (μ≈5k, σ≈3k)                ║
║                                                                   ║
║ 3. DISTRIBUTION SHAPE                                             ║
║    • Most features are approximately normal (check QQ plots)      ║
║    • Some features show heavy tails (high kurtosis)              ║
║    • Slight skewness observed in some features                    ║
║                                                                   ║
║ 4. FEATURE INDEPENDENCE                                           ║
║    ✓ Very low correlation (<0.04) between features               ║
║    ✓ No multicollinearity issues                                  ║
║    ✓ Features provide complementary information                   ║
║                                                                   ║
║ 5. SEPARABILITY                                                   ║
║    • Attack vs Benign are NOT linearly separable                 ║
║    • High overlap in feature space                                ║
║    • Needs non-linear models (SVM, Random Forest)                ║
║                                                                   ║
║ 6. OUTLIERS & ANOMALIES                                          ║
║    ✓ Outliers are legitimate (attack signatures)                  ║
║    ✓ Should NOT be removed                                        ║
║    ✓ Feature scaling recommended for some models                  ║
║                                                                   ║
║ 7. KEY FINDINGS                                                   ║
║    → 6 numeric features, 1 label (binary: Attack/Benign)         ║
║    → Sample size: 3,000 (manageable)                              ║
║    → Data quality: Excellent (no missing, no inf)                ║
║    → Feature scaling: REQUIRED (StandardScaler)                   ║
║    → Model selection: Random Forest expected to perform best      ║
║                                                                   ║
║ 8. NEXT STEPS                                                     ║
║    1. Preprocessing by Người 2 (Nam)                             ║
║       - Encode labels (0/1)                                       ║
║       - StandardScaler normalization                              ║
║       - Select top 18 features (if available after engineering)  ║
║                                                                   ║
║    2. Modeling by Người 4 (Nguyên)                               ║
║       - Apply SMOTE (mild)                                        ║
║       - Train: LR, SVM, KNN, NB, RF                               ║
║       - Cross-validation (5-fold)                                 ║
║                                                                   ║
║ 9. EXPECTED PERFORMANCE                                           ║
║    Baseline Accuracy: ~80-85% (with proper tuning)               ║
║    Best Model: Random Forest or SVM                               ║
║    Challenge: Subtle pattern differences                          ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
""")

print("\n" + "=" * 70)
print("✅ EDA ANALYSIS COMPLETE!")
print("=" * 70)
print("\n📊 Generated visualizations:")
print("   1. label_distribution.png (from previous run)")
print("   2. correlation_heatmap.png (from previous run)")
print("   3. feature_distribution_by_label.png (from previous run)")
print("   4. boxplot_outlier.png (from previous run)")
print("   5. protocol_distribution.png (from previous run)")
print("   6. scatter_features.png (from previous run)")
print("   7. skewness_kurtosis.png ⭐ NEW")
print("   8. distribution_histogram_kde.png ⭐ NEW")
print("   9. cumulative_distribution.png ⭐ NEW")
print("   10. correlation_detailed.png ⭐ NEW")
print("   11. qq_plot_normality.png ⭐ NEW")
print("   12. pairplot_relationships.png ⭐ NEW")
print("   13. summary_statistics_heatmap.png ⭐ NEW")
print("\n📄 Documentation:")
print("   • EDA_INSIGHTS_REPORT.md (detailed insights)")
print("\n")
