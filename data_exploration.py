"""
Features:
- Comprehensive statistical analysis
- Feature importance rankings
- Advanced visualizations
- Pickle file output for GUI viewer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from scipy.stats import chi2_contingency, skew, kurtosis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif, f_classif, SelectKBest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle
import warnings

warnings.filterwarnings("ignore")

# Configuration
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("=" * 120)
print("ULTIMATE EDA - FINANCIAL FRAUD DETECTION")
print("=" * 120)
print("\nInitializing comprehensive analysis pipeline...")

# Load data
df = pd.read_csv("financial_fraud_detection_dataset.csv")
df_original = df.copy()

# Initialize results dictionary for pickle
results = {
    "metadata": {},
    "data_quality": {},
    "numerical_analysis": {},
    "categorical_analysis": {},
    "temporal_analysis": {},
    "correlation_analysis": {},
    "feature_importance": {},
    "pca_analysis": {},
    "clustering_analysis": {},
    "statistical_tests": {},
    "model_readiness": {},
    "recommendations": [],
    "visualizations": {},
    "raw_data": df_original,
}

# ============================================================================
# 1. METADATA & DATA PROFILING
# ============================================================================
print("\n" + "=" * 120)
print("1. DATASET METADATA & PROFILING")
print("=" * 120)

metadata = {
    "total_rows": len(df),
    "total_columns": len(df.columns),
    "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
    "bytes_per_row": df.memory_usage(deep=True).sum() / len(df),
    "column_info": {},
    "dtypes": df.dtypes.value_counts().to_dict(),
}

print(f"\nDataset Overview:")
print(f"   Rows: {metadata['total_rows']:,}")
print(f"   Columns: {metadata['total_columns']}")
print(f"   Memory: {metadata['memory_mb']:.2f} MB")
print(f"   Bytes/Row: {metadata['bytes_per_row']:.0f}")

# Detailed column info
for col in df.columns:
    metadata["column_info"][col] = {
        "dtype": str(df[col].dtype),
        "non_null": int(df[col].count()),
        "null_count": int(df[col].isnull().sum()),
        "null_pct": float(df[col].isnull().sum() / len(df) * 100),
        "unique": int(df[col].nunique()),
        "unique_pct": float(df[col].nunique() / len(df) * 100),
    }

results["metadata"] = metadata

# ============================================================================
# 2. DATA QUALITY ASSESSMENT
# ============================================================================
print("\n" + "=" * 120)
print("2. DATA QUALITY ASSESSMENT")
print("=" * 120)

# Missing values
total_missing = df.isnull().sum().sum()
total_cells = len(df) * len(df.columns)
completeness = ((total_cells - total_missing) / total_cells) * 100

# Duplicates
duplicates = df.duplicated().sum()

# Quality scores
quality_metrics = {
    "completeness": completeness,
    "uniqueness": (1 - duplicates / len(df)) * 100,
    "consistency": 100.0,  # Placeholder
}

overall_quality = np.mean(list(quality_metrics.values()))

data_quality = {
    "total_missing": int(total_missing),
    "completeness_pct": float(completeness),
    "duplicate_rows": int(duplicates),
    "duplicate_pct": float(duplicates / len(df) * 100),
    "quality_metrics": quality_metrics,
    "overall_quality": float(overall_quality),
    "missing_by_column": df.isnull().sum().to_dict(),
}

print(f"\nQuality Metrics:")
print(f"   Completeness: {completeness:.2f}%")
print(f"   Duplicates: {duplicates} ({duplicates / len(df) * 100:.2f}%)")
print(f"   Overall Quality: {overall_quality:.2f}%")

results["data_quality"] = data_quality

# ============================================================================
# 3. NUMERICAL FEATURES ANALYSIS
# ============================================================================
print("\n" + "=" * 120)
print("3. NUMERICAL FEATURES ANALYSIS")
print("=" * 120)

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nAnalyzing {len(numerical_cols)} numerical features...")

numerical_analysis = {
    "feature_count": len(numerical_cols),
    "features": {},
    "summary_stats": {},
}

if len(numerical_cols) > 0:
    # Comprehensive statistics
    for col in numerical_cols:
        if df[col].notna().sum() > 0:
            data = df[col].dropna()

            # Basic stats
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1

            # Outliers
            outliers_iqr = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
            outliers_zscore = (np.abs(stats.zscore(data)) > 3).sum()

            # Normality test
            if len(data) >= 3:
                _, normality_p = stats.normaltest(data)
            else:
                normality_p = np.nan

            feature_stats = {
                "count": int(data.count()),
                "mean": float(data.mean()),
                "median": float(data.median()),
                "std": float(data.std()),
                "min": float(data.min()),
                "max": float(data.max()),
                "q1": float(Q1),
                "q3": float(Q3),
                "iqr": float(IQR),
                "range": float(data.max() - data.min()),
                "skewness": float(data.skew()),
                "kurtosis": float(data.kurtosis()),
                "cv": float(data.std() / data.mean() * 100) if data.mean() != 0 else 0,
                "outliers_iqr": int(outliers_iqr),
                "outliers_zscore": int(outliers_zscore),
                "normality_p_value": float(normality_p)
                if not np.isnan(normality_p)
                else None,
                "is_normal": bool(normality_p > 0.05)
                if not np.isnan(normality_p)
                else None,
            }

            numerical_analysis["features"][col] = feature_stats

            print(f"\n{col}:")
            print(
                f"   Mean: {feature_stats['mean']:.4f}, Median: {feature_stats['median']:.4f}"
            )
            print(
                f"   Skewness: {feature_stats['skewness']:.4f}, Kurtosis: {feature_stats['kurtosis']:.4f}"
            )
            print(
                f"   Outliers (IQR): {outliers_iqr}, Normal: {feature_stats['is_normal']}"
            )

results["numerical_analysis"] = numerical_analysis

# ============================================================================
# 4. CATEGORICAL FEATURES ANALYSIS
# ============================================================================
print("\n" + "=" * 120)
print("4. CATEGORICAL FEATURES ANALYSIS")
print("=" * 120)

categorical_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
print(f"\nAnalyzing {len(categorical_cols)} categorical features...")

categorical_analysis = {"feature_count": len(categorical_cols), "features": {}}

for col in categorical_cols:
    if col not in ["transaction_id", "timestamp", "ip_address"]:
        value_counts = df[col].value_counts()

        feature_info = {
            "unique_count": int(df[col].nunique()),
            "top_value": str(value_counts.index[0]) if len(value_counts) > 0 else None,
            "top_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            "top_pct": float(value_counts.iloc[0] / len(df) * 100)
            if len(value_counts) > 0
            else 0,
            "entropy": float(stats.entropy(value_counts)),
            "value_counts": value_counts.to_dict(),
            "cardinality": "high"
            if df[col].nunique() / len(df) > 0.5
            else "medium"
            if df[col].nunique() / len(df) > 0.1
            else "low",
        }

        categorical_analysis["features"][col] = feature_info

        print(
            f"\n{col}: {feature_info['unique_count']} unique values ({feature_info['cardinality']} cardinality)"
        )
        if feature_info["unique_count"] <= 10:
            for val, cnt in list(value_counts.items())[:5]:
                print(f"   • {val}: {cnt} ({cnt / len(df) * 100:.1f}%)")

results["categorical_analysis"] = categorical_analysis

# ============================================================================
# 5. TEMPORAL ANALYSIS
# ============================================================================
if "timestamp" in df.columns:
    print("\n" + "=" * 120)
    print("5. TEMPORAL ANALYSIS")
    print("=" * 120)

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")

    # Extract temporal features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_name"] = df["timestamp"].dt.day_name()
    df["month"] = df["timestamp"].dt.month
    df["month_name"] = df["timestamp"].dt.month_name()
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    df["quarter"] = df["timestamp"].dt.quarter

    time_span = (df["timestamp"].max() - df["timestamp"].min()).days

    temporal_analysis = {
        "min_date": str(df["timestamp"].min()),
        "max_date": str(df["timestamp"].max()),
        "time_span_days": int(time_span),
        "peak_hour": int(df["hour"].mode().values[0]),
        "peak_day": str(df["day_name"].mode().values[0]),
        "peak_month": str(df["month_name"].mode().values[0]),
        "weekend_pct": float(df["is_weekend"].sum() / len(df) * 100),
        "hourly_distribution": df["hour"].value_counts().sort_index().to_dict(),
        "daily_distribution": df["day_name"].value_counts().to_dict(),
        "monthly_distribution": df["month_name"].value_counts().to_dict(),
    }

    print(f"\n📅 Temporal Coverage: {time_span} days")
    print(f"   From: {temporal_analysis['min_date']}")
    print(f"   To: {temporal_analysis['max_date']}")
    print(f"\nPatterns:")
    print(f"   • Peak Hour: {temporal_analysis['peak_hour']}:00")
    print(f"   • Peak Day: {temporal_analysis['peak_day']}")
    print(f"   • Weekend Activity: {temporal_analysis['weekend_pct']:.1f}%")

    results["temporal_analysis"] = temporal_analysis

# ============================================================================
# 6. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 120)
print("6. CORRELATION & MULTICOLLINEARITY ANALYSIS")
print("=" * 120)

correlation_analysis = {
    "pearson_matrix": {},
    "spearman_matrix": {},
    "strong_correlations": [],
}

if len(numerical_cols) > 1:
    # Pearson correlation
    pearson_corr = df[numerical_cols].corr(method="pearson")
    correlation_analysis["pearson_matrix"] = pearson_corr.to_dict()

    # Spearman correlation
    spearman_corr = df[numerical_cols].corr(method="spearman")
    correlation_analysis["spearman_matrix"] = spearman_corr.to_dict()

    print("\n📊 Correlation Analysis:")
    print("Pearson Correlation Matrix:")
    print(pearson_corr.round(3))

    # Strong correlations
    for i in range(len(pearson_corr.columns)):
        for j in range(i + 1, len(pearson_corr.columns)):
            if abs(pearson_corr.iloc[i, j]) > 0.5 and not np.isnan(
                pearson_corr.iloc[i, j]
            ):
                correlation_analysis["strong_correlations"].append(
                    {
                        "feature1": pearson_corr.columns[i],
                        "feature2": pearson_corr.columns[j],
                        "pearson": float(pearson_corr.iloc[i, j]),
                        "spearman": float(spearman_corr.iloc[i, j]),
                    }
                )

    print(
        f"\n📍 Strong Correlations (|r| > 0.5): {len(correlation_analysis['strong_correlations'])}"
    )

results["correlation_analysis"] = correlation_analysis

# ============================================================================
# 7. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 120)
print("7. FEATURE IMPORTANCE ANALYSIS")
print("=" * 120)

# Prepare ML dataset
df_ml = df.copy()
label_encoders = {}

# Encode categorical features
for col in categorical_cols:
    if col not in ["transaction_id", "timestamp", "ip_address"]:
        le = LabelEncoder()
        df_ml[col + "_enc"] = le.fit_transform(df_ml[col].astype(str))
        label_encoders[col] = le

# Get numeric features
numeric_features = [
    c
    for c in df_ml.select_dtypes(include=[np.number]).columns
    if c not in ["is_fraud"] and df_ml[c].notna().sum() > 0
]

feature_importance = {
    "has_target": False,
    "variance_analysis": {},
    "rf_importance": {},
    "gb_importance": {},
    "mutual_info": {},
}

has_target = "is_fraud" in df.columns and df["is_fraud"].sum() > 0

if has_target:
    print("\n🎯 SUPERVISED FEATURE IMPORTANCE")
    feature_importance["has_target"] = True

    X = df_ml[numeric_features].fillna(df_ml[numeric_features].mean())
    y = df_ml["is_fraud"].astype(int)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    feature_importance["rf_importance"] = dict(
        zip(numeric_features, rf.feature_importances_)
    )

    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X, y)
    feature_importance["gb_importance"] = dict(
        zip(numeric_features, gb.feature_importances_)
    )

    # Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    feature_importance["mutual_info"] = dict(zip(numeric_features, mi_scores))

    # Print top features
    rf_sorted = sorted(
        feature_importance["rf_importance"].items(), key=lambda x: x[1], reverse=True
    )
    print("\n📊 Top 10 Features (Random Forest):")
    for feat, imp in rf_sorted[:10]:
        print(f"   • {feat}: {imp:.4f}")

else:
    print("\nUNSUPERVISED ANALYSIS (No fraud cases)")

    X = df_ml[numeric_features].fillna(df_ml[numeric_features].mean())

    # Variance analysis
    variances = X.var()
    feature_importance["variance_analysis"] = variances.to_dict()

    print("\n📊 Top 10 Features by Variance:")
    for feat, var in variances.sort_values(ascending=False).head(10).items():
        print(f"   • {feat}: {var:.4f}")

results["feature_importance"] = feature_importance

# ============================================================================
# 8. PCA ANALYSIS
# ============================================================================
print("\n" + "=" * 120)
print("8. DIMENSIONALITY REDUCTION (PCA)")
print("=" * 120)

pca_analysis = {}

if len(numeric_features) > 2:
    X = df_ml[numeric_features].fillna(df_ml[numeric_features].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)

    # Store results
    pca_analysis = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance": pca.explained_variance_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "n_components_95": int(
            np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
        ),
        "total_components": len(pca.explained_variance_ratio_),
        "reduction_potential_pct": float(
            (
                1
                - (np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1)
                / len(numeric_features)
            )
            * 100
        ),
    }

    print(f"\n📊 PCA Results:")
    print(
        f"   • Components for 95% variance: {pca_analysis['n_components_95']}/{pca_analysis['total_components']}"
    )
    print(f"   • Reduction potential: {pca_analysis['reduction_potential_pct']:.1f}%")

    # Transform data for visualization
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    pca_analysis["pca_2d_data"] = X_pca_2d.tolist()

results["pca_analysis"] = pca_analysis

# ============================================================================
# 9. CLUSTERING ANALYSIS
# ============================================================================
print("\n" + "=" * 120)
print("9. CLUSTERING ANALYSIS")
print("=" * 120)

clustering_analysis = {}

if len(numeric_features) > 2:
    X = df_ml[numeric_features].fillna(df_ml[numeric_features].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-means clustering
    inertias = []
    silhouette_scores = []

    for k in range(2, min(11, len(df))):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)

        from sklearn.metrics import silhouette_score

        sil_score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(sil_score)

    # Find optimal k
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2

    # Fit final model
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_scaled)

    clustering_analysis = {
        "optimal_k": int(optimal_k),
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "cluster_labels": cluster_labels.tolist(),
        "cluster_sizes": pd.Series(cluster_labels).value_counts().to_dict(),
    }

    print(f"\n📊 K-Means Clustering:")
    print(f"   • Optimal clusters: {optimal_k}")
    print(f"   • Silhouette score: {silhouette_scores[optimal_k - 2]:.4f}")
    print(f"\n   Cluster sizes:")
    for cluster, size in clustering_analysis["cluster_sizes"].items():
        print(f"      Cluster {cluster}: {size} transactions")

results["clustering_analysis"] = clustering_analysis

# ============================================================================
# 10. STATISTICAL TESTS
# ============================================================================
print("\n" + "=" * 120)
print("10. STATISTICAL HYPOTHESIS TESTS")
print("=" * 120)

statistical_tests = {"chi_square_tests": [], "t_tests": []}

# Chi-square tests for categorical associations
cat_cols_test = [
    c
    for c in categorical_cols
    if c not in ["transaction_id", "timestamp", "ip_address"]
    and df[c].nunique() > 1
    and df[c].nunique() < 20
]

if len(cat_cols_test) >= 2:
    print("\n📊 Chi-Square Independence Tests:")
    for i, col1 in enumerate(cat_cols_test[:3]):
        for col2 in cat_cols_test[i + 1 : i + 4]:
            try:
                contingency = pd.crosstab(df[col1], df[col2])
                chi2, p_value, dof, expected = chi2_contingency(contingency)

                test_result = {
                    "variable1": col1,
                    "variable2": col2,
                    "chi2_statistic": float(chi2),
                    "p_value": float(p_value),
                    "dof": int(dof),
                    "significant": bool(p_value < 0.05),
                }

                statistical_tests["chi_square_tests"].append(test_result)
                print(f"\n{col1} vs {col2}:")
                print(
                    f"   χ² = {chi2:.4f}, p = {p_value:.4f}, {'Significant' if p_value < 0.05 else 'Not significant'}"
                )
            except:
                pass

results["statistical_tests"] = statistical_tests

# ============================================================================
# 11. MODEL READINESS ASSESSMENT
# ============================================================================
print("\n" + "=" * 120)
print("11. MODEL READINESS ASSESSMENT")
print("=" * 120)

model_readiness = {
    "completeness_score": completeness,
    "completeness_pass": completeness > 90,
    "sample_size": len(df),
    "sample_size_pass": len(df) >= 1000,
    "feature_count": len(numerical_cols) + len(categorical_cols),
    "feature_pass": (len(numerical_cols) + len(categorical_cols)) > 5,
    "class_balance_score": 0,
    "class_balance_pass": False,
    "overall_readiness": "NOT READY",
}

if "is_fraud" in df.columns:
    fraud_rate = df["is_fraud"].sum() / len(df)
    model_readiness["class_balance_score"] = fraud_rate * 100
    model_readiness["class_balance_pass"] = fraud_rate > 0 and fraud_rate < 0.99

    print(f"\nModel Readiness Checklist:")
    print(
        f"   1. Completeness: {completeness:.1f}% - {'PASS' if completeness > 90 else 'REVIEW'}"
    )
    print(f"   2. Sample Size: {len(df):,} - {'PASS' if len(df) >= 1000 else 'FAIL'}")
    print(
        f"   3. Class Balance: {fraud_rate * 100:.2f}% fraud - {'PASS' if fraud_rate > 0 else 'FAIL'}"
    )
    print(
        f"   4. Feature Quality: {len(numerical_cols) + len(categorical_cols)} features - PASS"
    )

    # Overall assessment
    checks_passed = sum(
        [
            model_readiness["completeness_pass"],
            model_readiness["sample_size_pass"],
            model_readiness["class_balance_pass"],
            model_readiness["feature_pass"],
        ]
    )

    if checks_passed >= 3:
        model_readiness["overall_readiness"] = "READY"
    elif checks_passed >= 2:
        model_readiness["overall_readiness"] = "NEEDS WORK"
    else:
        model_readiness["overall_readiness"] = "NOT READY"

    print(f"\n   Overall: {model_readiness['overall_readiness']}")

results["model_readiness"] = model_readiness

# ============================================================================
# 12. RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 120)
print("12. RECOMMENDATIONS")
print("=" * 120)

recommendations = []

if completeness < 90:
    recommendations.append(
        {
            "priority": "HIGH",
            "category": "Data Quality",
            "issue": f"Missing {100 - completeness:.1f}% of data",
            "action": "Implement imputation strategy or collect more complete data",
        }
    )

if "is_fraud" in df.columns and df["is_fraud"].sum() == 0:
    recommendations.append(
        {
            "priority": "CRITICAL",
            "category": "Target Variable",
            "issue": "No fraud cases in dataset",
            "action": "Collect diverse data including fraud examples or use synthetic data",
        }
    )

if len(df) < 1000:
    recommendations.append(
        {
            "priority": "HIGH",
            "category": "Sample Size",
            "issue": f"Only {len(df)} transactions",
            "action": "Collect more data for robust model training (target: 10,000+)",
        }
    )

if "is_fraud" in df.columns and (df["is_fraud"].sum() / len(df)) < 0.01:
    recommendations.append(
        {
            "priority": "HIGH",
            "category": "Class Imbalance",
            "issue": "Severe class imbalance",
            "action": "Use SMOTE, class weights, or ensemble methods",
        }
    )

recommendations.append(
    {
        "priority": "MEDIUM",
        "category": "Feature Engineering",
        "issue": "Limited feature set",
        "action": "Create interaction features, ratios, and temporal aggregations",
    }
)

recommendations.append(
    {
        "priority": "MEDIUM",
        "category": "Model Selection",
        "issue": "Temporal dependencies",
        "action": "Use time-aware cross-validation and consider LSTM/GRU models",
    }
)

results["recommendations"] = recommendations

print("\n🎯 Actionable Recommendations:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. [{rec['priority']}] {rec['category']}: {rec['action']}")


print("\nCreating visualization suite...")

# List to track all saved plots
saved_plots = []

# 1. Numerical distributions with KDE
if len(numerical_cols) > 0:
    num_plots = len([c for c in numerical_cols if df[c].notna().sum() > 1])
    cols_per_row = 4
    num_rows = (num_plots + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(16, 4 * num_rows))
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    plot_idx = 0
    for col in numerical_cols:
        if df[col].notna().sum() > 1:
            ax = axes[plot_idx]
            df[col].hist(
                bins=30,
                edgecolor="black",
                alpha=0.7,
                ax=ax,
                density=True,
                color="skyblue",
            )
            try:
                df[col].plot(
                    kind="kde", ax=ax, color="red", linewidth=2, secondary_y=False
                )
            except:
                pass
            ax.axvline(
                df[col].mean(), color="green", linestyle="--", linewidth=2, label="Mean"
            )
            ax.axvline(
                df[col].median(),
                color="orange",
                linestyle="--",
                linewidth=2,
                label="Median",
            )
            ax.set_xlabel(col)
            ax.set_ylabel("Density")
            ax.set_title(f"{col} Distribution")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig("01_numerical_distributions.png", dpi=300, bbox_inches="tight")
    saved_plots.append("01_numerical_distributions.png")
    print("Saved: 01_numerical_distributions.png")
    plt.close()

# 2. Box plots for outliers
if len(numerical_cols) > 0:
    cols_to_plot = [c for c in numerical_cols[:6] if df[c].notna().sum() > 1]
    if len(cols_to_plot) > 0:
        cols_per_row = 3
        num_rows = (len(cols_to_plot) + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(14, 4 * num_rows))
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for idx, col in enumerate(cols_to_plot):
            df.boxplot(column=col, ax=axes[idx], vert=False)
            axes[idx].set_xlabel(col)
            axes[idx].set_title(f"{col} - Outlier Detection")
            axes[idx].grid(alpha=0.3)

        # Hide unused subplots
        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig("02_outlier_detection_boxplots.png", dpi=300, bbox_inches="tight")
        saved_plots.append("02_outlier_detection_boxplots.png")
        print("Saved: 02_outlier_detection_boxplots.png")
        plt.close()

# 3. Categorical distributions
cat_viz = [
    c
    for c in categorical_cols
    if c not in ["transaction_id", "timestamp", "ip_address"]
][:6]
if len(cat_viz) > 0:
    cols_per_row = 3
    num_rows = (len(cat_viz) + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(15, 5 * num_rows))
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, col in enumerate(cat_viz):
        vc = df[col].value_counts().head(10)
        if len(vc) <= 6:
            colors = sns.color_palette("husl", len(vc))
            axes[idx].pie(
                vc.values,
                labels=vc.index,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
            )
            axes[idx].set_title(f"{col} Distribution")
        else:
            vc.plot(kind="barh", ax=axes[idx], color="lightcoral", edgecolor="black")
            axes[idx].set_xlabel("Count")
            axes[idx].set_title(f"{col} (Top 10)")
            axes[idx].grid(alpha=0.3, axis="x")

    # Hide unused subplots
    for idx in range(len(cat_viz), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig("03_categorical_distributions.png", dpi=300, bbox_inches="tight")
    saved_plots.append("03_categorical_distributions.png")
    print("Saved: 03_categorical_distributions.png")
    plt.close()

# 4. Correlation heatmap
if len(numerical_cols) > 1:
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[numerical_cols].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
    )
    ax.set_title("Pearson Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("04_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    saved_plots.append("04_correlation_heatmap.png")
    print("Saved: 04_correlation_heatmap.png")
    plt.close()

# 5. Temporal patterns
if "timestamp" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Hourly pattern
    hourly = df["hour"].value_counts().sort_index()
    hourly.plot(
        kind="line",
        marker="o",
        ax=axes[0],
        linewidth=2.5,
        markersize=8,
        color="darkblue",
    )
    axes[0].fill_between(hourly.index, hourly.values, alpha=0.3)
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Transaction Count")
    axes[0].set_title("Hourly Transaction Pattern")
    axes[0].grid(alpha=0.3)

    # Day of week
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    day_counts = df["day_name"].value_counts().reindex(day_order)
    colors = [
        "#ff6b6b" if day in ["Saturday", "Sunday"] else "#4ecdc4" for day in day_order
    ]
    day_counts.plot(
        kind="bar", ax=axes[1], color=colors, edgecolor="black", linewidth=1.5
    )
    axes[1].set_xlabel("Day of Week")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Transactions by Day of Week")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("05_temporal_patterns.png", dpi=300, bbox_inches="tight")
    saved_plots.append("05_temporal_patterns.png")
    print("Saved: 05_temporal_patterns.png")
    plt.close()

# 6. Feature importance
if feature_importance["has_target"] and feature_importance["rf_importance"]:
    fig, ax = plt.subplots(figsize=(10, 8))
    rf_imp = (
        pd.Series(feature_importance["rf_importance"])
        .sort_values(ascending=True)
        .tail(15)
    )
    rf_imp.plot(kind="barh", ax=ax, color="forestgreen", edgecolor="black")
    ax.set_xlabel("Importance Score")
    ax.set_title("Top 15 Features (Random Forest)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig("06_feature_importance.png", dpi=300, bbox_inches="tight")
    saved_plots.append("06_feature_importance.png")
    print("Saved: 06_feature_importance.png")
    plt.close()

# 7. PCA analysis
if pca_analysis:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scree plot
    cumvar = pca_analysis["cumulative_variance"]
    axes[0].plot(
        range(1, len(cumvar) + 1),
        [v * 100 for v in cumvar],
        marker="o",
        linewidth=2.5,
        markersize=8,
        color="purple",
    )
    axes[0].axhline(
        y=95, color="red", linestyle="--", linewidth=2, label="95% Threshold"
    )
    axes[0].set_xlabel("Number of Components")
    axes[0].set_ylabel("Cumulative Explained Variance (%)")
    axes[0].set_title("PCA Scree Plot")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # PCA 2D projection
    if "pca_2d_data" in pca_analysis:
        pca_data = np.array(pca_analysis["pca_2d_data"])
        scatter = axes[1].scatter(
            pca_data[:, 0],
            pca_data[:, 1],
            c=range(len(pca_data)),
            cmap="viridis",
            alpha=0.6,
            s=100,
            edgecolors="black",
            linewidths=0.5,
        )
        axes[1].set_xlabel(
            f"PC1 ({pca_analysis['explained_variance_ratio'][0] * 100:.1f}%)"
        )
        axes[1].set_ylabel(
            f"PC2 ({pca_analysis['explained_variance_ratio'][1] * 100:.1f}%)"
        )
        axes[1].set_title("PCA 2D Projection")
        axes[1].grid(alpha=0.3)
        plt.colorbar(scatter, ax=axes[1], label="Transaction Index")

    plt.tight_layout()
    plt.savefig("07_pca_analysis.png", dpi=300, bbox_inches="tight")
    saved_plots.append("07_pca_analysis.png")
    print("Saved: 07_pca_analysis.png")
    plt.close()

# 8. Clustering visualization
if clustering_analysis and "pca_2d_data" in pca_analysis:
    fig, ax = plt.subplots(figsize=(10, 8))
    pca_data = np.array(pca_analysis["pca_2d_data"])
    labels = np.array(clustering_analysis["cluster_labels"])
    scatter = ax.scatter(
        pca_data[:, 0],
        pca_data[:, 1],
        c=labels,
        cmap="Set3",
        alpha=0.7,
        s=100,
        edgecolors="black",
        linewidths=0.5,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        f"K-Means Clustering (k={clustering_analysis['optimal_k']})",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    plt.savefig("08_clustering_analysis.png", dpi=300, bbox_inches="tight")
    saved_plots.append("08_clustering_analysis.png")
    print("Saved: 08_clustering_analysis.png")
    plt.close()

# Store visualization metadata
results["visualizations"] = {
    "plots": saved_plots,
    "plot_count": len(saved_plots),
}


# ============================================================================
# 15. SAVE RESULTS TO PICKLE
# ============================================================================
print("\n" + "=" * 120)
print("15. SAVING RESULTS TO PICKLE FILE")
print("=" * 120)

# Add timestamp
results["timestamp"] = datetime.now().isoformat()
results["dataset_name"] = "financial_fraud_detection_dataset.csv"

# Save to pickle
with open("eda_results.pkl", "wb") as f:
    pickle.dump(results, f)

print("Saved: eda_results.pkl")
print(
    f"\n📦 Pickle file size: {pd.Series(pickle.dumps(results)).memory_usage() / 1024:.2f} KB"
)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 120)
print("\nANALYSIS COMPLETE!")
print("=" * 120)

print(f"""
OUTPUTS GENERATED:

1. Console Report (this output)
2. Visualizations ({len(saved_plots)} separate PNG files):
   - 01_numerical_distributions.png (feature distributions with KDE)
   - 02_outlier_detection_boxplots.png (box plots for outlier analysis)
   - 03_categorical_distributions.png (categorical feature distributions)
   - 04_correlation_heatmap.png (feature correlation matrix)
   - 05_temporal_patterns.png (hourly and daily transaction patterns)
   - 06_feature_importance.png (top 15 important features)
   - 07_pca_analysis.png (PCA scree plot and 2D projection)
   - 08_clustering_analysis.png (K-means clustering results)
3. Pickle File: eda_results.pkl (for GUI viewer)

ANALYSIS SUMMARY:
   - Dataset: {len(df):,} transactions, {len(df.columns)} features
   - Quality Score: {overall_quality:.1f}%
   - Temporal Span: {temporal_analysis.get("time_span_days", "N/A")} days
   - Model Readiness: {model_readiness["overall_readiness"]}

""")
print("=" * 120)
print("\nAnalysis pipeline completed successfully!")
