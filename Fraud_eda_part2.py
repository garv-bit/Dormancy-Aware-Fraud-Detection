"""
Fraud EDA Part 2 — Correlation, Feature Importance, PCA, Clustering

All methods are attached to FraudEDA in Fraud_eda.py.
Leakage guard: config.LEAKY_COLUMNS enforced in every feature matrix.
Memory:        RF, GB, clustering run on a stratified sample (SAMPLE_SIZE rows).
Checkpointing: each section saves/loads from eda_checkpoints/<section>.joblib.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Any

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

from Utils import (
    safe_divide,
    get_numeric_columns,
    get_categorical_columns,
    print_section_header,
)

warnings.filterwarnings("ignore")


# ==============================================================================
# 7. CORRELATION ANALYSIS
# ==============================================================================

def analyze_correlation(self, numerical_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compute Pearson and Spearman correlation matrices on leak-free numeric columns.
    Flags pairs where |r| > config.CORRELATION_THRESHOLD.
    """
    print_section_header("6. CORRELATION & MULTICOLLINEARITY ANALYSIS", logger=self.logger)

    cached = self._ckpt_load("correlation_analysis")
    if cached is not None:
        self.results['correlation_analysis'] = cached
        return cached

    leaky = self._leaky()

    if numerical_cols is None:
        numerical_cols = [
            c for c in get_numeric_columns(self.df, exclude=['is_fraud'])
            if c not in leaky
        ]

    correlation_analysis = {
        'pearson_matrix':      {},
        'spearman_matrix':     {},
        'strong_correlations': [],
    }

    if len(numerical_cols) <= 1:
        self.logger.warning("Need ≥2 numerical columns for correlation analysis")
        return correlation_analysis

    try:
        pearson_corr  = self.df[numerical_cols].corr(method='pearson')
        spearman_corr = self.df[numerical_cols].corr(method='spearman')

        correlation_analysis['pearson_matrix']  = pearson_corr.to_dict()
        correlation_analysis['spearman_matrix'] = spearman_corr.to_dict()

        print("\n📊 Correlation Analysis:")
        print("Pearson Correlation Matrix:")
        print(pearson_corr.round(3))

        cm   = pearson_corr.values
        sm   = spearman_corr.values
        cols = pearson_corr.columns
        mask = np.triu(np.ones_like(cm, dtype=bool), k=1)

        for i, j in zip(*np.where(
            (np.abs(cm) > self.config.CORRELATION_THRESHOLD) & mask & ~np.isnan(cm)
        )):
            correlation_analysis['strong_correlations'].append({
                'feature1': cols[i],
                'feature2': cols[j],
                'pearson':  float(cm[i, j]),
                'spearman': float(sm[i, j]),
            })

        n_strong = len(correlation_analysis['strong_correlations'])
        print(f"\n🔍 Strong Correlations (|r| > {self.config.CORRELATION_THRESHOLD}): {n_strong}")
        for corr in correlation_analysis['strong_correlations'][:10]:
            print(f"   • {corr['feature1']} ↔ {corr['feature2']}: r={corr['pearson']:.3f}")

    except Exception as e:
        self.logger.error(f"Error in correlation analysis: {str(e)}")

    self.results['correlation_analysis'] = correlation_analysis
    self._ckpt_save("correlation_analysis", correlation_analysis)
    self.logger.info(
        f"Correlation analysis complete, found "
        f"{len(correlation_analysis['strong_correlations'])} strong correlations"
    )
    return correlation_analysis


# ==============================================================================
# 8. FEATURE IMPORTANCE
# ==============================================================================

def analyze_feature_importance(self) -> Dict[str, Any]:
    """
    Compute supervised feature importance (RF, GB, mutual info) on a stratified
    sample. Falls back to variance analysis when no target is available.

    Leakage: encoding loop and feature matrix both skip LEAKY_COLUMNS.
    Memory:  runs on config.SAMPLE_SIZE rows, not all 5M.
    """
    print_section_header("7. FEATURE IMPORTANCE ANALYSIS", logger=self.logger)

    cached = self._ckpt_load("feature_importance")
    if cached is not None:
        self.results['feature_importance'] = cached
        return cached

    leaky  = self._leaky()
    # Stratified sample to avoid OOM on 5M rows
    df_sample = self._get_sample()
    df_ml     = df_sample.copy()

    # Encode categoricals — skip leaky columns
    for col in get_categorical_columns(self.df, exclude=self.config.SKIP_CATEGORICAL_ANALYSIS):
        if col in leaky:
            self.logger.info(f"[LEAKAGE GUARD] Skipping encoding of '{col}'")
            continue
        try:
            df_ml[col + '_enc'] = LabelEncoder().fit_transform(df_ml[col].astype(str))
        except Exception as e:
            self.logger.warning(f"Encoding failed for {col}: {e}")

    # Build leak-free numeric feature list
    all_numeric      = df_ml.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [
        c for c in all_numeric
        if c not in leaky and df_ml[c].notna().sum() > 0
    ]
    dropped = [c for c in all_numeric if c in leaky]

    if dropped:
        self.logger.info(f"[LEAKAGE GUARD] Dropped from feature matrix X: {dropped}")
        print(f"\n⚠️  Leaky columns removed from feature matrix: {dropped}")

    feature_importance = {
        'has_target':             False,
        'leaky_columns_removed':  dropped,
        'sample_size_used':       len(df_ml),
        'variance_analysis':      {},
        'rf_importance':          {},
        'gb_importance':          {},
        'mutual_info':            {},
    }

    has_target = 'is_fraud' in df_ml.columns and df_ml['is_fraud'].sum() > 0

    if has_target and numeric_features:
        print(f"\n🎯 SUPERVISED FEATURE IMPORTANCE (sample: {len(df_ml):,} rows)")
        feature_importance['has_target'] = True

        try:
            X = df_ml[numeric_features].fillna(df_ml[numeric_features].mean())
            y = df_ml['is_fraud'].astype(int)

            X_train, _, y_train, _ = train_test_split(
                X, y,
                test_size=self.config.TEST_SIZE,
                random_state=self.config.RANDOM_STATE,
                stratify=y if y.sum() > 1 else None,
            )

            # Random Forest
            self.logger.info("Training Random Forest for feature importance...")
            rf = RandomForestClassifier(
                n_estimators=self.config.N_ESTIMATORS,
                random_state=self.config.RANDOM_STATE,
                n_jobs=self.config.N_JOBS,
            )
            rf.fit(X_train, y_train)
            feature_importance['rf_importance'] = dict(
                zip(numeric_features, rf.feature_importances_)
            )

            # Gradient Boosting
            self.logger.info("Training Gradient Boosting for feature importance...")
            gb = GradientBoostingClassifier(
                n_estimators=self.config.N_ESTIMATORS,
                random_state=self.config.RANDOM_STATE,
            )
            gb.fit(X_train, y_train)
            feature_importance['gb_importance'] = dict(
                zip(numeric_features, gb.feature_importances_)
            )

            # Mutual Information
            self.logger.info("Calculating mutual information...")
            mi = mutual_info_classif(
                X_train, y_train, random_state=self.config.RANDOM_STATE
            )
            feature_importance['mutual_info'] = dict(zip(numeric_features, mi))

            print(f"\n📊 Top {self.config.TOP_N_FEATURES} Features (Random Forest):")
            for feat, imp in sorted(
                feature_importance['rf_importance'].items(),
                key=lambda x: x[1], reverse=True
            )[:self.config.TOP_N_FEATURES]:
                print(f"   • {feat}: {imp:.4f}")

        except Exception as e:
            self.logger.error(f"Error in supervised feature importance: {str(e)}")

    else:
        print("\n📊 UNSUPERVISED MODE — Variance-based importance")
        if numeric_features:
            try:
                X = df_ml[numeric_features].fillna(df_ml[numeric_features].mean())
                variances = X.var()
                feature_importance['variance_analysis'] = variances.to_dict()
                print(f"\n📊 Top {self.config.TOP_N_FEATURES} Features by Variance:")
                for feat, var in (
                    variances.sort_values(ascending=False)
                    .head(self.config.TOP_N_FEATURES)
                    .items()
                ):
                    print(f"   • {feat}: {var:.4f}")
            except Exception as e:
                self.logger.error(f"Error in variance analysis: {str(e)}")

    self.results['feature_importance'] = feature_importance
    self._ckpt_save("feature_importance", feature_importance)
    self.logger.info("Feature importance analysis complete")
    return feature_importance


# ==============================================================================
# 9. PCA — DIMENSIONALITY REDUCTION
# ==============================================================================

def analyze_pca(self, numerical_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    PCA on leak-free numeric columns.
    Reports how many components explain config.PCA_VARIANCE_THRESHOLD of variance.
    """
    print_section_header("8. DIMENSIONALITY REDUCTION (PCA)", logger=self.logger)

    cached = self._ckpt_load("pca_analysis")
    if cached is not None:
        self.results['pca_analysis'] = cached
        return cached

    leaky = self._leaky()

    if numerical_cols is None:
        numerical_cols = [
            c for c in get_numeric_columns(self.df, exclude=['is_fraud'])
            if c not in leaky
        ]

    pca_analysis = {}

    if len(numerical_cols) < 2:
        self.logger.warning("Need ≥2 numerical columns for PCA")
        return pca_analysis

    try:
        X        = self.df[numerical_cols].fillna(self.df[numerical_cols].mean())
        X_scaled = StandardScaler().fit_transform(X)

        pca    = PCA()
        pca.fit(X_scaled)

        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n95    = int(np.argmax(cumvar >= self.config.PCA_VARIANCE_THRESHOLD) + 1)

        pca_analysis = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance':      cumvar.tolist(),
            'n_components_95':          n95,
            'total_components':         len(pca.explained_variance_ratio_),
            'reduction_potential_pct':  float((1 - n95 / len(numerical_cols)) * 100),
        }

        # Store a 2D projection for visualisation (capped to MAX_PCA_POINTS)
        pca_2d = PCA(n_components=2).fit_transform(X_scaled)
        mp     = min(self.config.MAX_PCA_POINTS, len(pca_2d))
        pca_analysis['pca_2d_data'] = pca_2d[:mp].tolist()

        print(f"\n📊 PCA Results:")
        print(f"   • Components for {self.config.PCA_VARIANCE_THRESHOLD*100:.0f}% variance: "
              f"{n95}/{len(numerical_cols)}")
        print(f"   • Reduction potential: {pca_analysis['reduction_potential_pct']:.1f}%")

    except Exception as e:
        self.logger.error(f"Error in PCA: {str(e)}")

    self.results['pca_analysis'] = pca_analysis
    self._ckpt_save("pca_analysis", pca_analysis)
    self.logger.info("PCA analysis complete")
    return pca_analysis


# ==============================================================================
# 10. CLUSTERING
# ==============================================================================

def analyze_clustering(self, numerical_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    K-Means clustering on a stratified sample.
    Silhouette score computed on a further subsample (SILHOUETTE_SAMPLE) to
    avoid the O(n²) memory allocation that caused OOM on the full 5M dataset.
    """
    print_section_header("9. CLUSTERING ANALYSIS", logger=self.logger)

    cached = self._ckpt_load("clustering_analysis")
    if cached is not None:
        self.results['clustering_analysis'] = cached
        return cached

    leaky = self._leaky()

    if numerical_cols is None:
        numerical_cols = [
            c for c in get_numeric_columns(self.df, exclude=['is_fraud'])
            if c not in leaky
        ]

    clustering_analysis = {}

    if len(numerical_cols) < 2:
        self.logger.warning("Need ≥2 numerical columns for clustering")
        return clustering_analysis

    try:
        # Use stratified sample to avoid OOM
        df_sample = self._get_sample()
        X         = df_sample[numerical_cols].fillna(df_sample[numerical_cols].mean())
        X_scaled  = StandardScaler().fit_transform(X)

        max_k    = min(self.config.MAX_CLUSTERS, len(df_sample))
        inertias = []
        sil_scores = []

        self.logger.info(
            f"K-means on {len(df_sample):,} rows, "
            f"k={self.config.MIN_CLUSTERS}…{max_k - 1}"
        )

        for k in range(self.config.MIN_CLUSTERS, max_k):
            km     = KMeans(n_clusters=k, random_state=self.config.RANDOM_STATE, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)

            # Silhouette with metric='euclidean' avoids materialising the
            # full pairwise distance matrix (O(n²) memory). Instead sklearn
            # computes distances on-the-fly in chunks, so 10k rows is fine
            # even when a single contiguous 763MB block can't be allocated.
            try:
                ss  = min(self.config.SILHOUETTE_SAMPLE, len(X_scaled))
                idx = np.random.default_rng(self.config.RANDOM_STATE).choice(
                    len(X_scaled), size=ss, replace=False
                )
                sil_scores.append(
                    float(silhouette_score(
                        X_scaled[idx], labels[idx], metric='euclidean'
                    ))
                )
            except Exception as e:
                self.logger.warning(f"Silhouette score failed for k={k}: {e}")
                sil_scores.append(0.0)

        optimal_k     = sil_scores.index(max(sil_scores)) + self.config.MIN_CLUSTERS
        cluster_labels = KMeans(
            n_clusters=optimal_k, random_state=self.config.RANDOM_STATE, n_init=10
        ).fit_predict(X_scaled)

        mp = min(self.config.MAX_CLUSTER_POINTS, len(cluster_labels))
        clustering_analysis = {
            'optimal_k':          int(optimal_k),
            'sample_size_used':   len(df_sample),
            'inertias':           inertias,
            'silhouette_scores':  sil_scores,
            'cluster_labels':     cluster_labels[:mp].tolist(),
            'cluster_sizes': {
                int(k): int(v)
                for k, v in pd.Series(cluster_labels).value_counts().items()
            },
        }

        best_sil = sil_scores[optimal_k - self.config.MIN_CLUSTERS]
        print(f"\n📊 Clustering Results:")
        print(f"   Optimal clusters : {optimal_k}")
        print(f"   Silhouette score : {best_sil:.4f}")
        for cluster_id, size in sorted(clustering_analysis['cluster_sizes'].items()):
            pct = safe_divide(size, len(cluster_labels), 0) * 100
            print(f"   Cluster {cluster_id}: {size} ({pct:.1f}%)")

    except Exception as e:
        self.logger.error(f"Error in clustering analysis: {str(e)}")

    self.results['clustering_analysis'] = clustering_analysis
    self._ckpt_save("clustering_analysis", clustering_analysis)
    self.logger.info("Clustering analysis complete")
    return clustering_analysis