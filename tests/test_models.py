"""
test_models.py — Unit tests for Model Layer
Dormancy-Aware Fraud Detection | COMP 385 Capstone

Covers:
    - BaseModel interface (train, predict, save, load, threshold)
    - LogisticModel
    - RandomForestModel
    - XGBoostModel (including scale_pos_weight computation)
    - Evaluator (PR-AUC, threshold optimization, compare, save)

Run:
    pytest tests/test_models.py -v
"""

import sys
import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "models"))

from models.base_model import BaseModel
from models.logistic_model import LogisticModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.evaluator import Evaluator


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def synthetic_data():
    """Small synthetic dataset: 200 rows, 5 features, ~10% fraud."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        'dormancy_risk_score':         np.random.randint(0, 4, n).astype(float),
        'hour':                        np.random.randint(0, 24, n).astype(float),
        'amount_log':                  np.random.exponential(5, n),
        'is_first_transaction':        np.random.choice([0, 1], n, p=[0.82, 0.18]).astype(float),
        'is_high_risk_hour':           np.random.choice([0, 1], n, p=[0.7, 0.3]).astype(float),
    })
    y = pd.Series(np.random.choice([0, 1], n, p=[0.90, 0.10]), name='is_fraud')
    return X, y


@pytest.fixture
def train_test_split(synthetic_data):
    X, y = synthetic_data
    split = int(len(X) * 0.8)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


@pytest.fixture
def trained_lr(train_test_split, tmp_path):
    X_train, _, y_train, _ = train_test_split
    model = LogisticModel(output_dir=str(tmp_path))
    model.train(X_train, y_train)
    return model


@pytest.fixture
def trained_rf(train_test_split, tmp_path):
    X_train, _, y_train, _ = train_test_split
    model = RandomForestModel(output_dir=str(tmp_path))
    model.train(X_train, y_train)
    return model


@pytest.fixture
def trained_xgb(train_test_split, tmp_path):
    X_train, _, y_train, _ = train_test_split
    model = XGBoostModel(output_dir=str(tmp_path))
    model.train(X_train, y_train)
    return model


# =============================================================================
# 1. BASE MODEL TESTS
# =============================================================================

class TestBaseModel:

    def test_base_model_is_abstract(self):
        """BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel(name="test")

    def test_logistic_model_is_base_model(self, tmp_path):
        model = LogisticModel(output_dir=str(tmp_path))
        assert isinstance(model, BaseModel)

    def test_random_forest_is_base_model(self, tmp_path):
        model = RandomForestModel(output_dir=str(tmp_path))
        assert isinstance(model, BaseModel)

    def test_xgboost_is_base_model(self, tmp_path):
        model = XGBoostModel(output_dir=str(tmp_path))
        assert isinstance(model, BaseModel)

    def test_is_fitted_false_before_training(self, tmp_path):
        model = LogisticModel(output_dir=str(tmp_path))
        assert model.is_fitted is False

    def test_is_fitted_true_after_training(self, trained_lr):
        assert trained_lr.is_fitted is True

    def test_predict_proba_raises_if_not_fitted(self, tmp_path, synthetic_data):
        X, _ = synthetic_data
        model = LogisticModel(output_dir=str(tmp_path))
        with pytest.raises(RuntimeError):
            model.predict_proba(X)

    def test_predict_raises_if_not_fitted(self, tmp_path, synthetic_data):
        X, _ = synthetic_data
        model = LogisticModel(output_dir=str(tmp_path))
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_set_threshold_updates_threshold(self, trained_lr):
        trained_lr.set_threshold(0.3)
        assert trained_lr.threshold == 0.3

    def test_default_threshold_is_0_5(self, tmp_path):
        model = LogisticModel(output_dir=str(tmp_path))
        assert model.threshold == 0.5

    def test_save_creates_joblib_file(self, trained_lr, tmp_path):
        path = trained_lr.save()
        assert Path(path).exists()

    def test_save_returns_correct_path(self, trained_lr, tmp_path):
        path = trained_lr.save()
        assert 'logistic_regression.joblib' in str(path)

    def test_load_returns_model_object(self, trained_lr, tmp_path):
        path = trained_lr.save()
        loaded = BaseModel.load(str(path))
        assert loaded is not None
        assert loaded.is_fitted is True

    def test_loaded_model_predicts_same_as_original(self, trained_lr, tmp_path, train_test_split):
        _, X_test, _, _ = train_test_split
        path = trained_lr.save()
        loaded = BaseModel.load(str(path))
        orig_preds  = trained_lr.predict_proba(X_test)
        loaded_preds = loaded.predict_proba(X_test)
        np.testing.assert_array_almost_equal(orig_preds, loaded_preds)

    def test_repr_contains_model_name(self, tmp_path):
        model = LogisticModel(output_dir=str(tmp_path))
        assert 'logistic_regression' in repr(model)

    def test_output_dir_created_on_init(self, tmp_path):
        new_dir = str(tmp_path / "new_models")
        LogisticModel(output_dir=new_dir)
        assert Path(new_dir).exists()


# =============================================================================
# 2. PREDICT_PROBA TESTS
# =============================================================================

class TestPredictProba:

    def test_returns_1d_array(self, trained_lr, train_test_split):
        _, X_test, _, _ = train_test_split
        probas = trained_lr.predict_proba(X_test)
        assert probas.ndim == 1

    def test_probas_between_0_and_1(self, trained_lr, train_test_split):
        _, X_test, _, _ = train_test_split
        probas = trained_lr.predict_proba(X_test)
        assert (probas >= 0).all() and (probas <= 1).all()

    def test_probas_length_matches_input(self, trained_lr, train_test_split):
        _, X_test, _, _ = train_test_split
        probas = trained_lr.predict_proba(X_test)
        assert len(probas) == len(X_test)

    def test_single_row_input(self, trained_lr, train_test_split):
        _, X_test, _, _ = train_test_split
        proba = trained_lr.predict_proba(X_test.iloc[:1])
        assert len(proba) == 1
        assert 0 <= proba[0] <= 1


# =============================================================================
# 3. PREDICT TESTS
# =============================================================================

class TestPredict:

    def test_returns_binary_array(self, trained_lr, train_test_split):
        _, X_test, _, _ = train_test_split
        preds = trained_lr.predict(X_test)
        assert set(preds).issubset({0, 1})

    def test_predict_length_matches_input(self, trained_lr, train_test_split):
        _, X_test, _, _ = train_test_split
        preds = trained_lr.predict(X_test)
        assert len(preds) == len(X_test)

    def test_low_threshold_predicts_more_fraud(self, trained_lr, train_test_split):
        _, X_test, _, _ = train_test_split
        preds_low  = trained_lr.predict(X_test, threshold=0.1)
        preds_high = trained_lr.predict(X_test, threshold=0.9)
        assert preds_low.sum() >= preds_high.sum()

    def test_threshold_0_predicts_all_fraud(self, trained_lr, train_test_split):
        _, X_test, _, _ = train_test_split
        preds = trained_lr.predict(X_test, threshold=0.0)
        assert preds.sum() == len(X_test)

    def test_threshold_1_predicts_all_legit(self, trained_lr, train_test_split):
        _, X_test, _, _ = train_test_split
        preds = trained_lr.predict(X_test, threshold=1.0)
        assert preds.sum() == 0

    def test_uses_stored_threshold_by_default(self, trained_lr, train_test_split):
        _, X_test, _, _ = train_test_split
        trained_lr.set_threshold(0.1)
        preds_stored = trained_lr.predict(X_test)
        preds_explicit = trained_lr.predict(X_test, threshold=0.1)
        np.testing.assert_array_equal(preds_stored, preds_explicit)


# =============================================================================
# 4. LOGISTIC MODEL TESTS
# =============================================================================

class TestLogisticModel:

    def test_name_is_logistic_regression(self, tmp_path):
        model = LogisticModel(output_dir=str(tmp_path))
        assert model.name == 'logistic_regression'

    def test_build_returns_sklearn_lr(self, tmp_path):
        from sklearn.linear_model import LogisticRegression
        model = LogisticModel(output_dir=str(tmp_path))
        built = model.build()
        assert isinstance(built, LogisticRegression)

    def test_class_weight_is_balanced(self, tmp_path):
        model = LogisticModel(output_dir=str(tmp_path))
        built = model.build()
        assert built.class_weight == 'balanced'

    def test_train_returns_self(self, tmp_path, train_test_split):
        X_train, _, y_train, _ = train_test_split
        model = LogisticModel(output_dir=str(tmp_path))
        result = model.train(X_train, y_train)
        assert result is model

    def test_trained_model_not_none(self, trained_lr):
        assert trained_lr.model is not None


# =============================================================================
# 5. RANDOM FOREST MODEL TESTS
# =============================================================================

class TestRandomForestModel:

    def test_name_is_random_forest(self, tmp_path):
        model = RandomForestModel(output_dir=str(tmp_path))
        assert model.name == 'random_forest'

    def test_build_returns_sklearn_rf(self, tmp_path):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestModel(output_dir=str(tmp_path))
        built = model.build()
        assert isinstance(built, RandomForestClassifier)

    def test_class_weight_is_balanced_subsample(self, tmp_path):
        model = RandomForestModel(output_dir=str(tmp_path))
        built = model.build()
        assert built.class_weight == 'balanced_subsample'

    def test_n_estimators_is_300(self, tmp_path):
        model = RandomForestModel(output_dir=str(tmp_path))
        built = model.build()
        assert built.n_estimators == 300

    def test_train_returns_self(self, tmp_path, train_test_split):
        X_train, _, y_train, _ = train_test_split
        model = RandomForestModel(output_dir=str(tmp_path))
        result = model.train(X_train, y_train)
        assert result is model

    def test_feature_importances_sum_to_one(self, trained_rf, train_test_split):
        X_train, _, _, _ = train_test_split
        importances = trained_rf.model.feature_importances_
        assert abs(importances.sum() - 1.0) < 0.001


# =============================================================================
# 6. XGBOOST MODEL TESTS
# =============================================================================

class TestXGBoostModel:

    def test_name_is_xgboost(self, tmp_path):
        model = XGBoostModel(output_dir=str(tmp_path))
        assert model.name == 'xgboost'

    def test_build_returns_xgb_classifier(self, tmp_path):
        from xgboost import XGBClassifier
        model = XGBoostModel(output_dir=str(tmp_path))
        built = model.build()
        assert isinstance(built, XGBClassifier)

    def test_scale_pos_weight_computed_from_data(self, tmp_path, train_test_split):
        X_train, _, y_train, _ = train_test_split
        model = XGBoostModel(output_dir=str(tmp_path))
        model.train(X_train, y_train)
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        expected = round(n_neg / n_pos, 2)
        assert model._scale_pos_weight == expected

    def test_scale_pos_weight_positive(self, trained_xgb):
        assert trained_xgb._scale_pos_weight > 0

    def test_scale_pos_weight_gt_1_for_imbalanced_data(self, trained_xgb):
        """With ~10% fraud, scale_pos_weight should be > 1."""
        assert trained_xgb._scale_pos_weight > 1.0

    def test_get_feature_importance_returns_dict(self, trained_xgb, synthetic_data):
        X, _ = synthetic_data
        importance = trained_xgb.get_feature_importance(list(X.columns))
        assert isinstance(importance, dict)

    def test_get_feature_importance_keys_match_features(self, trained_xgb, synthetic_data):
        X, _ = synthetic_data
        importance = trained_xgb.get_feature_importance(list(X.columns))
        assert set(importance.keys()) == set(X.columns)

    def test_get_feature_importance_values_sum_to_one(self, trained_xgb, synthetic_data):
        X, _ = synthetic_data
        importance = trained_xgb.get_feature_importance(list(X.columns))
        total = sum(importance.values())
        assert abs(total - 1.0) < 0.001

    def test_get_feature_importance_sorted_descending(self, trained_xgb, synthetic_data):
        X, _ = synthetic_data
        importance = trained_xgb.get_feature_importance(list(X.columns))
        values = list(importance.values())
        assert values == sorted(values, reverse=True)

    def test_get_feature_importance_raises_if_not_fitted(self, tmp_path, synthetic_data):
        X, _ = synthetic_data
        model = XGBoostModel(output_dir=str(tmp_path))
        with pytest.raises(RuntimeError):
            model.get_feature_importance(list(X.columns))


# =============================================================================
# 7. EVALUATOR TESTS
# =============================================================================

class TestEvaluator:

    def test_evaluate_returns_dict(self, trained_lr, train_test_split):
        _, X_test, _, y_test = train_test_split
        evaluator = Evaluator()
        result = evaluator.evaluate(trained_lr, X_test, y_test)
        assert isinstance(result, dict)

    def test_evaluate_has_required_keys(self, trained_lr, train_test_split):
        _, X_test, _, y_test = train_test_split
        evaluator = Evaluator()
        result = evaluator.evaluate(trained_lr, X_test, y_test)
        for key in ['model', 'pr_auc', 'roc_auc', 'optimal_threshold',
                    'f1', 'recall', 'precision', 'false_positive_rate',
                    'confusion_matrix']:
            assert key in result, f"Missing key: {key}"

    def test_pr_auc_between_0_and_1(self, trained_lr, train_test_split):
        _, X_test, _, y_test = train_test_split
        evaluator = Evaluator()
        result = evaluator.evaluate(trained_lr, X_test, y_test)
        assert 0 <= result['pr_auc'] <= 1

    def test_roc_auc_between_0_and_1(self, trained_lr, train_test_split):
        _, X_test, _, y_test = train_test_split
        evaluator = Evaluator()
        result = evaluator.evaluate(trained_lr, X_test, y_test)
        assert 0 <= result['roc_auc'] <= 1

    def test_recall_between_0_and_1(self, trained_lr, train_test_split):
        _, X_test, _, y_test = train_test_split
        evaluator = Evaluator()
        result = evaluator.evaluate(trained_lr, X_test, y_test)
        assert 0 <= result['recall'] <= 1

    def test_precision_between_0_and_1(self, trained_lr, train_test_split):
        _, X_test, _, y_test = train_test_split
        evaluator = Evaluator()
        result = evaluator.evaluate(trained_lr, X_test, y_test)
        assert 0 <= result['precision'] <= 1

    def test_threshold_set_on_model_after_evaluate(self, trained_lr, train_test_split):
        _, X_test, _, y_test = train_test_split
        evaluator = Evaluator()
        result = evaluator.evaluate(trained_lr, X_test, y_test)
        assert trained_lr.threshold == pytest.approx(result['optimal_threshold'], abs=0.01)

    def test_confusion_matrix_has_correct_keys(self, trained_lr, train_test_split):
        _, X_test, _, y_test = train_test_split
        evaluator = Evaluator()
        result = evaluator.evaluate(trained_lr, X_test, y_test)
        cm = result['confusion_matrix']
        for key in ['tn', 'fp', 'fn', 'tp']:
            assert key in cm

    def test_confusion_matrix_sums_to_test_size(self, trained_lr, train_test_split):
        _, X_test, _, y_test = train_test_split
        evaluator = Evaluator()
        result = evaluator.evaluate(trained_lr, X_test, y_test)
        cm = result['confusion_matrix']
        total = cm['tn'] + cm['fp'] + cm['fn'] + cm['tp']
        assert total == len(y_test)

    def test_model_name_in_results(self, trained_lr, train_test_split):
        _, X_test, _, y_test = train_test_split
        evaluator = Evaluator()
        result = evaluator.evaluate(trained_lr, X_test, y_test)
        assert result['model'] == 'logistic_regression'

    def test_compare_returns_best_by_pr_auc(self, trained_lr, trained_rf, train_test_split):
        _, X_test, _, y_test = train_test_split
        evaluator = Evaluator()
        r1 = evaluator.evaluate(trained_lr, X_test, y_test)
        r2 = evaluator.evaluate(trained_rf, X_test, y_test)
        best = evaluator.compare([r1, r2])
        expected_best = r1 if r1['pr_auc'] >= r2['pr_auc'] else r2
        assert best['model'] == expected_best['model']

    def test_compare_with_single_model(self, trained_lr, train_test_split):
        _, X_test, _, y_test = train_test_split
        evaluator = Evaluator()
        result = evaluator.evaluate(trained_lr, X_test, y_test)
        best = evaluator.compare([result])
        assert best['model'] == 'logistic_regression'

    def test_save_results_creates_file(self, trained_lr, train_test_split, tmp_path):
        _, X_test, _, y_test = train_test_split
        evaluator = Evaluator()
        result = evaluator.evaluate(trained_lr, X_test, y_test)
        out_path = str(tmp_path / 'eval_results.joblib')
        evaluator.save_results([result], path=out_path)
        assert Path(out_path).exists()

    def test_save_results_loadable(self, trained_lr, train_test_split, tmp_path):
        _, X_test, _, y_test = train_test_split
        evaluator = Evaluator()
        result = evaluator.evaluate(trained_lr, X_test, y_test)
        out_path = str(tmp_path / 'eval_results.joblib')
        evaluator.save_results([result], path=out_path)
        loaded = joblib.load(out_path)
        assert isinstance(loaded, list)
        assert loaded[0]['model'] == 'logistic_regression'

    def test_feature_importance_included_for_xgboost(self, trained_xgb, train_test_split, synthetic_data):
        _, X_test, _, y_test = train_test_split
        X, _ = synthetic_data
        evaluator = Evaluator()
        result = evaluator.evaluate(trained_xgb, X_test, y_test, feature_names=list(X.columns))
        assert 'feature_importance' in result

    def test_optimal_threshold_between_0_and_1(self, trained_lr, train_test_split):
        _, X_test, _, y_test = train_test_split
        evaluator = Evaluator()
        result = evaluator.evaluate(trained_lr, X_test, y_test)
        assert 0 < result['optimal_threshold'] < 1

    def test_evaluate_all_three_models(self, trained_lr, trained_rf, trained_xgb, train_test_split):
        _, X_test, _, y_test = train_test_split
        evaluator = Evaluator()
        results = [
            evaluator.evaluate(trained_lr,  X_test, y_test),
            evaluator.evaluate(trained_rf,  X_test, y_test),
            evaluator.evaluate(trained_xgb, X_test, y_test),
        ]
        assert len(results) == 3
        names = [r['model'] for r in results]
        assert 'logistic_regression' in names
        assert 'random_forest' in names
        assert 'xgboost' in names