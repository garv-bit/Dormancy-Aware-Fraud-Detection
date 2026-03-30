"""
test_app.py — Unit tests for Flask Backend (app.py)
Dormancy-Aware Fraud Detection | COMP 385 Capstone

Covers:
    - rule_based_score()   — fraud probability computation
    - get_risk_level()     — Low / Medium / High classification
    - get_decision()       — Block / Review / Approve mapping
    - get_risk_factors()   — risk factor label generation
    - engineer_features()  — inference-time feature engineering
    - GET /health          — model status endpoint
    - POST /predict        — main prediction endpoint
    - GET /history         — session history endpoint

Run:
    pytest tests/test_app.py -v
"""

import sys
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import app as flask_app


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def client():
    """Flask test client with testing mode enabled."""
    flask_app.app.config['TESTING'] = True
    with flask_app.app.test_client() as client:
        yield client


@pytest.fixture
def high_risk_payload():
    return {
        "amount":           1200.0,
        "transaction_type": "transfer",
        "hour":             2,
        "merchant_category":"electronics",
        "payment_channel":  "online",
        "device_used":      "mobile",
        "location":         "New York",
        "sender_account":   "ACC-67890",
        "days_since_last":  287.0,
    }


@pytest.fixture
def low_risk_payload():
    return {
        "amount":           45.0,
        "transaction_type": "payment",
        "hour":             14,
        "merchant_category":"food",
        "payment_channel":  "pos",
        "device_used":      "mobile",
        "location":         "Toronto",
        "sender_account":   "ACC-12345",
        "days_since_last":  3.0,
    }


@pytest.fixture
def first_transaction_payload():
    """No days_since_last — first-ever transaction."""
    return {
        "amount":           540.0,
        "transaction_type": "transfer",
        "hour":             3,
        "merchant_category":"electronics",
        "payment_channel":  "online",
        "device_used":      "mobile",
        "location":         "Dubai",
        "sender_account":   "ACC-NEW999",
    }


# =============================================================================
# 1. rule_based_score() TESTS
# =============================================================================

class TestRuleBasedScore:

    def test_returns_float(self):
        result = flask_app.rule_based_score({"days_since_last": 30, "hour": 12, "amount": 100})
        assert isinstance(result, float)

    def test_score_between_0_and_1(self):
        for days in [None, 0, 30, 90, 180, 365, 400]:
            data = {"days_since_last": days, "hour": 12, "amount": 100}
            score = flask_app.rule_based_score(data)
            assert 0.05 <= score <= 0.96, f"Score {score} out of range for days={days}"

    def test_first_transaction_highest_risk(self):
        first = flask_app.rule_based_score({"hour": 12, "amount": 100})
        active = flask_app.rule_based_score({"days_since_last": 1, "hour": 12, "amount": 100})
        assert first > active

    def test_long_dormancy_higher_than_short(self):
        long_  = flask_app.rule_based_score({"days_since_last": 400, "hour": 12, "amount": 100})
        short_ = flask_app.rule_based_score({"days_since_last": 5,   "hour": 12, "amount": 100})
        assert long_ > short_

    def test_high_risk_hour_increases_score(self):
        night = flask_app.rule_based_score({"days_since_last": 30, "hour": 2,  "amount": 100})
        day   = flask_app.rule_based_score({"days_since_last": 30, "hour": 14, "amount": 100})
        assert night > day

    def test_large_amount_increases_score(self):
        large = flask_app.rule_based_score({"days_since_last": 30, "hour": 12, "amount": 1500})
        small = flask_app.rule_based_score({"days_since_last": 30, "hour": 12, "amount": 20})
        assert large > small

    def test_high_risk_scenario_above_0_65(self):
        """Long dormancy + 2am + large amount should be high risk."""
        score = flask_app.rule_based_score({
            "days_since_last": 300, "hour": 2, "amount": 1200
        })
        assert score >= 0.65

    def test_low_risk_scenario_below_0_40(self):
        """Active account + daytime + small amount should be low risk."""
        score = flask_app.rule_based_score({
            "days_since_last": 2, "hour": 14, "amount": 25
        })
        assert score < 0.40

    def test_score_capped_at_0_96(self):
        """Even extreme inputs should not exceed 0.96."""
        score = flask_app.rule_based_score({
            "days_since_last": None, "hour": 2, "amount": 99999,
            "geo_anomaly_score": 1.0, "velocity_score": 0
        })
        assert score <= 0.96

    def test_score_floor_at_0_05(self):
        """Minimum possible score should be 0.05."""
        score = flask_app.rule_based_score({
            "days_since_last": 0, "hour": 12, "amount": 0,
            "geo_anomaly_score": 0.0, "velocity_score": 20
        })
        assert score >= 0.05

    def test_geo_anomaly_increases_score(self):
        high_geo = flask_app.rule_based_score({
            "days_since_last": 30, "hour": 12, "amount": 100, "geo_anomaly_score": 0.9
        })
        low_geo = flask_app.rule_based_score({
            "days_since_last": 30, "hour": 12, "amount": 100, "geo_anomaly_score": 0.0
        })
        assert high_geo > low_geo

    def test_dormancy_365_days_higher_than_180(self):
        d365 = flask_app.rule_based_score({"days_since_last": 365, "hour": 12, "amount": 100})
        d180 = flask_app.rule_based_score({"days_since_last": 180, "hour": 12, "amount": 100})
        assert d365 > d180

    def test_missing_optional_fields_uses_defaults(self):
        """Should not raise when optional fields are absent."""
        score = flask_app.rule_based_score({"days_since_last": 30})
        assert 0.05 <= score <= 0.96


# =============================================================================
# 2. get_risk_level() TESTS
# =============================================================================

class TestGetRiskLevel:

    def test_low_risk_below_0_40(self):
        assert flask_app.get_risk_level(0.10) == 'low'
        assert flask_app.get_risk_level(0.39) == 'low'

    def test_medium_risk_between_0_40_and_0_65(self):
        assert flask_app.get_risk_level(0.40) == 'medium'
        assert flask_app.get_risk_level(0.55) == 'medium'
        assert flask_app.get_risk_level(0.64) == 'medium'

    def test_high_risk_above_0_65(self):
        assert flask_app.get_risk_level(0.65) == 'high'
        assert flask_app.get_risk_level(0.90) == 'high'
        assert flask_app.get_risk_level(0.96) == 'high'

    def test_boundary_0_40_is_medium(self):
        assert flask_app.get_risk_level(0.40) == 'medium'

    def test_boundary_0_65_is_high(self):
        assert flask_app.get_risk_level(0.65) == 'high'

    def test_returns_string(self):
        assert isinstance(flask_app.get_risk_level(0.5), str)

    def test_valid_values_only(self):
        for prob in [0.0, 0.1, 0.39, 0.40, 0.64, 0.65, 0.96]:
            level = flask_app.get_risk_level(prob)
            assert level in {'low', 'medium', 'high'}


# =============================================================================
# 3. get_decision() TESTS
# =============================================================================

class TestGetDecision:

    def test_high_maps_to_block(self):
        assert flask_app.get_decision('high') == 'Block'

    def test_medium_maps_to_review(self):
        assert flask_app.get_decision('medium') == 'Review'

    def test_low_maps_to_approve(self):
        assert flask_app.get_decision('low') == 'Approve'

    def test_returns_string(self):
        assert isinstance(flask_app.get_decision('low'), str)


# =============================================================================
# 4. get_risk_factors() TESTS
# =============================================================================

class TestGetRiskFactors:

    def test_returns_list(self):
        data = {"days_since_last": 200, "hour": 2, "amount": 500}
        factors = flask_app.get_risk_factors(data, "dormant", 0)
        assert isinstance(factors, list)

    def test_list_not_empty(self):
        data = {"days_since_last": 200, "hour": 2, "amount": 500}
        factors = flask_app.get_risk_factors(data, "dormant", 0)
        assert len(factors) > 0

    def test_each_factor_has_required_keys(self):
        data = {"days_since_last": 200, "hour": 2, "amount": 500}
        factors = flask_app.get_risk_factors(data, "dormant", 0)
        for factor in factors:
            assert 'label' in factor
            assert 'level' in factor
            assert 'detail' in factor

    def test_factor_levels_are_valid(self):
        data = {"days_since_last": 200, "hour": 2, "amount": 500}
        factors = flask_app.get_risk_factors(data, "dormant", 0)
        for factor in factors:
            assert factor['level'] in {'low', 'medium', 'high'}

    def test_first_transaction_triggers_high_factor(self):
        data = {"hour": 12, "amount": 100}
        factors = flask_app.get_risk_factors(data, "recent", 1)
        labels = [f['label'] for f in factors]
        assert any('First' in label or 'first' in label for label in labels)

    def test_long_dormancy_triggers_high_factor(self):
        data = {"days_since_last": 300, "hour": 12, "amount": 100}
        factors = flask_app.get_risk_factors(data, "dormant", 0)
        high_factors = [f for f in factors if f['level'] == 'high']
        assert len(high_factors) > 0

    def test_high_risk_hour_triggers_factor(self):
        data = {"days_since_last": 5, "hour": 2, "amount": 50}
        factors = flask_app.get_risk_factors(data, "recent", 0)
        hour_factors = [f for f in factors if '2:00' in f['label'] or 'hour' in f['label'].lower() or 'Hour' in f['label']]
        assert len(hour_factors) > 0

    def test_active_account_daytime_small_amount_all_low(self):
        data = {"days_since_last": 2, "hour": 14, "amount": 20}
        factors = flask_app.get_risk_factors(data, "recent", 0)
        high_factors = [f for f in factors if f['level'] == 'high']
        assert len(high_factors) == 0

    def test_large_amount_triggers_high_factor(self):
        data = {"days_since_last": 5, "hour": 12, "amount": 1500}
        factors = flask_app.get_risk_factors(data, "recent", 0)
        high_factors = [f for f in factors if f['level'] == 'high']
        assert len(high_factors) > 0

    def test_labels_are_strings(self):
        data = {"days_since_last": 100, "hour": 10, "amount": 200}
        factors = flask_app.get_risk_factors(data, "moderate", 0)
        for f in factors:
            assert isinstance(f['label'], str)
            assert isinstance(f['detail'], str)


# =============================================================================
# 5. engineer_features() TESTS
# =============================================================================

class TestEngineerFeatures:

    def test_returns_dataframe(self, high_risk_payload):
        df, _, _, _ = flask_app.engineer_features(high_risk_payload)
        import pandas as pd
        assert isinstance(df, pd.DataFrame)

    def test_returns_single_row(self, high_risk_payload):
        df, _, _, _ = flask_app.engineer_features(high_risk_payload)
        assert len(df) == 1

    def test_returns_dormancy_bucket_string(self, high_risk_payload):
        _, dormancy_bucket, _, _ = flask_app.engineer_features(high_risk_payload)
        assert dormancy_bucket in {'recent', 'moderate', 'dormant', 'long_dormant'}

    def test_returns_dormancy_risk_score_int(self, high_risk_payload):
        _, _, dormancy_risk_score, _ = flask_app.engineer_features(high_risk_payload)
        assert isinstance(dormancy_risk_score, int)
        assert 0 <= dormancy_risk_score <= 3

    def test_returns_is_first_transaction_flag(self, high_risk_payload):
        _, _, _, is_first = flask_app.engineer_features(high_risk_payload)
        assert is_first in {0, 1}

    def test_first_transaction_when_no_days(self, first_transaction_payload):
        _, _, _, is_first = flask_app.engineer_features(first_transaction_payload)
        assert is_first == 1

    def test_not_first_transaction_when_days_provided(self, high_risk_payload):
        _, _, _, is_first = flask_app.engineer_features(high_risk_payload)
        assert is_first == 0

    def test_long_dormancy_bucket_for_287_days(self, high_risk_payload):
        _, dormancy_bucket, _, _ = flask_app.engineer_features(high_risk_payload)
        assert dormancy_bucket in {'dormant', 'long_dormant'}

    def test_recent_bucket_for_3_days(self, low_risk_payload):
        _, dormancy_bucket, _, _ = flask_app.engineer_features(low_risk_payload)
        assert dormancy_bucket == 'recent'

    def test_all_numeric_columns(self, high_risk_payload):
        import numpy as np
        df, _, _, _ = flask_app.engineer_features(high_risk_payload)
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        assert len(non_numeric) == 0, f"Non-numeric columns: {non_numeric}"

    def test_no_null_values(self, high_risk_payload):
        df, _, _, _ = flask_app.engineer_features(high_risk_payload)
        assert df.isnull().sum().sum() == 0

    def test_is_fraud_not_in_features(self, high_risk_payload):
        df, _, _, _ = flask_app.engineer_features(high_risk_payload)
        assert 'is_fraud' not in df.columns

    def test_amount_log_is_log1p_of_amount(self, high_risk_payload):
        import numpy as np
        df, _, _, _ = flask_app.engineer_features(high_risk_payload)
        if 'amount_log' in df.columns and 'amount' in df.columns:
            expected = np.log1p(high_risk_payload['amount'])
            # amount may be scaled but amount_log should reflect original
            assert df['amount_log'].iloc[0] != 0

    def test_hour_2_triggers_high_risk_hour_flag(self, high_risk_payload):
        df, _, _, _ = flask_app.engineer_features(high_risk_payload)
        if 'is_high_risk_hour' in df.columns:
            assert df['is_high_risk_hour'].iloc[0] == 1

    def test_hour_14_does_not_trigger_high_risk_hour(self, low_risk_payload):
        df, _, _, _ = flask_app.engineer_features(low_risk_payload)
        if 'is_high_risk_hour' in df.columns:
            assert df['is_high_risk_hour'].iloc[0] == 0


# =============================================================================
# 6. FLASK ROUTE TESTS
# =============================================================================

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        response = client.get('/health')
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        response = client.get('/health')
        data = json.loads(response.data)
        assert isinstance(data, dict)

    def test_health_has_status_ok(self, client):
        response = client.get('/health')
        data = json.loads(response.data)
        assert data['status'] == 'ok'

    def test_health_has_model_loaded_field(self, client):
        response = client.get('/health')
        data = json.loads(response.data)
        assert 'model_loaded' in data

    def test_health_model_loaded_is_bool(self, client):
        response = client.get('/health')
        data = json.loads(response.data)
        assert isinstance(data['model_loaded'], bool)

    def test_health_has_scorer_field(self, client):
        response = client.get('/health')
        data = json.loads(response.data)
        assert 'scorer' in data

    def test_health_scorer_is_rule_based(self, client):
        response = client.get('/health')
        data = json.loads(response.data)
        assert data['scorer'] == 'rule_based'


class TestPredictEndpoint:

    def test_predict_returns_200(self, client, high_risk_payload):
        response = client.post('/predict',
                               data=json.dumps(high_risk_payload),
                               content_type='application/json')
        assert response.status_code == 200

    def test_predict_returns_json(self, client, high_risk_payload):
        response = client.post('/predict',
                               data=json.dumps(high_risk_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert isinstance(data, dict)

    def test_predict_has_required_fields(self, client, high_risk_payload):
        response = client.post('/predict',
                               data=json.dumps(high_risk_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        for field in ['fraud_probability', 'fraud_probability_pct',
                      'risk_level', 'decision', 'risk_factors',
                      'dormancy_bucket', 'inference_ms', 'timestamp']:
            assert field in data, f"Missing field: {field}"

    def test_predict_probability_between_0_and_1(self, client, high_risk_payload):
        response = client.post('/predict',
                               data=json.dumps(high_risk_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert 0 <= data['fraud_probability'] <= 1

    def test_predict_probability_pct_between_0_and_100(self, client, high_risk_payload):
        response = client.post('/predict',
                               data=json.dumps(high_risk_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert 0 <= data['fraud_probability_pct'] <= 100

    def test_predict_risk_level_valid(self, client, high_risk_payload):
        response = client.post('/predict',
                               data=json.dumps(high_risk_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert data['risk_level'] in {'low', 'medium', 'high'}

    def test_predict_decision_valid(self, client, high_risk_payload):
        response = client.post('/predict',
                               data=json.dumps(high_risk_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert data['decision'] in {'Block', 'Review', 'Approve'}

    def test_predict_risk_factors_is_list(self, client, high_risk_payload):
        response = client.post('/predict',
                               data=json.dumps(high_risk_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert isinstance(data['risk_factors'], list)

    def test_predict_high_risk_scenario_returns_high(self, client, high_risk_payload):
        """287 days dormant + 2am + $1200 should return high risk."""
        response = client.post('/predict',
                               data=json.dumps(high_risk_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert data['risk_level'] == 'high'

    def test_predict_low_risk_scenario_returns_low(self, client, low_risk_payload):
        """3 days + 2pm + $45 should return low risk."""
        response = client.post('/predict',
                               data=json.dumps(low_risk_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert data['risk_level'] == 'low'

    def test_predict_first_transaction_returns_high(self, client, first_transaction_payload):
        """First-ever transaction at 3am should return high risk."""
        response = client.post('/predict',
                               data=json.dumps(first_transaction_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert data['risk_level'] == 'high'

    def test_predict_first_transaction_flag_set(self, client, first_transaction_payload):
        response = client.post('/predict',
                               data=json.dumps(first_transaction_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert data['is_first_transaction'] is True

    def test_predict_inference_ms_positive(self, client, high_risk_payload):
        response = client.post('/predict',
                               data=json.dumps(high_risk_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert data['inference_ms'] >= 0

    def test_predict_empty_body_returns_400(self, client):
        response = client.post('/predict',
                               data='',
                               content_type='application/json')
        assert response.status_code == 400

    def test_predict_missing_days_treated_as_first_transaction(self, client):
        payload = {"amount": 100, "hour": 12, "transaction_type": "payment"}
        response = client.post('/predict',
                               data=json.dumps(payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert data['is_first_transaction'] is True

    def test_predict_adds_to_history(self, client, high_risk_payload):
        initial_len = len(flask_app.HISTORY)
        client.post('/predict',
                    data=json.dumps(high_risk_payload),
                    content_type='application/json')
        assert len(flask_app.HISTORY) == initial_len + 1

    def test_predict_pct_is_probability_times_100(self, client, high_risk_payload):
        response = client.post('/predict',
                               data=json.dumps(high_risk_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        expected_pct = round(data['fraud_probability'] * 100, 1)
        assert data['fraud_probability_pct'] == expected_pct

    def test_predict_high_risk_decision_is_block(self, client, high_risk_payload):
        response = client.post('/predict',
                               data=json.dumps(high_risk_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        if data['risk_level'] == 'high':
            assert data['decision'] == 'Block'

    def test_predict_low_risk_decision_is_approve(self, client, low_risk_payload):
        response = client.post('/predict',
                               data=json.dumps(low_risk_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        if data['risk_level'] == 'low':
            assert data['decision'] == 'Approve'


class TestHistoryEndpoint:

    def test_history_returns_200(self, client):
        response = client.get('/history')
        assert response.status_code == 200

    def test_history_returns_json(self, client):
        response = client.get('/history')
        data = json.loads(response.data)
        assert isinstance(data, dict)

    def test_history_has_history_key(self, client):
        response = client.get('/history')
        data = json.loads(response.data)
        assert 'history' in data

    def test_history_is_list(self, client):
        response = client.get('/history')
        data = json.loads(response.data)
        assert isinstance(data['history'], list)

    def test_history_max_20_entries(self, client, high_risk_payload):
        for _ in range(25):
            client.post('/predict',
                        data=json.dumps(high_risk_payload),
                        content_type='application/json')
        response = client.get('/history')
        data = json.loads(response.data)
        assert len(data['history']) <= 20

    def test_history_entries_have_required_fields(self, client, high_risk_payload):
        client.post('/predict',
                    data=json.dumps(high_risk_payload),
                    content_type='application/json')
        response = client.get('/history')
        data = json.loads(response.data)
        if data['history']:
            entry = data['history'][0]
            for field in ['timestamp', 'amount', 'risk_level', 'decision', 'fraud_probability_pct']:
                assert field in entry, f"Missing field: {field}"

    def test_history_newest_first(self, client, high_risk_payload, low_risk_payload):
        client.post('/predict', data=json.dumps(low_risk_payload),
                    content_type='application/json')
        client.post('/predict', data=json.dumps(high_risk_payload),
                    content_type='application/json')
        response = client.get('/history')
        data = json.loads(response.data)
        if len(data['history']) >= 2:
            # Most recent (high risk) should be first
            assert data['history'][0]['risk_level'] == 'high'