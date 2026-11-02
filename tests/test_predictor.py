import pytest

from app.predictor import ArtifactError, InputError, MigrainePredictor

from .helpers import CLASSES, sample_features


def test_predict_returns_valid_class_and_probabilities(artifact_dir):
    predictor = MigrainePredictor.from_dir(artifact_dir)
    result = predictor.predict(sample_features())
    assert result["predicted_class"] in CLASSES
    assert 0 < result["confidence"] <= 1
    assert abs(sum(result["probabilities"].values()) - 1) < 0.01


def test_missing_feature_is_rejected(artifact_dir):
    predictor = MigrainePredictor.from_dir(artifact_dir)
    features = sample_features()
    features.pop("Age")
    with pytest.raises(InputError):
        predictor.predict(features)


def test_unexpected_feature_is_rejected(artifact_dir):
    predictor = MigrainePredictor.from_dir(artifact_dir)
    features = sample_features()
    features["Not_A_Feature"] = 1
    with pytest.raises(InputError):
        predictor.predict(features)


def test_missing_artifacts_raise(tmp_path):
    with pytest.raises(ArtifactError):
        MigrainePredictor.from_dir(tmp_path)
