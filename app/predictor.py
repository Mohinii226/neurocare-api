"""Core prediction logic for NeuroCare (no web-framework dependency).

Loads the artifacts saved during training (model, scaler, label encoder and
feature order) and turns a dict of symptom codes into a migraine-subtype
prediction. Keeping this separate from FastAPI makes it easy to unit test.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# File names inside the artifact directory. Rename these (or the files) so they
# match what your training notebook saved.
DEFAULT_FILES = {
    "model": "migraine_model.pkl",
    "scaler": "migraine_scaler.pkl",
    "label_encoder": "migraine_label_encoder.pkl",
    "feature_columns": "migraine_feature_columns.pkl",
}


class ArtifactError(RuntimeError):
    """Raised when required model artifacts are missing or unreadable."""


class InputError(ValueError):
    """Raised when the request does not match the training feature set."""


class MigrainePredictor:
    def __init__(self, model, feature_columns, label_encoder=None, scaler=None):
        self.model = model
        self.feature_columns = list(feature_columns)
        self.label_encoder = label_encoder
        self.scaler = scaler

    @classmethod
    def from_dir(cls, directory, files=None):
        files = {**DEFAULT_FILES, **(files or {})}
        directory = Path(directory)

        def load(key, required):
            path = directory / files[key]
            if not path.exists():
                if required:
                    raise ArtifactError(f"Missing artifact: {path}")
                return None
            try:
                return joblib.load(path)
            except Exception as exc:  # unpickling errors vary widely
                raise ArtifactError(f"Could not load {path}: {exc}") from exc

        return cls(
            model=load("model", True),
            feature_columns=load("feature_columns", True),
            label_encoder=load("label_encoder", False),
            scaler=load("scaler", False),
        )

    def _class_names(self):
        classes = np.asarray(self.model.classes_)
        if self.label_encoder is not None:
            return [str(c) for c in self.label_encoder.inverse_transform(classes)]
        return [str(c) for c in classes]

    def predict(self, features):
        missing = [c for c in self.feature_columns if c not in features]
        extra = [k for k in features if k not in self.feature_columns]
        if missing or extra:
            raise InputError(f"Missing features: {missing}; unexpected features: {extra}")

        # Build the row in the exact order used during training.
        row = pd.DataFrame([[features[c] for c in self.feature_columns]],
                           columns=self.feature_columns)
        try:
            row = row.astype(float)
        except (TypeError, ValueError) as exc:
            raise InputError(f"All feature values must be numeric codes: {exc}") from exc

        if self.scaler is not None:
            values = self.scaler.transform(
                row if hasattr(self.scaler, "feature_names_in_") else row.values
            )
        else:
            values = row.values

        probs = np.asarray(self.model.predict_proba(values))[0]
        names = self._class_names()
        best = int(np.argmax(probs))
        ranked = sorted(zip(names, probs), key=lambda p: p[1], reverse=True)
        return {
            "predicted_class": names[best],
            "confidence": round(float(probs[best]), 4),
            "probabilities": {n: round(float(p), 4) for n, p in ranked},
        }
