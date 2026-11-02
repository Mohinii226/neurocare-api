"""Builds small dummy artifacts so tests and CI run without your real model."""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

FEATURES = [
    "Age", "Duration", "Frequency", "Location", "Character", "Intensity", "Nausea", "Vomit",
    "Phonophobia", "Photophobia", "Visual", "Sensory", "Dysphasia", "Dysarthria", "Vertigo",
    "Tinnitus", "Hypoacusis", "Diplopia", "Defect", "Ataxia", "Conscience", "Paresthesia", "DPF",
]
CLASSES = [
    "Migraine without aura", "Typical aura with migraine", "Typical aura without migraine",
    "Basilar-type aura", "Familial hemiplegic migraine", "Sporadic hemiplegic migraine", "Other",
]


def build_dummy_artifacts(directory):
    rng = np.random.default_rng(0)
    X = rng.integers(0, 4, size=(210, len(FEATURES))).astype(float)
    y_text = np.array([CLASSES[i % len(CLASSES)] for i in range(len(X))])
    encoder = LabelEncoder().fit(y_text)
    scaler = StandardScaler().fit(X)
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(
        scaler.transform(X), encoder.transform(y_text)
    )
    joblib.dump(model, directory / "migraine_model.pkl")
    joblib.dump(scaler, directory / "migraine_scaler.pkl")
    joblib.dump(encoder, directory / "migraine_label_encoder.pkl")
    joblib.dump(FEATURES, directory / "migraine_feature_columns.pkl")


def sample_features():
    return {name: 1.0 for name in FEATURES}
