"""Paste this into the END of your training notebook (adapt names) to save the
four files the API needs. Run it in the same environment you trained in."""
import joblib

# model            -> your best fitted classifier (e.g. the XGBoost model)
# scaler           -> the fitted StandardScaler (or None if you did not scale)
# label_encoder    -> the fitted LabelEncoder for the 'Type' column
# X.columns        -> the feature DataFrame columns, in training order
joblib.dump(model, "artifacts/migraine_model.pkl")
joblib.dump(scaler, "artifacts/migraine_scaler.pkl")
joblib.dump(label_encoder, "artifacts/migraine_label_encoder.pkl")
joblib.dump(list(X.columns), "artifacts/migraine_feature_columns.pkl")
