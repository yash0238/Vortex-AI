"""Classification metrics for regime predictions."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_classification(y_true, y_pred_proba, threshold: float = 0.5) -> dict:
	y_true = np.asarray(y_true, dtype=int).ravel()
	probabilities = np.asarray(y_pred_proba, dtype=float).ravel()
	if y_true.shape != probabilities.shape:
		raise ValueError("y_true and y_pred_proba must have the same length")
	if np.any((probabilities < 0) | (probabilities > 1)):
		probabilities = 1.0 / (1.0 + np.exp(-np.clip(probabilities, -60, 60)))
	predictions = (probabilities >= threshold).astype(int)
	return {
		"accuracy": accuracy_score(y_true, predictions),
		"precision": precision_score(y_true, predictions, zero_division=0),
		"recall": recall_score(y_true, predictions, zero_division=0),
		"f1": f1_score(y_true, predictions, zero_division=0),
		"roc_auc": roc_auc_score(y_true, probabilities) if np.unique(y_true).size > 1 else np.nan,
		"confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
		"classification_report": classification_report(y_true, predictions, zero_division=0),
	}


def compare_models(y_true, model_prob_dict: dict[str, np.ndarray], threshold: float = 0.5):
	import pandas as pd
	rows = []
	for name, probabilities in model_prob_dict.items():
		metrics = evaluate_classification(y_true, probabilities, threshold)
		rows.append({"Model": name, "Accuracy": metrics["accuracy"], "F1": metrics["f1"], "ROC-AUC": metrics["roc_auc"]})
	return pd.DataFrame(rows)
