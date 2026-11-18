"""Traditional ML Baseline: Char 1-3 gram TF-IDF + Calibrated LinearSVC."""

from __future__ import annotations

import time
from typing import Dict, List, Tuple
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from mvcsn.training.metrics import compute_classification_metrics


def run_tfidf_svm(
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    seed: int = 42,
) -> Tuple[Dict[str, float], List[float], float, object, object]:
    t0 = time.time()
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2, max_features=80000, sublinear_tf=True)
    X_tr = vec.fit_transform(train_texts)
    X_te = vec.transform(test_texts)
    clf = CalibratedClassifierCV(LinearSVC(C=1.0, random_state=seed), cv=3)
    clf.fit(X_tr, train_labels)
    train_sec = time.time() - t0

    t1 = time.time()
    probs = clf.predict_proba(X_te)[:, 1].tolist()
    infer_ms = (time.time() - t1) * 1000.0 / max(1, len(test_texts))
    metrics = compute_classification_metrics(test_labels, probs)
    metrics["infer_ms_per_sample"] = infer_ms
    metrics["train_seconds"] = train_sec
    return metrics, probs, infer_ms, vec, clf
