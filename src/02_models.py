"""Modellvergleich mit verschachtelter Kreuzvalidierung.

Verglichen werden Trivialklassifikator, logistische Regression, Random Forest,
Gradient Boosting und ein neuronales Netz -- jeweils ohne und mit
Ueberabtastung (SMOTE). Die Hyperparameterwahl laeuft in einem inneren
3-Fold-Schritt; die Bewertung in einem aeusseren 10-Fold-Schritt. Damit sieht
die Parameterwahl die Bewertungsdaten nie -- die Kennzahlen sind Messungen,
keine Obergrenzen.

SMOTE sitzt in der Pipeline und wirkt nur auf die Trainingsfolds; sonst
gelangten kuenstliche Punkte in die Bewertung und die Kennzahlen waeren zu
guenstig.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FIGURES, RESULTS, ensure_dirs, load_labeled,
                    write_german_csv)

warnings.filterwarnings("ignore")

SCORING = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}

# (Name, Modell, Parameterraster). Die Praefixe beziehen sich auf die
# Pipeline-Stufe "clf".
ZOO = {
    "Trivial (Mehrheitsklasse)": (
        DummyClassifier(strategy="most_frequent"), {}),
    "Logistische Regression": (
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        {"clf__C": [0.1, 1.0, 10.0]}),
    "Random Forest": (
        RandomForestClassifier(random_state=42, class_weight="balanced",
                               n_jobs=1),
        {"clf__n_estimators": [300], "clf__max_depth": [None, 6]}),
    "Gradient Boosting": (
        HistGradientBoostingClassifier(random_state=42),
        {"clf__learning_rate": [0.05, 0.1]}),
    "Neuronales Netz": (
        MLPClassifier(random_state=42, max_iter=800),
        {"clf__hidden_layer_sizes": [(16, 8), (32, 16)],
         "clf__alpha": [1e-4]}),
}

# Raster fuer den Rauchtest -- nur ein Punkt je Modell
ZOO_SMOKE = {k: (v[0], {p: [w[0]] for p, w in v[1].items()})
             for k, v in ZOO.items()}


def baue_pipeline(modell, mit_smote):
    """Pipeline mit Skalierung und optionalem SMOTE vor dem Klassifikator."""
    schritte = [("scaler", StandardScaler())]
    if mit_smote:
        schritte.append(("smote", SMOTE(random_state=42)))
    schritte.append(("clf", modell))
    return ImbPipeline(schritte) if mit_smote else Pipeline(schritte)


def evaluiere(X, y, zoo, n_splits, seed=42):
    """Verschachtelte CV; gibt eine Liste von Fold-Ergebnissen zurueck."""
    aeussere = StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=seed)
    innere = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    ergebnisse = []

    for name, (modell, raster) in zoo.items():
        for mit_smote in (False, True):
            variante = f"{name}{' + SMOTE' if mit_smote else ''}"
            t0 = time.time()
            for fold, (tr, te) in enumerate(aeussere.split(X, y), 1):
                pipe = baue_pipeline(modell, mit_smote)
                if raster:
                    suche = GridSearchCV(pipe, raster, cv=innere,
                                         scoring="f1", n_jobs=-1,
                                         refit=True)
                    suche.fit(X[tr], y[tr])
                    best = suche.best_estimator_
                    params = json.dumps(suche.best_params_, default=str)
                else:
                    pipe.fit(X[tr], y[tr])
                    best = pipe
                    params = "{}"
                pred = best.predict(X[te])
                prob = best.predict_proba(X[te])[:, 1]
                zeile = {"modell": variante, "fold": fold,
                         "best_params": params}
                from sklearn.metrics import (accuracy_score,
                                             average_precision_score,
                                             f1_score, precision_score,
                                             recall_score, roc_auc_score)
                zeile.update({
                    "precision": precision_score(y[te], pred,
                                                 zero_division=0),
                    "recall": recall_score(y[te], pred, zero_division=0),
                    "f1": f1_score(y[te], pred, zero_division=0),
                    "accuracy": accuracy_score(y[te], pred),
                    "roc_auc": roc_auc_score(y[te], prob),
                    "pr_auc": average_precision_score(y[te], prob),
                })
                ergebnisse.append(zeile)
            print(f"  {variante:38s} {time.time()-t0:6.1f} s")
    return ergebnisse


def zusammenfassung(ergebnisse):
    """Mittelwert und Streuung je Variante und Kennzahl."""
    from collections import defaultdict
    gruppen = defaultdict(list)
    for r in ergebnisse:
        gruppen[r["modell"]].append(r)
    out = {}
    for modell, zeilen in gruppen.items():
        out[modell] = {}
        for k in ["f1", "precision", "recall", "accuracy", "roc_auc",
                  "pr_auc"]:
            werte = [z[k] for z in zeilen]
            out[modell][k] = {"mean": round(float(np.mean(werte)), 4),
                              "std": round(float(np.std(werte, ddof=1)), 4),
                              "n": len(werte)}
    return out


def fig_vergleich(zus):
    """Balkenvergleich F1 je Modell, ohne und mit SMOTE, mit Streuung."""
    basen = ["Trivial (Mehrheitsklasse)", "Logistische Regression",
             "Random Forest", "Gradient Boosting", "Neuronales Netz"]
    kurznamen = ["Trivial", "Log. Regression", "Random Forest",
                 "Gradient Boosting", "Neuronales Netz"]
    ohne = [zus[b]["f1"]["mean"] for b in basen]
    ohne_s = [zus[b]["f1"]["std"] for b in basen]
    mit = [zus[b + " + SMOTE"]["f1"]["mean"] for b in basen]
    mit_s = [zus[b + " + SMOTE"]["f1"]["std"] for b in basen]

    xx = np.arange(len(basen))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(xx - w/2, ohne, w, yerr=ohne_s, capsize=4, color="#c9ccd1",
           edgecolor="#8b9099", label="ohne Ueberabtastung",
           error_kw=dict(ecolor="#5a6069", lw=1.3))
    ax.bar(xx + w/2, mit, w, yerr=mit_s, capsize=4, color="#3b7dd8",
           edgecolor="#25518f", label="mit SMOTE",
           error_kw=dict(ecolor="#1b3a66", lw=1.3))
    for x, v, e in zip(xx - w/2, ohne, ohne_s):
        ax.text(x, v + e + 0.012, f"{v:.3f}".replace(".", ","), ha="center",
                fontsize=7.5, color="#5a6069")
    for x, v, e in zip(xx + w/2, mit, mit_s):
        ax.text(x, v + e + 0.012, f"{v:.3f}".replace(".", ","), ha="center",
                fontsize=7.5, color="#1b3a66", fontweight="bold")
    ax.set_xticks(xx)
    ax.set_xticklabels(kurznamen, fontsize=9)
    ax.set_ylabel("F1-Wert (Ausfallerklasse)")
    ax.set_ylim(0, 0.92)
    ax.legend(fontsize=9, frameon=False)
    ax.set_title("Modellvergleich, verschachtelte Kreuzvalidierung\n"
                 "Fehlerbalken = Streuung ueber 10 aeussere Folds",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = FIGURES / "05_modellvergleich_f1.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Schnelltest: 2 Folds, je ein Parameterpunkt")
    args = ap.parse_args()
    ensure_dirs()

    X, y = load_labeled()
    zoo = ZOO_SMOKE if args.smoke else ZOO
    n_splits = 2 if args.smoke else 10

    print("=" * 74)
    print("Modellvergleich -- verschachtelte Kreuzvalidierung")
    print(f"  Beobachtungen: {len(y)}   Positive (Ausfall): "
          f"{100*y.mean():.2f} %   aeussere Folds: {n_splits}")
    print("=" * 74)

    t0 = time.time()
    ergebnisse = evaluiere(X, y, zoo, n_splits)
    zus = zusammenfassung(ergebnisse)
    print(f"\nGesamtlaufzeit: {time.time()-t0:.1f} s")

    print(f"\n{'Variante':38s} {'F1':>14s} {'ROC-AUC':>14s} {'PR-AUC':>14s}")
    print("-" * 82)
    for name, m in zus.items():
        f = lambda k: f"{m[k]['mean']:.4f}+-{m[k]['std']:.4f}"
        print(f"{name:38s} {f('f1'):>14s} {f('roc_auc'):>14s} {f('pr_auc'):>14s}")

    suffix = "_smoke" if args.smoke else ""
    kopf = ["modell", "fold", "precision", "recall", "f1", "accuracy",
            "roc_auc", "pr_auc", "best_params"]
    zeilen = [[r[k] for k in kopf] for r in ergebnisse]
    write_german_csv(RESULTS / f"02_modelle{suffix}.csv", kopf, zeilen)
    print(f"\ngeschrieben: {RESULTS}/02_modelle{suffix}.csv")

    with open(RESULTS / f"02_zusammenfassung{suffix}.json", "w",
              encoding="utf-8") as fh:
        json.dump(zus, fh, ensure_ascii=False, indent=2)
    print(f"geschrieben: {RESULTS}/02_zusammenfassung{suffix}.json")

    if not args.smoke:
        print("geschrieben:", fig_vergleich(zus))


if __name__ == "__main__":
    main()
