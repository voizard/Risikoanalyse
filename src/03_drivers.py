"""Treiberanalyse: Was haengt mit dem Ausfall zusammen?

Drei Blickwinkel auf denselben Datensatz:

1. Logistische Regression, standardisiert -- Vorzeichen und Odds-Ratio je
   Merkmal. Standardisiert heisst: die Koeffizienten sind untereinander
   vergleichbar, weil alle Merkmale dieselbe Skala haben.
2. Random-Forest-Wichtigkeit -- welches Merkmal traegt die Trennung,
   unabhaengig von der Richtung.
3. Permutationswichtigkeit auf den Testdaten -- wie stark bricht die
   Trennschaerfe ein, wenn ein Merkmal zufaellig durchmischt wird. Das ist
   das ehrlichere Mass, weil es am Ergebnis gemessen wird.

Alle Groessen werden per Bootstrap mit Unsicherheit versehen: sonst steht
eine Zahl im Bericht, die von der zufaelligen Aufteilung abhaengt.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FEATURES, FIGURES, LABELS_DE, RESULTS, ensure_dirs,
                    load_labeled, write_german_csv)

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "figure.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE, RED = "#3b7dd8", "#c0392b"


def logreg_treiber(X, y, n_boot=400, seed=42):
    """Standardisierte Koeffizienten + Odds-Ratios mit Bootstrap-Intervall."""
    def fitte(Xs, ys):
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000,
                                                    class_weight="balanced"))])
        pipe.fit(Xs, ys)
        return pipe.named_steps["clf"].coef_[0]

    coef = fitte(X, y)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, len(FEATURES)))
    n = len(y)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            boot[b] = np.nan
            continue
        boot[b] = fitte(X[idx], y[idx])
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)

    out = {}
    for i, name in enumerate(FEATURES):
        out[name] = {
            "coef": round(float(coef[i]), 3),
            "odds_ratio": round(float(np.exp(coef[i])), 3),
            "ki95_coef": [round(float(lo[i]), 3), round(float(hi[i]), 3)],
            "signifikant": bool(lo[i] > 0 or hi[i] < 0),
        }
    return out


def rf_wichtigkeit(X, y, seed=42):
    rf = RandomForestClassifier(n_estimators=800, random_state=seed,
                                class_weight="balanced", n_jobs=-1)
    rf.fit(X, y)
    return {f: round(float(v), 4)
            for f, v in zip(FEATURES, rf.feature_importances_)}


def perm_wichtigkeit(X, y, seed=42, n_splits=5, n_repeats=3):
    """Permutationswichtigkeit, gemittelt ueber 5 Folds (ROC-AUC-Abfall).

    sklearn 1.9 hat den Parameter ``cv`` aus ``permutation_importance``
    entfernt. Die Kreuzvalidierung wird deshalb hier von Hand gebaut: je Fold
    ein Modell fitten, dann auf den zurueckgehaltenen Daten jedes Merkmal
    zufaellig durchmischen und messen, wie weit die ROC-AUC faellt.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    from sklearn.metrics import roc_auc_score
    summen = np.zeros(len(FEATURES))
    summen_q = np.zeros(len(FEATURES))
    n_laeufe = 0

    for tr, te in cv.split(X, y):
        rf = RandomForestClassifier(n_estimators=500, random_state=seed,
                                    class_weight="balanced", n_jobs=-1)
        rf.fit(X[tr], y[tr])
        basis = roc_auc_score(y[te], rf.predict_proba(X[te])[:, 1])
        for i in range(len(FEATURES)):
            abfaelle = []
            for rep in range(n_repeats):
                Xp = X[te].copy()
                rng = np.random.default_rng(seed + rep)
                Xp[:, i] = rng.permutation(Xp[:, i])
                abfaelle.append(basis - roc_auc_score(
                    y[te], rf.predict_proba(Xp)[:, 1]))
            summen[i] += np.mean(abfaelle)
            summen_q[i] += np.var(abfaelle)
        n_laeufe += 1

    mittel = summen / n_laeufe
    streuung = np.sqrt(summen_q / n_laeufe)
    return {f: {"mean": round(float(m), 4), "std": round(float(s), 4)}
            for f, m, s in zip(FEATURES, mittel, streuung)}


def fig_treiber(lr, perm):
    order = sorted(FEATURES, key=lambda f: -abs(lr[f]["coef"]))
    names = [LABELS_DE[f] for f in order]
    coef = [lr[f]["coef"] for f in order]
    lo = [lr[f]["ki95_coef"][0] for f in order]
    hi = [lr[f]["ki95_coef"][1] for f in order]
    err = [[max(0.0, c - l) for c, l in zip(coef, lo)],
           [max(0.0, h - c) for c, h in zip(coef, hi)]]
    colors = [RED if c > 0 else BLUE for c in coef]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                  gridspec_kw={"width_ratios": [1.15, 1]})
    yy = np.arange(len(names))
    ax.barh(yy, coef, 0.6, color=colors, edgecolor="#3d454d", linewidth=0.6,
            xerr=err, capsize=4, error_kw=dict(ecolor="#555", lw=1.2))
    ax.axvline(0, color="#333", lw=1.2)
    ax.set_yticks(yy)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    for yi, c, s in zip(yy, coef, [lr[f]["signifikant"] for f in order]):
        mark = f"{c:+.2f}".replace(".", ",")
        if not s:
            mark += "  (n.s.)"
        ax.text(c + (0.05 if c > 0 else -0.05), yi, mark, va="center",
                ha="left" if c > 0 else "right", fontsize=8.5,
                color="#222" if s else "#8b9099")
    ax.set_xlabel("Standardisierter Koeffizient (log. Regression)")
    ax.set_title("Richtung und Staerke des Zusammenhangs\n"
                 "Rot = erhoeht das Ausfallrisiko, blau = senkt es",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(min(lo) - 0.35, max(hi) + 0.5)

    order2 = sorted(FEATURES, key=lambda f: -perm[f]["mean"])
    names2 = [LABELS_DE[f] for f in order2]
    pm = [perm[f]["mean"] for f in order2]
    ps = [perm[f]["std"] for f in order2]
    yy2 = np.arange(len(names2))
    ax2.barh(yy2, pm, 0.6, color="#4a9d5f", edgecolor="#2f6b3d",
             linewidth=0.6, xerr=ps, capsize=4,
             error_kw=dict(ecolor="#555", lw=1.2))
    ax2.set_yticks(yy2)
    ax2.set_yticklabels(names2, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Abfall der ROC-AUC beim Durchmischen")
    ax2.set_title("Permutationswichtigkeit\n"
                  "Wie viel Trennschaerfe geht verloren, "
                  "wenn das Merkmal nutzlos wird?",
                  fontsize=10, fontweight="bold")
    ax2.grid(axis="x", alpha=0.25, lw=0.6)
    ax2.set_axisbelow(True)
    fig.tight_layout()
    p = FIGURES / "06_treiber.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ensure_dirs()
    X, y = load_labeled()

    print("=" * 74)
    print("Treiberanalyse -- was haengt mit dem Ausfall zusammen?")
    print("=" * 74)

    lr = logreg_treiber(X, y)
    rf = rf_wichtigkeit(X, y)
    perm = perm_wichtigkeit(X, y)

    print(f"\n{'Merkmal':30s} {'coef':>7s} {'OR':>7s} "
          f"{'KI95 (coef)':>20s}  signifikant")
    print("-" * 74)
    for f in sorted(FEATURES, key=lambda f: -abs(lr[f]["coef"])):
        v = lr[f]
        print(f"{LABELS_DE[f]:30s} {v['coef']:>+7.2f} {v['odds_ratio']:>7.2f} "
              f" [{v['ki95_coef'][0]:>+5.2f}; {v['ki95_coef'][1]:>+5.2f}]"
              f"  {'ja' if v['signifikant'] else 'nein'}")

    print(f"\nPermutationswichtigkeit (ROC-AUC-Abfall, Top 5):")
    for f in sorted(FEATURES, key=lambda f: -perm[f]["mean"])[:5]:
        print(f"  {LABELS_DE[f]:30s} {perm[f]['mean']:+.4f}"
              f" +- {perm[f]['std']:.4f}")

    bericht = {"logreg": lr, "rf_importance": rf, "permutation": perm}
    with open(RESULTS / "03_treiber.json", "w", encoding="utf-8") as fh:
        json.dump(bericht, fh, ensure_ascii=False, indent=2)

    kopf = ["merkmal", "coef", "odds_ratio", "ki_unten", "ki_oben",
            "signifikant", "rf_wichtigkeit", "perm_auc_abfall"]
    zeilen = [[LABELS_DE[f], lr[f]["coef"], lr[f]["odds_ratio"],
               lr[f]["ki95_coef"][0], lr[f]["ki95_coef"][1],
               "ja" if lr[f]["signifikant"] else "nein",
               rf[f], perm[f]["mean"]] for f in FEATURES]
    write_german_csv(RESULTS / "03_treiber.csv", kopf, zeilen)

    print("geschrieben:", fig_treiber(lr, perm))
    print("geschrieben:", RESULTS / "03_treiber.json")
    print("geschrieben:", RESULTS / "03_treiber.csv")


if __name__ == "__main__":
    main()
