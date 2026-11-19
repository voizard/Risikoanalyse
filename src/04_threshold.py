"""Schwellenwert und Kostenmodell -- der eigentliche Hebel.

Die Modelle ordnen die Kunden gut (ROC-AUC um 0,85), aber die feste Schwelle
0,5 nutzt das nicht aus: Bei einem Ausfallanteil von 26 % ist 0,5 eine
willkuerliche Grenze. Hier wird die Schwelle aus dem erwarteten Nutzen
abgeleitet.

Kostenmodell
------------
Zwei Fehler sind moeglich:
  * verpasster Ausfall (False Negative) -- der Kunde faellt aus, ohne dass
    gegengesteuert wurde; teuer.
  * Fehlalarm (False Positive) -- ein guter Kunde wird geprueft; kostet
    Zeit und vielleicht Geschaeftsbeziehung.

Mit ``kosten_fn`` und ``kosten_fp`` laesst sich das als Verhaeltnis
ausdruecken. Die optimale Schwelle maximiert den erwarteten Nutzen
(Ersparnis minus Kosten). Gerechnet wird fuer mehrere Verhaeltnisse, weil
die echten Zahlen beim Nutzer liegen.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FIGURES, RESULTS, ensure_dirs, load_labeled,
                    write_german_csv)

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "figure.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE, RED, GREEN = "#3b7dd8", "#c0392b", "#4a9d5f"

# Verhaeltnis "Kosten eines verpassten Ausfalls" : "Kosten eines Fehlalarms"
VERHAELTNISSE = [1, 5, 10]


def out_of_fold_probs(X, y, seed=42):
    """Ausfallwahrscheinlichkeit je Kunde, out-of-fold geschaetzt.

    Jede Wahrscheinlichkeit stammt aus einem Modell, das diesen Kunden nicht
    gesehen hat -- sonst waeren die Schwellenwerte auf den Trainingsdaten
    geschoenigt.
    """
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    prob = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000,
                                                    class_weight="balanced"))])
        pipe.fit(X[tr], y[tr])
        prob[te] = pipe.predict_proba(X[te])[:, 1]
    return prob


def kennzahlen_bei(prob, y, schwelle):
    pred = (prob >= schwelle).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"schwelle": round(float(schwelle), 3), "tp": tp, "fp": fp,
            "fn": fn, "tn": tn, "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4)}


def kosten(prob, y, schwelle, kosten_fn, kosten_fp=1.0):
    """Erwartete Gesamtkosten bei gegebener Schwelle (in Einheiten von fp)."""
    k = kennzahlen_bei(prob, y, schwelle)
    return k["fn"] * kosten_fn + k["fp"] * kosten_fp


def optimale_schwelle(prob, y, kosten_fn, kosten_fp=1.0):
    schwellen = np.unique(np.round(prob, 4))
    kosten_je = [kosten(prob, y, s, kosten_fn, kosten_fp) for s in schwellen]
    i = int(np.argmin(kosten_je))
    return float(schwellen[i]), float(kosten_je[i])


def fig_schwellen(prob, y, tabelle):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Links: Precision/Recall/F1 ueber der Schwelle
    schwellen = np.linspace(0.02, 0.98, 97)
    ks = [kennzahlen_bei(prob, y, s) for s in schwellen]
    ax.plot(schwellen, [k["precision"] for k in ks], color=BLUE, lw=1.8,
            label="Precision")
    ax.plot(schwellen, [k["recall"] for k in ks], color=RED, lw=1.8,
            label="Recall")
    ax.plot(schwellen, [k["f1"] for k in ks], color=GREEN, lw=1.8,
            label="F1")
    ax.axvline(0.5, color="#333", ls=":", lw=1.4, label="Standard 0,5")
    best_f1 = max(ks, key=lambda k: k["f1"])
    ax.axvline(best_f1["schwelle"], color=GREEN, ls="--", lw=1.4,
               label=f"bestes F1 ({best_f1['schwelle']:.2f})".replace(".", ","))
    for v in tabelle:
        if v["verhaeltnis"] != 5:
            continue
        ax.axvline(v["schwelle"], color="#8e44ad", ls="-.", lw=1.6,
                   label=f"Kostenoptimum 5:1 ({v['schwelle']:.2f})".replace(".", ","))
    ax.set_xlabel("Schwellenwert")
    ax.set_ylabel("Kennzahl")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_title("Kennzahlen ueber der Schwelle\n"
                 "F1 ist bei 26 % Ausfaellen kein guter Kompass",
                 fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)

    # Rechts: erwartete Kosten ueber der Schwelle, je Verhaeltnis
    farben = {1: "#7f8c8d", 5: "#8e44ad", 10: "#c0392b"}
    for v in tabelle:
        kosten_je = [kosten(prob, y, s, v["verhaeltnis"]) for s in schwellen]
        # normieren, damit die Kurven vergleichbar sind
        norm = np.array(kosten_je) / max(kosten_je)
        ax2.plot(schwellen, norm, lw=1.8, color=farben[v["verhaeltnis"]],
                 label=f"{v['verhaeltnis']}:1  "
                       f"(Optimum {v['schwelle']:.2f})".replace(".", ","))
        ax2.axvline(v["schwelle"], color=farben[v["verhaeltnis"]], ls="--",
                    lw=1.1, alpha=0.7)
    ax2.axvline(0.5, color="#333", ls=":", lw=1.4)
    ax2.text(0.505, 0.97, "Standard 0,5", fontsize=8, rotation=90,
             va="top", color="#333")
    ax2.set_xlabel("Schwellenwert")
    ax2.set_ylabel("Erwartete Kosten (normiert auf Maximum)")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8, frameon=False)
    ax2.set_title("Kosten ueber der Schwelle, je Kostenverhaeltnis\n"
                  "Je teurer ein verpasster Ausfall, desto niedriger die "
                  "optimale Schwelle",
                  fontsize=10, fontweight="bold")
    ax2.grid(alpha=0.25, lw=0.6)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    p = FIGURES / "07_schwellenwerte.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_konfusionsmatrix(prob, y, schwelle, titel, pfad):
    k = kennzahlen_bei(prob, y, schwelle)
    m = np.array([[k["tn"], k["fp"]], [k["fn"], k["tp"]]])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(m, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(m[i, j]), ha="center", va="center",
                    fontsize=15, fontweight="bold",
                    color="white" if m[i, j] > m.max() * 0.55 else "#222")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["kein Ausfall", "Ausfall"])
    ax.set_yticklabels(["kein Ausfall", "Ausfall"])
    ax.set_xlabel("Vorhersage")
    ax.set_ylabel("Tatsaechlich")
    ax.set_title(f"{titel}\nSchwelle {schwelle:.2f}".replace(".", ","),
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(pfad, bbox_inches="tight")
    plt.close(fig)
    return pfad


def main():
    ensure_dirs()
    X, y = load_labeled()
    prob = out_of_fold_probs(X, y)

    print("=" * 74)
    print("Schwellenwert und Kostenmodell")
    print("=" * 74)

    tabelle = []
    for v in VERHAELTNISSE:
        s, k = optimale_schwelle(prob, y, kosten_fn=v)
        kenn = kennzahlen_bei(prob, y, s)
        tabelle.append({"verhaeltnis": v, "schwelle": round(s, 3),
                        "kosten": round(k, 1), **{q: kenn[q] for q in
                        ["tp", "fp", "fn", "tn", "precision", "recall", "f1"]}})

    kenn05 = kennzahlen_bei(prob, y, 0.5)

    print(f"\nStandard-Schwelle 0,5:")
    print(f"  Precision {kenn05['precision']:.3f}  Recall {kenn05['recall']:.3f}"
          f"  F1 {kenn05['f1']:.3f}")
    print(f"  TP {kenn05['tp']}  FP {kenn05['fp']}  "
          f"FN {kenn05['fn']}  TN {kenn05['tn']}")

    print(f"\n{'Verh.':>6s} {'Schwelle':>9s} {'Precision':>10s} "
          f"{'Recall':>8s} {'F1':>7s} {'TP':>5s} {'FP':>5s} {'FN':>5s}")
    print("-" * 74)
    for t in tabelle:
        print(f"{t['verhaeltnis']:>5d}:1 {t['schwelle']:>9.3f} "
              f"{t['precision']:>10.3f} {t['recall']:>8.3f} {t['f1']:>7.3f} "
              f"{t['tp']:>5d} {t['fp']:>5d} {t['fn']:>5d}")

    bericht = {"schwelle_standard": 0.5, "kennzahlen_standard": kenn05,
               "optima": tabelle}
    with open(RESULTS / "04_schwellen.json", "w", encoding="utf-8") as fh:
        json.dump(bericht, fh, ensure_ascii=False, indent=2)

    kopf = ["verhaeltnis_fn_zu_fp", "schwelle", "precision", "recall", "f1",
            "tp", "fp", "fn", "tn"]
    zeilen = [[t["verhaeltnis"], t["schwelle"], t["precision"], t["recall"],
               t["f1"], t["tp"], t["fp"], t["fn"], t["tn"]] for t in tabelle]
    zeilen.insert(0, [0, 0.5, kenn05["precision"], kenn05["recall"],
                      kenn05["f1"], kenn05["tp"], kenn05["fp"], kenn05["fn"],
                      kenn05["tn"]])
    write_german_csv(RESULTS / "04_schwellen.csv", kopf, zeilen)

    print("\ngeschrieben:", fig_schwellen(prob, y, tabelle))
    s5 = next(t["schwelle"] for t in tabelle if t["verhaeltnis"] == 5)
    print("geschrieben:", fig_konfusionsmatrix(
        prob, y, 0.5, "Verwirrungsmatrix bei Standard 0,5",
        FIGURES / "08_konfusion_050.png"))
    print("geschrieben:", fig_konfusionsmatrix(
        prob, y, s5, "Verwirrungsmatrix im Kostenoptimum",
        FIGURES / "09_konfusion_kosten.png"))
    print("geschrieben:", RESULTS / "04_schwellen.json")
    print("geschrieben:", RESULTS / "04_schwellen.csv")


if __name__ == "__main__":
    main()
