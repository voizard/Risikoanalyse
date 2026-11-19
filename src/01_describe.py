"""Deskriptive Analyse des bankloans-Datensatzes.

Erzeugt Kennzahlen (JSON) und Abbildungen (PNG) zur Verteilung der Merkmale,
zur Ausfallquote je Merkmal und zur Korrelationsstruktur. Grundlage sind
ausschliesslich die 700 gelabelten Zeilen.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FEATURES, FIGURES, LABELS_DE, RESULTS, ensure_dirs,
                    load_labeled)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE, RED, GREY = "#3b7dd8", "#c0392b", "#9aa4ad"


def stats(X, y):
    """Kennzahlen je Merkmal, getrennt nach Ausfall und gesamt."""
    out = {}
    for i, name in enumerate(FEATURES):
        col = X[:, i]
        out[name] = {
            "gesamt": {"mean": round(float(col.mean()), 3),
                       "std": round(float(col.std()), 3),
                       "min": round(float(col.min()), 3),
                       "max": round(float(col.max()), 3)},
            "ohne_ausfall": {"mean": round(float(col[y == 0].mean()), 3)},
            "mit_ausfall": {"mean": round(float(col[y == 1].mean()), 3)},
        }
    return out


def univariate_auc(X, y):
    """Trennschaerfe jedes Merkmals allein (Mann-Whitney-Statistik).

    Die AUC eines einzelnen Merkmals ist die Wahrscheinlichkeit, dass ein
    zufaelliger Ausfaller einen hoeheren Wert hat als ein zufaelliger
    Nicht-Ausfaller. 0,5 = kein Zusammenhang.
    """
    out = {}
    for i, name in enumerate(FEATURES):
        col = X[:, i]
        pos, neg = col[y == 1], col[y == 0]
        # Rangsumme, gleiche Werte gemittelt
        alle = np.concatenate([pos, neg])
        order = alle.argsort()
        raenge = np.empty(len(alle), dtype=float)
        raenge[order] = np.arange(1, len(alle) + 1)
        # Bindungen mitteln
        _, inv, counts = np.unique(alle, return_inverse=True,
                                   return_counts=True)
        if (counts > 1).any():
            summen = np.zeros(len(counts))
            np.add.at(summen, inv, raenge)
            mittel = summen / counts
            raenge = mittel[inv]
        r_pos = raenge[:len(pos)].sum()
        u = r_pos - len(pos) * (len(pos) + 1) / 2
        auc = u / (len(pos) * len(neg))
        out[name] = round(float(auc), 4)
    return out


def ausfallquote_nach_ed(X, y):
    """Ausfallquote je Bildungsstufe mit Wilson-Konfidenzintervall."""
    def wilson(k, n, z=1.96):
        p = k / n
        d = 1 + z * z / n
        c = p + z * z / (2 * n)
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return (max(0.0, 100 * (c - h) / d), min(100.0, 100 * (c + h) / d))

    out = {}
    for stufe in sorted(np.unique(X[:, 1]).astype(int)):
        maske = X[:, 1] == stufe
        n = int(maske.sum())
        k = int(y[maske].sum())
        lo, hi = wilson(k, n)
        out[int(stufe)] = {"n": n, "ausfaelle": k,
                           "quote_pct": round(100 * k / n, 1),
                           "ki95": [round(lo, 1), round(hi, 1)]}
    return out


def fig_verteilungen(X, y):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for ax, i, name in zip(axes.ravel(), range(len(FEATURES)), FEATURES):
        col = X[:, i]
        ax.hist(col[y == 0], bins=24, alpha=0.65, label="ohne Ausfall",
                color=BLUE, edgecolor="white", linewidth=0.3)
        ax.hist(col[y == 1], bins=24, alpha=0.65, label="mit Ausfall",
                color=RED, edgecolor="white", linewidth=0.3)
        ax.set_title(LABELS_DE[name], fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("Verteilung der Merkmale, getrennt nach Ausfall "
                 "(700 gelabelte Kreditnehmer)", fontsize=11,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = FIGURES / "01_verteilungen.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_trennschaerfe(uni):
    order = sorted(uni.items(), key=lambda kv: abs(kv[1] - 0.5), reverse=True)
    names = [LABELS_DE[k] for k, _ in order]
    vals = [v for _, v in order]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    colors = [RED if v > 0.5 else BLUE for v in vals]
    yy = np.arange(len(names))
    ax.barh(yy, [v - 0.5 for v in vals], 0.6, left=0.5, color=colors,
            edgecolor="#3d454d", linewidth=0.6)
    ax.axvline(0.5, color="#333", lw=1.2)
    for yi, v in zip(yy, vals):
        # Werte rechts vom Balkenende; links vom Zentrum liegt der Balken,
        # dort weisse Schrift, sonst schreibt sie in die Achsenbeschriftung.
        if v >= 0.5:
            ax.text(v + 0.008, yi, f"{v:.3f}".replace(".", ","),
                    va="center", ha="left", fontsize=8.5)
        else:
            ax.text(v + 0.010, yi, f"{v:.3f}".replace(".", ","),
                    va="center", ha="left", fontsize=8.5, color="white")
    ax.set_yticks(yy)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("AUC des Merkmals allein (0,5 = kein Zusammenhang)")
    ax.set_title("Welches Merkmal trennt Ausfaller von Nicht-Ausfallern?\n"
                 "Rot = hoeherer Wert geht mit hoeherem Risiko einher",
                 fontsize=10.5, fontweight="bold")
    ax.set_xlim(0.38, 0.78)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = FIGURES / "02_trennschaerfe_einzelmerkmale.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_korrelation(X):
    C = np.corrcoef(X, rowvar=False)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_yticks(range(len(FEATURES)))
    ax.set_xticklabels([LABELS_DE[f] for f in FEATURES], rotation=40,
                       ha="right", fontsize=8)
    ax.set_yticklabels([LABELS_DE[f] for f in FEATURES], fontsize=8)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            v = C[i, j]
            ax.text(j, i, f"{v:.2f}".replace(".", ","), ha="center",
                    va="center", fontsize=7,
                    color="white" if abs(v) > 0.55 else "#222")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Korrelation")
    ax.set_title("Korrelation der Merkmale untereinander",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    p = FIGURES / "03_korrelation.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_ed(ed_tab, gesamt_quote):
    stufen = sorted(ed_tab)
    n = [ed_tab[s]["n"] for s in stufen]
    q = [ed_tab[s]["quote_pct"] for s in stufen]
    lo = [ed_tab[s]["ki95"][0] for s in stufen]
    hi = [ed_tab[s]["ki95"][1] for s in stufen]
    # Nach dem Runden kann eine Grenze minimal ueber dem Wert liegen -> auf 0 klemmen
    err = [[max(0.0, a - b) for a, b in zip(q, lo)],
           [max(0.0, c - b) for b, c in zip(hi, q)]]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    xx = np.arange(len(stufen))
    ax.bar(xx, q, 0.6, color=BLUE, edgecolor="#25518f",
           yerr=err, capsize=6, error_kw=dict(ecolor=RED, lw=1.6))
    ax.axhline(gesamt_quote, color="#333", ls="--", lw=1.2,
               label=f"Gesamtquote {gesamt_quote:.1f} %".replace(".", ","))
    for x, v, nn in zip(xx, q, n):
        ax.text(x, v + 3.2, f"{v:.1f} %".replace(".", ","), ha="center",
                fontsize=8.5, fontweight="bold")
        ax.text(x, 1.5, f"n={nn}", ha="center", fontsize=8, color="white")
    ax.set_xticks(xx)
    ax.set_xticklabels([f"Stufe {s}" for s in stufen])
    ax.set_ylabel("Ausfallquote (%)")
    ax.set_ylim(0, max(hi) + 8)
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("Ausfallquote nach Bildungsstufe\n"
                 "Die Konfidenzintervalle ueberlappen breit -- "
                 "kein belastbarer Effekt", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = FIGURES / "04_ausfallquote_bildung.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ensure_dirs()
    X, y = load_labeled()
    quote = 100 * y.mean()

    bericht = {
        "n_labeled": int(len(y)),
        "n_ausfaelle": int(y.sum()),
        "n_ohne": int((y == 0).sum()),
        "ausfallquote_pct": round(float(quote), 2),
        "merkmale": stats(X, y),
        "auc_einzelmerkmale": univariate_auc(X, y),
        "ausfallquote_nach_bildung": ausfallquote_nach_ed(X, y),
        "korrelation_ed_debtinc": round(float(np.corrcoef(X[:, 1], X[:, 5])[0, 1]), 4),
    }

    print("=" * 72)
    print("Deskriptive Analyse -- bankloans (700 gelabelte Zeilen)")
    print("=" * 72)
    print(f"\nAusfaelle: {bericht['n_ausfaelle']} von {bericht['n_labeled']}"
          f"  =  {bericht['ausfallquote_pct']} %")
    print(f"\n{'Merkmal':30s} {'ohne Ausfall':>13s} {'mit Ausfall':>13s}"
          f" {'AUC':>7s}")
    print("-" * 72)
    for f in FEATURES:
        m = bericht["merkmale"][f]
        print(f"{LABELS_DE[f]:30s} {m['ohne_ausfall']['mean']:>13.2f}"
              f" {m['mit_ausfall']['mean']:>13.2f}"
              f" {bericht['auc_einzelmerkmale'][f]:>7.3f}")

    print(f"\nAusfallquote nach Bildungsstufe:")
    for s, v in bericht["ausfallquote_nach_bildung"].items():
        print(f"  Stufe {s}: n={v['n']:4d}  {v['quote_pct']:5.1f} %"
              f"  KI95 [{v['ki95'][0]:5.1f}; {v['ki95'][1]:5.1f}]")
    print(f"\nKorrelation ed <-> debtinc: "
          f"{bericht['korrelation_ed_debtinc']}")

    for p in [fig_verteilungen(X, y), fig_trennschaerfe(
            bericht["auc_einzelmerkmale"]), fig_korrelation(X),
            fig_ed(bericht["ausfallquote_nach_bildung"], quote)]:
        print("geschrieben:", p)

    with open(RESULTS / "01_deskriptiv.json", "w", encoding="utf-8") as fh:
        json.dump(bericht, fh, ensure_ascii=False, indent=2)
    print("geschrieben:", RESULTS / "01_deskriptiv.json")


if __name__ == "__main__":
    main()
