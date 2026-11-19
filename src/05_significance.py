"""Signifikanztest fuer den Modellvergleich.

Die Fold-Ergebnisse je Modell beruhen auf denselben aeusseren Aufteilungen und
sind damit gepaart. Ein gepaarter Wilcoxon-Vorzeichen-Rang-Test prueft, ob der
Unterschied zwischen zwei Varianten groesser ist, als die Streuung zwischen den
Folds erklaert -- und ist strenger als der optische Vergleich von Fehlerbalken.

Verglichen werden:
  * jedes Modell ohne gegen mit SMOTE
  * jedes Modell gegen den Trivialklassifikator
  * die besten Modelle untereinander
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, write_german_csv


def num(s: str) -> float:
    """Liest Zahlen mit Dezimalkomma (Projektformat) oder Punkt."""
    s = s.strip()
    if "," in s and "." not in s:
        return float(s.replace(",", "."))
    return float(s)


def lade():
    by = defaultdict(dict)
    with open(RESULTS / "02_modelle.csv", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            by[row["modell"]][int(row["fold"])] = num(row["f1"])
    return by


def teste(by, a, b, metrik="f1"):
    folds = sorted(set(by[a]) & set(by[b]))
    xa = [by[a][f] for f in folds]
    xb = [by[b][f] for f in folds]
    diff = [p - q for p, q in zip(xa, xb)]
    if all(abs(d) < 1e-12 for d in diff):
        return {"a": a, "b": b, "n": len(folds),
                "mittel_a": sum(xa) / len(xa), "mittel_b": sum(xb) / len(xb),
                "delta": 0.0, "p": 1.0}
    r = wilcoxon(xa, xb)
    return {"a": a, "b": b, "n": len(folds),
            "mittel_a": round(sum(xa) / len(xa), 4),
            "mittel_b": round(sum(xb) / len(xb), 4),
            "delta": round(sum(xa) / len(xa) - sum(xb) / len(xb), 4),
            "p": round(float(r.pvalue), 4)}


def main():
    by = lade()
    modelle = list(by)

    paare = []
    # ohne gegen mit SMOTE
    for m in modelle:
        if m + " + SMOTE" in by:
            paare.append((m + " + SMOTE", m))
    # gegen Trivial
    triv = "Trivial (Mehrheitsklasse)"
    for m in modelle:
        if m != triv and "+ SMOTE" not in m:
            paare.append((m, triv))
    # beste Modelle untereinander
    nicht_triv = [m for m in modelle if m != triv and "+ SMOTE" not in m]
    for i in range(len(nicht_triv)):
        for j in range(i + 1, len(nicht_triv)):
            paare.append((nicht_triv[i], nicht_triv[j]))

    print("=" * 84)
    print("Signifikanztest (Wilcoxon-Vorzeichen-Rang, gepaart ueber die Folds)")
    print("=" * 84)
    print(f"\n{'Variante A':32s} {'Variante B':32s} {'dF1':>8s} {'p':>8s}  Befund")
    print("-" * 84)
    ergebnis = []
    for a, b in paare:
        r = teste(by, a, b)
        befund = "signifikant" if r["p"] < 0.05 else "nicht unterscheidbar"
        ergebnis.append({**r, "befund": befund})
        print(f"{a[:32]:32s} {b[:32]:32s} {r['delta']:>+8.4f} "
              f"{r['p']:>8.4f}  {befund}")

    with open(RESULTS / "05_signifikanz.json", "w", encoding="utf-8") as fh:
        json.dump(ergebnis, fh, ensure_ascii=False, indent=2)

    kopf = ["variante_a", "variante_b", "f1_a", "f1_b", "delta_f1", "p_wert",
            "befund"]
    zeilen = [[r["a"], r["b"], r["mittel_a"], r["mittel_b"], r["delta"],
               r["p"], r["befund"]] for r in ergebnis]
    write_german_csv(RESULTS / "05_signifikanz.csv", kopf, zeilen)
    print(f"\ngeschrieben: {RESULTS}/05_signifikanz.json")
    print(f"geschrieben: {RESULTS}/05_signifikanz.csv")


if __name__ == "__main__":
    main()
