#!/usr/bin/env python3
"""Beschafft den bankloans-Datensatz und prueft seine Form.

Die Datei gehoert nicht ins Repository (fremde Daten). Dieses Skript holt sie
und bricht ab, wenn Zeilen-, Spaltenzahl oder Ausfallquote abweichen -- eine
stillschweigend geaenderte Quelle wuerde sonst alle Ergebnisse verfaelschen.
"""
from __future__ import annotations

import csv
import io
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZIEL = HERE / "bankloans.csv"

# Kanonische Quelle (Erstveroeffentlichung)
QUELLE = ("https://raw.githubusercontent.com/Niranjankumar-c/"
          "CreditRiskAnalytics/master/Data/bankloans.csv")

ERWARTET = {
    "zeilen_gesamt": 1150,
    "zeilen_labeled": 700,
    "zeilen_unlabeled": 450,
    "spalten": ["age", "ed", "employ", "address", "income", "debtinc",
                "creddebt", "othdebt", "default"],
    "ausfallquote_pct": 26.1,
}


def pruefe(text: str) -> None:
    """Prueft die geladene Datei gegen die erwartete Form."""
    zeilen = list(csv.reader(io.StringIO(text)))
    kopf, daten = zeilen[0], zeilen[1:]

    fehler = []
    if kopf != ERWARTET["spalten"]:
        fehler.append(f"Spalten: {kopf}")
    if len(daten) != ERWARTET["zeilen_gesamt"]:
        fehler.append(f"Zeilen: {len(daten)} "
                      f"(erwartet {ERWARTET['zeilen_gesamt']})")
    labeled = [r for r in daten if r[-1].strip() != ""]
    if len(labeled) != ERWARTET["zeilen_labeled"]:
        fehler.append(f"gelabelte Zeilen: {len(labeled)}")
    quote = 100 * sum(int(r[-1]) for r in labeled) / len(labeled)
    if abs(quote - ERWARTET["ausfallquote_pct"]) > 0.2:
        fehler.append(f"Ausfallquote: {quote:.1f} %")

    if fehler:
        print("ABBRUCH -- die Quelle hat sich geaendert:", file=sys.stderr)
        for f in fehler:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)

    print(f"Form geprueft: {len(daten)} Zeilen, {len(kopf)} Spalten, "
          f"{len(labeled)} mit Label, Ausfallquote {quote:.1f} %")


def main() -> None:
    if ZIEL.exists():
        print(f"schon vorhanden: {ZIEL}")
        pruefe(ZIEL.read_text(encoding="utf-8-sig"))
        return
    print(f"lade: {QUELLE}")
    with urllib.request.urlopen(QUELLE, timeout=60) as r:
        text = r.read().decode("utf-8-sig")
    pruefe(text)
    ZIEL.write_text(text, encoding="utf-8")
    print(f"geschrieben: {ZIEL}")


if __name__ == "__main__":
    main()
