"""Gemeinsame Hilfsfunktionen: Daten laden, Kennzahlen, deutscher Export.

Der Datensatz ``bankloans`` enthaelt 1150 Zeilen, von denen nur 700 ein
Label tragen. Die hinteren 450 Zeilen sind ein Block ohne Zielvariable und
fuer ueberwachtes Lernen nicht verwendbar (siehe ``data/README.md``).
Alle Skripte laden daher ueber :func:`load_labeled`.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "bankloans.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

FEATURES = ["age", "ed", "employ", "address",
            "income", "debtinc", "creddebt", "othdebt"]
TARGET = "default"

# Lesbare Beschriftungen fuer Abbildungen und Bericht
LABELS_DE = {
    "age": "Alter",
    "ed": "Bildungsstufe",
    "employ": "Beschaeftigungsjahre",
    "address": "Wohnjahre an der Adresse",
    "income": "Einkommen (Tsd.)",
    "debtinc": "Schulden-Einkommens-Quote",
    "creddebt": "Kreditkartenschuld",
    "othdebt": "Sonstige Schulden",
}


def load_raw(path: Path = DATA_RAW):
    """Liest die Rohdatei und gibt (Kopfzeile, Datenzeilen) zurueck."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    header, data = rows[0], rows[1:]
    expected = FEATURES + [TARGET]
    if header != expected:
        raise SystemExit(f"Unerwarteter Kopf in {path}: {header}")
    return header, data


def is_labeled(row) -> bool:
    """Wahr, wenn die Zielvariable gesetzt ist."""
    return row[-1].strip() != ""


def load_labeled(path: Path = DATA_RAW):
    """Gibt (X, y) der gelabelten Zeilen zurueck -- die einzige Lernbasis."""
    _, data = load_raw(path)
    labeled = [r for r in data if is_labeled(r)]
    X = np.array([[float(v) for v in r[:-1]] for r in labeled], dtype=float)
    y = np.array([int(r[-1]) for r in labeled], dtype=int)
    return X, y


def load_unlabeled(path: Path = DATA_RAW, dedupe: bool = True):
    """Gibt X der ungelabelten Zeilen zurueck (nur zum Scoring, nie zum Lernen).

    Die gelieferte Datei enthaelt den ungelabelten Block verdreifacht
    (150 Muster, jedes dreimal). Mit ``dedupe=True`` wird daraus die
    kanonische Menge von 150 Zeilen -- sonst zaehlt jedes Muster dreifach
    in jede Auswertung hinein.
    """
    _, data = load_raw(path)
    rows = [r for r in data if not is_labeled(r)]
    if dedupe:
        seen, unique = set(), []
        for r in rows:
            key = tuple(r[:-1])
            if key not in seen:
                seen.add(key)
                unique.append(r)
        rows = unique
    return np.array([[float(v) for v in r[:-1]] for r in rows], dtype=float)


def write_german_csv(path: Path, header, rows) -> None:
    """Schreibt eine CSV im deutschen Power-BI-Format.

    Dezimalkomma, Semikolon, UTF-8 mit BOM, kein Tausenderpunkt -- so laesst
    sie sich in Power BI mit deutscher Locale ohne Zusatzschritt importieren.
    """
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}".replace(".", ",")
        return str(v)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        w.writerow(header)
        for row in rows:
            w.writerow([fmt(v) for v in row])


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
