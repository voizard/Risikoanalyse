# data/ — die Rohdaten liegen hier, nicht im Repository

Die Analyse liest `bankloans.csv`. Diese Datei gehört **nicht** ins Repository
(fremde Daten, siehe Lizenz unten), sondern wird mit `fetch_data.py` beschafft.

## Kanonische Quelle

- **Datensatz:** `bankloans` (Kreditausfall, 8 Merkmale + Zielvariable)
- **Erstveröffentlichung:** `https://github.com/Niranjankumar-c/CreditRiskAnalytics/blob/master/Data/bankloans.csv`

## Erwartete Form

| Eigenschaft | Wert |
|---|---|
| Zeilen gesamt | 1150 |
| Zeilen **mit** Label | 700 |
| Zeilen ohne Label | 450 |
| Spalten | 9 (`age`, `ed`, `employ`, `address`, `income`, `debtinc`, `creddebt`, `othdebt`, `default`) |
| Ausfallquote (der gelabelten Zeilen) | 26,1 % |
| Trennzeichen | Komma |
| Dezimaltrennzeichen | Punkt |

## Wichtiger Hinweis zur Struktur

Die gelieferte Datei enthält den **ungelabelten Block verdreifacht**: 450
Zeilen, die aus 150 verschiedenen Merkmalsmustern bestehen, jedes genau
dreimal. Die ersten 700 Zeilen tragen ein Label, die letzten 450 nicht.

Für überwachtes Lernen sind nur die **700 gelabelten Zeilen** nutzbar. Die
450 ungelabelten enthalten keine Zielvariable und keine neuen
Merkmalskombinationen — sie sind ein Artefakt der Dateizusammenstellung.
`src/common.py` entfernt die Verdreifachung beim Laden (`dedupe=True`).

**Entscheidung:** Die Analyse beschränkt sich bewusst auf die 700 gelabelten
Zeilen. Die 450 ungelabelten werden nicht verwendet — sie bringen weder eine
Zielvariable noch neue Merkmalsmuster mit.

## Lizenz

Der Datensatz stammt aus dem oben genannten Repository. Die Rechte liegen beim
Herausgeber; er wird hier **nicht weitergegeben**, sondern nur zur Laufzeit
beschafft. Vor einer Weiterverwendung die Lizenz des Ursprungsrepos prüfen.

## Beschaffung

```bash
python3 data/fetch_data.py
```

Das Skript lädt die Datei und prüft anschließend Zeilen-, Spaltenzahl und
Ausfallquote. Weicht etwas ab, bricht es ab — eine stillschweigend geänderte
Quelle würde sonst die Ergebnisse verfälschen.
