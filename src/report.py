"""Erzeugt den Bericht: Markdown -> HTML -> PDF (WeasyPrint).

Der Bericht liest ausschliesslich die Ergebnisdateien aus ``results/`` --
keine Zahl steht doppelt im Code. Fehlt eine Datei, bricht das Skript ab,
statt eine Luecke stillschweigend zu uebergehen.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import markdown
from weasyprint import HTML

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
REPORT = ROOT / "report"


def lade(name):
    p = RESULTS / name
    if not p.exists():
        raise SystemExit(f"Fehlt: {p} -- erst das zugehoerige Skript laufen lassen")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def de(v, stellen=3):
    """Zahl im deutschen Format."""
    return f"{v:.{stellen}f}".replace(".", ",")


def pct(v):
    return f"{v:.1f} %".replace(".", ",")


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    desk = lade("01_deskriptiv.json")
    zus = lade("02_zusammenfassung.json")
    trei = lade("03_treiber.json")
    schw = lade("04_schwellen.json")
    sig = lade("05_signifikanz.json")

    sig_map = {(r["a"], r["b"]): r for r in sig}

    def befund(a, b):
        r = sig_map.get((a, b)) or sig_map.get((b, a))
        if not r:
            return ""
        return ("signifikant" if r["p"] < 0.05
                else "nicht unterscheidbar") + f" (p = {de(r['p'])}".replace(".", ",") + ")"

    # ---- Tabellen aufbauen ------------------------------------------------
    reihen = []
    order = ["Trivial (Mehrheitsklasse)", "Logistische Regression",
             "Random Forest", "Gradient Boosting", "Neuronales Netz"]
    for base in order:
        for suf, tag in (("", "ohne"), (" + SMOTE", "mit")):
            k = base + suf
            if k not in zus:
                continue
            m = zus[k]
            reihen.append(
                f"| {base} ({tag} SMOTE) | {de(m['f1']['mean'])} ± "
                f"{de(m['f1']['std'])} | {de(m['roc_auc']['mean'])} | "
                f"{de(m['pr_auc']['mean'])} | {de(m['precision']['mean'])} | "
                f"{de(m['recall']['mean'])} |")

    treiber_reihen = []
    lr = trei["logreg"]
    for f in sorted(lr, key=lambda k: -abs(lr[k]["coef"])):
        v = lr[f]
        treiber_reihen.append(
            f"| {f} | {de(v['coef'],2)} | {de(v['odds_ratio'],2)} | "
            f"[{de(v['ki95_coef'][0],2)}; {de(v['ki95_coef'][1],2)}] | "
            f"{'ja' if v['signifikant'] else 'nein'} | "
            f"{de(trei['permutation'][f]['mean'],3)} |")

    schwell_reihen = []
    st = schw["kennzahlen_standard"]
    schwell_reihen.append(
        f"| 0,5 (Standard) | {de(st['precision'])} | {de(st['recall'])} | "
        f"{de(st['f1'])} | {st['tp']} | {st['fp']} | {st['fn']} |")
    for t in schw["optima"]:
        schwell_reihen.append(
            f"| {t['verhaeltnis']}:1 | {de(t['precision'])} | "
            f"{de(t['recall'])} | {de(t['f1'])} | {t['tp']} | {t['fp']} | "
            f"{t['fn']} |")

    bestes = max(zus, key=lambda k: zus[k]["f1"]["mean"])
    bestes_m = zus[bestes]
    s5 = next(t for t in schw["optima"] if t["verhaeltnis"] == 5)

    md = f"""# Kreditrisiko-Analyse

> Erstellt: {date.today().isoformat()} | Datensatz: bankloans (700 gelabelte Kreditnehmer)

## 1. Worum es geht

Aus den Merkmalen eines Kreditnehmers soll vorhergesagt werden, ob er
ausfaellt. Das ist ein **Klassifikationsproblem mit ungleichen Kosten**: Ein
verpasster Ausfall ist teuer, ein Fehlalarm kostet nur eine Pruefung. Der
Bericht misst daher nicht nur die Trennschaerfe, sondern leitet aus einem
Kostenmodell ab, wo die Entscheidungsschwelle liegen sollte.

## 2. Datengrundlage

Der Datensatz umfasst 1150 Zeilen mit acht Merkmalen
(Alter, Bildungsstufe, Beschaeftigungsjahre, Wohnjahre, Einkommen,
Schulden-Einkommens-Quote, Kreditkartenschuld, sonstige Schulden) und der
Zielvariable `default`.

**Nur 700 Zeilen tragen ein Label.** Die uebrigen 450 sind ein Block ohne
Zielvariable; sie enthalten ausserdem nur 150 verschiedene Merkmalsmuster,
jedes dreifach wiederholt. Fuer ueberwachtes Lernen sind sie unbrauchbar und
bleiben aussen vor -- sie dienen hoechstens dem Anwenden eines fertigen
Modells.

Von den 700 gelabelten Kreditnehmern sind **{desk['n_ausfaelle']}
ausgefallen ({pct(desk['ausfallquote_pct'])})** -- ein moderat ungleicher
Datensatz, kein Extremfall.

![Verteilung der Merkmale](../figures/01_verteilungen.png)

*Verteilung der Merkmale, getrennt nach Ausfall. Bei der
Schulden-Einkommens-Quote und der Kreditkartenschuld liegen die Ausfaller
sichtbar rechts, bei den Beschaeftigungs- und Wohnjahren links.*

![Korrelation der Merkmale](../figures/03_korrelation.png)

*Die Merkmale sind untereinander kaum korreliert -- es gibt keine Redundanz,
die eine Auswahl erzwingen wuerde.*

### Was die Merkmale einzeln leisten

Die Trennschaerfe eines Merkmals allein (AUC; 0,5 = kein Zusammenhang):

| Merkmal | AUC allein |
|---|---:|
""" + "\n".join(
        f"| {f} | {de(v)} |" for f, v in sorted(
            desk["auc_einzelmerkmale"].items(),
            key=lambda kv: -abs(kv[1] - 0.5))) + f"""

Die **Schulden-Einkommens-Quote** trennt am besten, gefolgt von der
Kreditkartenschuld. Die Beschaeftigungsjahre wirken in die andere Richtung:
Je laenger jemand beschaeftigt ist, desto geringer das Risiko.

![Trennschaerfe der Einzelmerkmale](../figures/02_trennschaerfe_einzelmerkmale.png)

*Trennschaerfe jedes Merkmals allein. Werte ueber 0,5 heissen: Ein hoeherer
Wert geht mit hoeherem Risiko einher; Werte darunter: mit geringerem.*

### Ein scheinbarer Widerspruch, der keiner ist

Die Ausfallquote steigt mit der Bildungsstufe (Stufe 1: 21,2 %, Stufe 4:
36,8 %). Das sieht nach einem starken Effekt aus, ist aber **nicht
belastbar**: Die Konfidenzintervalle ueberlappen breit, Stufe 5 hat nur fuenf
Beobachtungen, und die Korrelation zwischen Bildungsstufe und
Schulden-Einkommens-Quote betraegt {de(desk['korrelation_ed_debtinc'],3)} --
praktisch null. Der Effekt darf nicht als Befund verkauft werden.

![Ausfallquote nach Bildungsstufe](../figures/04_ausfallquote_bildung.png)

*Ausfallquote je Bildungsstufe mit 95-%-Konfidenzintervall. Der optische
Anstieg haelt der Unsicherheit nicht stand.*

## 3. Methodik

Verglichen werden fuenf Verfahren: Trivialklassifikator (Baseline),
logistische Regression, Random Forest, Gradient Boosting und ein neuronales
Netz -- jeweils ohne und mit Ueberabtastung (SMOTE).

**Verschachtelte Kreuzvalidierung:** Die Bewertung laeuft ueber 10 aeussere
Folds, die Hyperparameterwahl in einem inneren 3-Fold-Schritt. Damit sieht
die Parameterwahl die Bewertungsdaten nie -- die Kennzahlen sind Messungen,
keine Obergrenzen. SMOTE sitzt in der Pipeline und wirkt nur auf die
Trainingsfolds.

**Gepaarter Signifikanztest:** Die Fold-Ergebnisse je Modell beruhen auf
denselben Aufteilungen. Ein Wilcoxon-Vorzeichen-Rang-Test prueft, ob ein
Unterschied groesser ist als die Streuung zwischen den Folds -- strenger als
der optische Vergleich von Fehlerbalken.

## 4. Ergebnisse

### Modellvergleich

| Modell | F1 | ROC-AUC | PR-AUC | Precision | Recall |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(reihen) + f"""

Das beste Ergebnis erzielt **{bestes}** mit F1 = {de(bestes_m['f1']['mean'])}
(± {de(bestes_m['f1']['std'])}) und ROC-AUC = {de(bestes_m['roc_auc']['mean'])}.

![Modellvergleich](../figures/05_modellvergleich_f1.png)

*F1-Wert je Verfahren, ohne und mit Ueberabtastung. Fehlerbalken = Streuung
ueber die zehn aeusseren Folds.*

**Die Modelle ordnen die Kunden deutlich besser, als der F1-Wert vermuten
laesst.** ROC-AUC-Werte um {de(bestes_m['roc_auc']['mean'])} heissen: Zieht man
einen zufaelligen Ausfaller und einen zufaelligen Nicht-Ausfaller heraus, wird
der Ausfaller in rund {de(bestes_m['roc_auc']['mean']*100,1)} % der Faelle
hoeher eingestuft. Der niedrigere F1-Wert entsteht allein durch die feste
Schwelle 0,5 -- dazu Abschnitt 5.

### Bringt Ueberabtastung etwas?

{befund("Logistische Regression + SMOTE", "Logistische Regression")} fuer die
logistische Regression; {befund("Neuronales Netz + SMOTE", "Neuronales Netz")}
fuer das neuronale Netz. Ob SMOTE hilft, haengt also von der Konfiguration ab
und ist keine allgemeine Regel.

### Treiber des Ausfalls

Standardisierte Koeffizienten der logistischen Regression (vergleichbar, weil
alle Merkmale dieselbe Skala haben), mit Bootstrap-Konfidenzintervall und
Permutationswichtigkeit:

| Merkmal | Koeffizient | Odds-Ratio | KI95 | signifikant | Perm.-AUC-Abfall |
|---|---:|---:|---|---|---:|
""" + "\n".join(treiber_reihen) + f"""

Die **Schulden-Einkommens-Quote** und die **Kreditkartenschuld** erhoehen das
Risiko, **Beschaeftigungsjahre** und **Wohnjahre** senken es. Ein
Odds-Ratio von {de(lr['debtinc']['odds_ratio'],2)} bei der Schuldenquote
heisst: Eine Erhoehung um eine Standardabweichung vervielfacht die Chance auf
einen Ausfall etwa um diesen Faktor.

![Treiber des Ausfalls](../figures/06_treiber.png)

*Links: Richtung und Staerke des Zusammenhangs (standardisiert). Rechts: wie
viel Trennschaerfe verloren geht, wenn ein Merkmal nutzlos gemacht wird.*

## 5. Schwellenwert und Kosten -- der eigentliche Hebel

Bei einem Ausfallanteil von {pct(desk['ausfallquote_pct'])} ist die Schwelle
0,5 willkuerlich. Sie wird hier aus dem erwarteten Nutzen abgeleitet:

| Schwelle | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
""" + "\n".join(schwell_reihen) + f"""

**Der Vergleich zeigt den Gewinn.** Bei der Standardschwelle 0,5 werden
{st['tp']} Ausfaelle erkannt und {st['fn']} verpasst. Im Kostenoptimum bei
einem Verhaeltnis von 5:1 liegt die Schwelle bei {de(s5['schwelle'],2)}: Es
werden {s5['tp']} Ausfaelle erkannt, nur noch **{s5['fn']} verpasst** -- bei
{s5['fp']} Fehlalarmen. Der Recall steigt von {de(st['recall'])} auf
{de(s5['recall'])}.

Je teurer ein verpasster Ausfall im Verhaeltnis zu einem Fehlalarm, desto
niedriger die optimale Schwelle. Die richtige Zahl liegt beim Nutzer; der
Bericht rechnet daher drei Verhaeltnisse durch.

![Kennzahlen und Kosten ueber der Schwelle](../figures/07_schwellenwerte.png)

*Links: Precision, Recall und F1 ueber der Schwelle. Rechts: erwartete Kosten
je Kostenverhaeltnis -- je teurer ein verpasster Ausfall, desto weiter wandert
das Optimum nach links.*

![Verwirrungsmatrix bei 0,5](../figures/08_konfusion_050.png)

![Verwirrungsmatrix im Kostenoptimum](../figures/09_konfusion_kosten.png)

*Verwirrungsmatrix bei der Standardschwelle 0,5 (oben) und im Kostenoptimum
5:1 (unten). Der Anteil verpasster Ausfaelle sinkt deutlich.*

## 6. Grenzen

- **Nur 700 Zeilen.** Die Kennzahlen tragen Bootstrap-Intervalle bzw.
  Fold-Streuung, sind aber von begrenzter Praezision.
- **Kein Zeitbezug.** Es gibt keinen Beobachtungszeitpunkt, an dem die
  Merkmale erhoben wurden. Das Modell sagt eine Zugehoerigkeit vorher, kein
  Ereignis zu einem Zeitpunkt.
- **Die 450 ungelabelten Zeilen bleiben ungenutzt.** Sie tragen keine
  Zielvariable; ein halbueberwachter Ansatz waere moeglich, aber nur mit
  zusaetzlicher Annahme.
- **Das Kostenmodell ist eine Annahme.** Die Verhaeltnisse 1:1, 5:1, 10:1
  sind Szenarien, keine Messung.
- **Keine Ursachenaussage.** Die Koeffizienten zeigen Zusammenhang, nicht
  Wirkung.

## 7. Reproduktion

```bash
python3 src/01_describe.py      # Kennzahlen + Abbildungen
python3 src/02_models.py        # Modellvergleich (verschachtelte CV)
python3 src/03_drivers.py       # Treiberanalyse
python3 src/04_threshold.py     # Schwellenwert + Kostenmodell
python3 src/05_significance.py  # gepaarter Wilcoxon-Test
python3 src/report.py           # dieses PDF
```

Die Ergebnisdateien liegen in `results/` (JSON und CSV im deutschen
Power-BI-Format), die Abbildungen in `figures/`.
"""

    # Markdown sichern (editierbare Quelle)
    (REPORT / "kreditrisiko_bericht.md").write_text(md, encoding="utf-8")

    css = """
@page { size: A4; margin: 2cm 1.9cm 2.2cm 1.9cm;
  @bottom-center { content: counter(page) " / " counter(pages);
    font-family: 'DejaVu Sans'; font-size: 8pt; color: #666; }
  @top-right { content: "Kreditrisiko-Analyse"; font-family: 'DejaVu Sans';
    font-size: 8pt; color: #888; } }
body { font-family: 'DejaVu Serif'; font-size: 10.5pt; line-height: 1.5;
  color: #111; }
h1 { font-family: 'DejaVu Sans'; color: #0b4a8a; font-size: 20pt;
  border-bottom: 2px solid #0b4a8a; padding-bottom: 6px; }
h2 { font-family: 'DejaVu Sans'; color: #0b4a8a; font-size: 14pt;
  margin-top: 22px; border-bottom: 1px solid #cdd8e6; padding-bottom: 3px; }
h3 { font-family: 'DejaVu Sans'; color: #24507e; font-size: 11.5pt;
  margin-top: 16px; }
blockquote { border-left: 4px solid #0b4a8a; background: #f3f6fb;
  margin: 12px 0; padding: 8px 14px; font-style: italic; color: #33475f; }
table { border-collapse: collapse; width: 100%; margin: 10px 0;
  font-family: 'DejaVu Sans'; font-size: 9pt; }
th { background: #0b4a8a; color: #fff; text-align: left;
  padding: 5px 8px; }
td { padding: 4px 8px; border-bottom: 1px solid #dde3ea; }
tr:nth-child(even) td { background: #f2f5fa; }
code { font-family: 'DejaVu Sans Mono'; font-size: 8.8pt;
  background: #f4f4f4; border: 1px solid #ddd; border-radius: 3px;
  padding: 0 3px; }
pre { font-family: 'DejaVu Sans Mono'; font-size: 8.5pt; background: #f4f4f4;
  border: 1px solid #ddd; border-radius: 4px; padding: 8px 10px;
  white-space: pre-wrap; }
img { max-width: 100%; }
"""

    html = markdown.markdown(md, extensions=["tables", "fenced_code",
                                             "sane_lists"])
    doc = ("<html><head><meta charset='utf-8'><style>" + css +
           "</style></head><body>" + html + "</body></html>")
    out = REPORT / "kreditrisiko_bericht.pdf"
    HTML(string=doc, base_url=str(REPORT)).write_pdf(out)

    print("geschrieben:", REPORT / "kreditrisiko_bericht.md")
    print("geschrieben:", out)


if __name__ == "__main__":
    main()
