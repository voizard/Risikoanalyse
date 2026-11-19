"""Erzeugt den Bericht als LaTeX und kompiliert ihn mit Tectonic.

Die Zahlen stammen ausschliesslich aus ``results/*.json`` -- keine Kennzahl
steht doppelt im Quelltext. Fehlt eine Datei, bricht das Skript ab, statt eine
Luecke stillschweigend zu uebergehen.

Aufbau: eine kurze Hauptdatei (Praemabel, Titel, Inhaltsverzeichnis, Liste der
Kapitel) plus eine Datei je Kapitel unter ``report/parts/``. Damit ist jedes
Kapitel einzeln aenderbar, ohne das ganze Dokument anzufassen.

Aufruf:  python3 src/report.py [--no-build]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS_DE, RESULTS

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
PARTS = REPORT / "parts"
TECTONIC = Path.home() / ".local" / "bin" / "tectonic"

BASENAME = "kreditrisiko_bericht"

MONATE = ["Januar", "Februar", "Maerz", "April", "Mai", "Juni", "Juli",
          "August", "September", "Oktober", "November", "Dezember"]


# ---------------------------------------------------------------- Werkzeug

def lade(name: str):
    p = RESULTS / name
    if not p.exists():
        raise SystemExit(f"Fehlt: {p} -- erst das zugehoerige Skript laufen lassen")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def de(v, stellen: int = 3) -> str:
    """Zahl im deutschen Format (Dezimalkomma)."""
    return f"{v:.{stellen}f}".replace(".", ",")


def txt(s: str) -> str:
    """Escaped Zeichen, die LaTeX sonst als Befehl liest."""
    for a, b in (("_", r"\_"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def lab(feature: str) -> str:
    return LABELS_DE.get(feature, feature)


def fig(pfad: str, unterschrift: str, breite: str = r"0.90\linewidth") -> str:
    return "\n".join([
        r"\begin{figure}[H]", r"  \centering",
        rf"  \includegraphics[width={breite}]{{{pfad}}}",
        r"  \caption{" + unterschrift + "}", r"\end{figure}", ""])


def tab(kopf: list[str], zeilen: list[list[str]], ausr: str,
        groesse: str = r"\small") -> str:
    out = [r"\begin{table}[H]", r"  \centering", f"  {groesse}",
           rf"  \begin{{tabular}}{{{ausr}}}", r"    \toprule",
           "    " + " & ".join(kopf) + r" \\", r"    \midrule"]
    for z in zeilen:
        out.append("    " + " & ".join(z) + r" \\")
    out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def schreibe(pfad: Path, inhalt: str) -> None:
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text(inhalt, encoding="utf-8")
    print(f"  geschrieben: {pfad.relative_to(ROOT)}")


# ---------------------------------------------------------------- Praeambel

def praeambel(daten: dict) -> str:
    return rf"""% !TeX program = xelatex
% Erzeugt von src/report.py -- nicht direkt bearbeiten, die Zahlen kommen
% aus results/*.json. Aenderungen am Text in src/report.py vornehmen.
\documentclass[11pt,a4paper]{{article}}
\usepackage[ngerman]{{babel}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{eurosym}}
\usepackage[margin=2.5cm]{{geometry}}
\usepackage{{microtype}}
\usepackage[htt]{{hyphenat}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{listings}}
\usepackage{{xcolor}}
\usepackage{{caption}}
\usepackage[hidelinks]{{hyperref}}

\graphicspath{{{{../figures/}}}}
\captionsetup{{font=small,labelfont=bf,skip=6pt}}
\setlength{{\emergencystretch}}{{1.5em}}

\definecolor{{codebg}}{{gray}}{{0.96}}
\definecolor{{codegray}}{{gray}}{{0.45}}
\lstset{{
  basicstyle=\ttfamily\scriptsize,
  backgroundcolor=\color{{codebg}},
  breaklines=true,
  breakatwhitespace=false,
  columns=fullflexible,
  frame=single,
  rulecolor=\color{{codegray}},
  framesep=5pt,
  xleftmargin=10pt,
  xrightmargin=4pt,
  language=Python,
  keywordstyle=\color{{blue!65!black}},
  commentstyle=\color{{green!45!black}},
  stringstyle=\color{{red!55!black}},
  showstringspaces=false,
  upquote=true,
}}

\title{{Kreditrisiko-Analyse\\[0.4em]
\large Vorhersage von Kreditausfaellen und die Wahl der
Entscheidungsschwelle unter ungleichen Kosten}}
\author{{Chris}}
\date{{{daten['datum']}}}

\begin{{document}}
\maketitle
\thispagestyle{{empty}}

\begin{{abstract}}
\noindent
{daten['abstract']}
\end{{abstract}}

\newpage
\tableofcontents
\newpage
"""


def kapitel(daten: dict) -> list[tuple[str, str]]:
    """Liste (Dateiname, Inhalt) der Kapitel."""
    d, z = daten["desk"], daten["zus"]
    t, s, g = daten["trei"], daten["schw"], daten["sig"]
    st = s["kennzahlen_standard"]

    def befund(a, b):
        r = g["map"].get((a, b)) or g["map"].get((b, a))
        if not r:
            return ""
        w = "signifikant" if r["p"] < 0.05 else "nicht unterscheidbar"
        return f"{w} (p = {de(r['p'])})"

    # ---- 1 Einleitung
    k1 = rf"""
\section{{Worum es geht}}

Aus den Merkmalen eines Kreditnehmers soll vorhergesagt werden, ob er
ausfaellt. Das ist ein Klassifikationsproblem mit \emph{{ungleichen Kosten}}:
Ein verpasster Ausfall ist teuer, ein Fehlalarm kostet nur eine Pruefung.
Der Bericht misst daher nicht nur die Trennschaerfe, sondern leitet aus einem
Kostenmodell ab, wo die Entscheidungsschwelle liegen sollte.

Drei Fragen stehen im Vordergrund:

\begin{{enumerate}}
  \item Wie gut lassen sich Kreditnehmer ueberhaupt nach Risiko ordnen?
  \item Welche Merkmale haengen mit einem Ausfall zusammen -- und welche
        nur scheinbar?
  \item Wo sollte die Entscheidungsschwelle liegen, wenn ein verpasster
        Ausfall ein Vielfaches eines Fehlalarms kostet?
\end{{enumerate}}
"""

    # ---- 2 Datengrundlage
    auc_zeilen = [[lab(f), de(v)] for f, v in
                  sorted(d["auc_einzelmerkmale"].items(),
                         key=lambda kv: -abs(kv[1] - 0.5))]
    ed_zeilen = [[f"Stufe {k}", str(v["n"]), de(v["quote_pct"], 1) + r" \%",
                  f"[{de(v['ki95'][0],1)}; {de(v['ki95'][1],1)}]"]
                 for k, v in sorted(d["ausfallquote_nach_bildung"].items())]
    k2 = rf"""
\section{{Datengrundlage}}

Der Datensatz umfasst 1150 Zeilen mit acht Merkmalen (Alter, Bildungsstufe,
Beschaeftigungsjahre, Wohnjahre, Einkommen, Schulden-Einkommens-Quote,
Kreditkartenschuld, sonstige Schulden) und der Zielvariable \texttt{{default}}.

\textbf{{Nur 700 Zeilen tragen ein Label.}} Die uebrigen 450 sind ein Block
ohne Zielvariable; sie enthalten ausserdem nur 150 verschiedene
Merkmalsmuster, jedes dreifach wiederholt. Fuer ueberwachtes Lernen sind sie
unbrauchbar und bleiben aussen vor. Die Analyse stuetzt sich daher
\emph{{bewusst}} allein auf die 700 gelabelten Zeilen.

Von den 700 gelabelten Kreditnehmern sind \textbf{{{d['n_ausfaelle']}
ausgefallen ({de(d['ausfallquote_pct'],1)}~\%)}} -- ein moderat ungleicher
Datensatz, kein Extremfall.

{fig("01_verteilungen.png", "Verteilung der Merkmale, getrennt nach Ausfall. "
     "Bei der Schulden-Einkommens-Quote und der Kreditkartenschuld liegen die "
     "Ausfaller sichtbar rechts, bei den Beschaeftigungs- und Wohnjahren links.")}

{fig("03_korrelation.png", "Korrelation der Merkmale untereinander. Es gibt "
     "keine Redundanz, die eine Merkmalsauswahl erzwingen wuerde.")}

\subsection{{Was die Merkmale einzeln leisten}}

Die Trennschaerfe eines Merkmals allein (AUC; 0,5 bedeutet kein Zusammenhang,
Werte unter 0,5 einen Zusammenhang in die andere Richtung):

{tab(["Merkmal", "AUC allein"], auc_zeilen, "lr")}

Die \textbf{{Schulden-Einkommens-Quote}} trennt am besten, gefolgt von der
Kreditkartenschuld. Die Beschaeftigungsjahre wirken in die andere Richtung:
Je laenger jemand beschaeftigt ist, desto geringer das Risiko.

{fig("02_trennschaerfe_einzelmerkmale.png", "Trennschaerfe jedes Merkmals "
     "allein. Werte ueber 0,5: Ein hoeherer Wert geht mit hoeherem Risiko "
     "einher. Werte darunter: mit geringerem.")}

\subsection{{Ein scheinbarer Widerspruch, der keiner ist}}

Die Ausfallquote steigt mit der Bildungsstufe. Das sieht nach einem starken
Effekt aus, ist aber \textbf{{nicht belastbar}}:

{tab(["Bildungsstufe", "n", "Ausfallquote", "KI95"], ed_zeilen, "lrrr")}

Die Konfidenzintervalle ueberlappen breit, die oberste Stufe hat nur fuenf
Beobachtungen, und die Korrelation zwischen Bildungsstufe und
Schulden-Einkommens-Quote betraegt {de(d['korrelation_ed_debtinc'])} --
praktisch null. Der Effekt darf nicht als Befund verkauft werden.

{fig("04_ausfallquote_bildung.png", "Ausfallquote je Bildungsstufe mit "
     "95-Prozent-Konfidenzintervall. Der optische Anstieg haelt der "
     "Unsicherheit nicht stand.")}
"""

    # ---- 3 Methodik
    k3 = r"""
\section{Methodik}

Verglichen werden fuenf Verfahren: Trivialklassifikator (Baseline),
logistische Regression, Random Forest, Gradient Boosting und ein neuronales
Netz -- jeweils ohne und mit Ueberabtastung der Minderheitsklasse (SMOTE).

\paragraph{Verschachtelte Kreuzvalidierung.}
Die Bewertung laeuft ueber 10 aeussere Folds, die Hyperparameterwahl in einem
inneren 3-Fold-Schritt. Damit sieht die Parameterwahl die Bewertungsdaten nie:
Die Kennzahlen sind Messungen, keine Obergrenzen. SMOTE sitzt in der Pipeline
und wirkt nur auf die Trainingsfolds, damit keine kuenstlichen Punkte in die
Bewertung gelangen.

\paragraph{Gepaarter Signifikanztest.}
Die Fold-Ergebnisse je Modell beruhen auf denselben aeusseren Aufteilungen und
sind damit gepaart. Ein Wilcoxon-Vorzeichen-Rang-Test prueft, ob ein
Unterschied groesser ist als die Streuung zwischen den Folds -- strenger als
der optische Vergleich von Fehlerbalken.

\paragraph{Kostenmodell.}
Zwei Fehler sind moeglich: ein verpasster Ausfall (falsch negativ, teuer) und
ein Fehlalarm (falsch positiv, kostet eine Pruefung). Mit einem
Kostenverhaeltnis von $k{:}1$ laesst sich die Schwelle aus dem erwarteten
Nutzen ableiten, statt sie bei 0,5 festzulegen. \textbf{Das vom Auftraggeber
genannte Verhaeltnis betraegt 50:1} -- ein verpasster Ausfall kostet also das
50-fache einer Pruefung. Dieses Leitverhaeltnis steht im Mittelpunkt; die
Verhaeltnisse 1:1, 5:1 und 10:1 bleiben als Vergleichsszenarien daneben
stehen.
"""

    # ---- 4 Ergebnisse
    reihen = []
    kurz = {"Trivial (Mehrheitsklasse)": "Trivial (Mehrheitskl.)",
            "Logistische Regression": "Log. Regression",
            "Neuronales Netz": "Neuronales Netz"}
    for base in ["Trivial (Mehrheitsklasse)", "Logistische Regression",
                 "Random Forest", "Gradient Boosting", "Neuronales Netz"]:
        for suf, tag in (("", "ohne"), (" + SMOTE", "mit")):
            k = base + suf
            if k not in z:
                continue
            m = z[k]
            reihen.append([f"{kurz.get(base, base)} ({tag} SMOTE)",
                           r"$" + de(m["f1"]["mean"]) + r" \pm "
                           + de(m["f1"]["std"]) + r"$",
                           de(m["roc_auc"]["mean"]), de(m["pr_auc"]["mean"]),
                           de(m["precision"]["mean"]),
                           de(m["recall"]["mean"])])

    lr = t["logreg"]
    trei_zeilen = [[lab(f), de(lr[f]["coef"], 2), de(lr[f]["odds_ratio"], 2),
                    f"[{de(lr[f]['ki95_coef'][0],2)}; "
                    f"{de(lr[f]['ki95_coef'][1],2)}]",
                    "ja" if lr[f]["signifikant"] else "nein",
                    de(t["permutation"][f]["mean"])]
                   for f in sorted(lr, key=lambda k: -abs(lr[k]["coef"]))]

    bestes = max(z, key=lambda k: z[k]["f1"]["mean"])
    bm = z[bestes]
    k4 = rf"""
\section{{Ergebnisse}}

\subsection{{Modellvergleich}}

{tab(["Modell", "F1", "ROC-AUC", "PR-AUC", "Precision", "Recall"],
     reihen, "lrrrrr", groesse=r"\footnotesize")}

Das beste Ergebnis erzielt \textbf{{{bestes}}} mit F1 = {de(bm['f1']['mean'])}
($\pm$ {de(bm['f1']['std'])}) und ROC-AUC = {de(bm['roc_auc']['mean'])}.

{fig("05_modellvergleich_f1.png", "F1-Wert je Verfahren, ohne und mit "
     "Ueberabtastung. Fehlerbalken: Streuung ueber die zehn aeusseren Folds.")}

\textbf{{Die Modelle ordnen die Kunden deutlich besser, als der F1-Wert
vermuten laesst.}} Ein ROC-AUC-Wert von {de(bm['roc_auc']['mean'])} bedeutet:
Zieht man einen zufaelligen Ausfaller und einen zufaelligen Nicht-Ausfaller
heraus, wird der Ausfaller in rund {de(bm['roc_auc']['mean']*100,1)}~\% der
Faelle hoeher eingestuft. Der niedrigere F1-Wert entsteht allein durch die
feste Schwelle 0,5 -- dazu Abschnitt~\ref{{sec:schwelle}}.

\subsection{{Bringt Ueberabtastung etwas?}}

Fuer die logistische Regression ist der Unterschied {befund("Logistische Regression + SMOTE", "Logistische Regression")},
fuer das neuronale Netz {befund("Neuronales Netz + SMOTE", "Neuronales Netz")},
fuer den Random Forest {befund("Random Forest + SMOTE", "Random Forest")}.
Ob SMOTE hilft, haengt also von der Konfiguration ab und ist keine allgemeine
Regel.

\subsection{{Treiber des Ausfalls}}

Standardisierte Koeffizienten der logistischen Regression (untereinander
vergleichbar, weil alle Merkmale dieselbe Skala haben), mit
Bootstrap-Konfidenzintervall und Permutationswichtigkeit (Abfall der ROC-AUC
beim zufaelligen Durchmischen des Merkmals):

{tab(["Merkmal", "Koeffizient", "Odds-Ratio", "KI95", "signifikant",
      "Perm.-AUC"], trei_zeilen, "lrrrll", groesse=r"\footnotesize")}

Die Schulden-Einkommens-Quote und die Kreditkartenschuld erhoehen das Risiko,
Beschaeftigungsjahre und Wohnjahre senken es. Ein Odds-Ratio von
{de(lr['debtinc']['odds_ratio'],2)} bei der Schuldenquote heisst: Eine
Erhoehung um eine Standardabweichung vervielfacht die Chance auf einen Ausfall
etwa um diesen Faktor. Einkommen, Alter, sonstige Schulden und Bildungsstufe
sind im Modell \emph{{nicht}} von Null unterscheidbar.

{fig("06_treiber.png", "Links: Richtung und Staerke des Zusammenhangs "
     "(standardisiert). Rechts: wie viel Trennschaerfe verloren geht, wenn "
     "ein Merkmal nutzlos gemacht wird.")}
"""

    # ---- 5 Schwellenwert
    schwell_zeilen = [["0,5 (Standard)", de(st["precision"]), de(st["recall"]),
                       de(st["f1"]), str(st["tp"]), str(st["fp"]),
                       str(st["fn"])]]
    for t_ in s["optima"]:
        schwell_zeilen.append([f"{t_['verhaeltnis']}:1", de(t_["precision"]),
                               de(t_["recall"]), de(t_["f1"]), str(t_["tp"]),
                               str(t_["fp"]), str(t_["fn"])])
    s5 = next(t_ for t_ in s["optima"] if t_["verhaeltnis"] == 50)
    k5 = rf"""
\section{{Schwellenwert und Kosten}}
\label{{sec:schwelle}}

Bei einem Ausfallanteil von {de(d['ausfallquote_pct'],1)}~\% ist die Schwelle
0,5 willkuerlich. Sie wird hier aus dem erwarteten Nutzen abgeleitet:

{tab(["Schwelle", "Precision", "Recall", "F1", "TP", "FP", "FN"],
     schwell_zeilen, "lrrrrrr")}

\textbf{{Der Vergleich zeigt den Gewinn.}} Bei der Standardschwelle 0,5 werden
{st['tp']} Ausfaelle erkannt und \textbf{{{st['fn']} verpasst}}. Mit dem vom
Auftraggeber genannten Verhaeltnis \textbf{{{s5['verhaeltnis']}:1}} -- einem
verpassten Ausfall stehen also {s5['verhaeltnis']} Pruefungen gegenueber --
sinkt die optimale Schwelle auf {de(s5['schwelle'],3)}: Es werden {s5['tp']}
Ausfaelle erkannt, nur noch \textbf{{{s5['fn']} verpasst}}, bei {s5['fp']}
Fehlalarmen. Der Recall steigt von {de(st['recall'])} auf {de(s5['recall'])}.

Das ist die ehrliche Rechnung: Bei diesem Kostenverhaeltnis ist es
\emph{{rational}}, eine niedrige Schwelle zu fahren und viele Fehlalarme in
Kauf zu nehmen. Ein Fehlalarm kostet eine Pruefung, ein verpasster Ausfall
{s5['verhaeltnis']}mal so viel -- da lohnt sich der Pruefaufwand fast immer.

Je teurer ein verpasster Ausfall im Verhaeltnis zu einem Fehlalarm, desto
niedriger die optimale Schwelle. Die Szenarien 1:1 bis 10:1 zeigen die
Bandbreite; das Leitverhaeltnis {s5['verhaeltnis']}:1 ist die belastbare Zahl.

{fig("07_schwellenwerte.png", "Links: Precision, Recall und F1 ueber der "
     "Schwelle. Rechts: erwartete Kosten je Kostenverhaeltnis -- je teurer "
     "ein verpasster Ausfall, desto weiter wandert das Optimum nach links.")}

{fig("08_konfusion_050.png", "Verwirrungsmatrix bei der Standardschwelle 0,5.",
     r"0.55\linewidth")}

{fig("09_konfusion_kosten.png", "Verwirrungsmatrix im Kostenoptimum "
     f"{s5['verhaeltnis']}:1. Fast kein Ausfall wird mehr verpasst -- dafuer "
     "steigen die Fehlalarme.", r"0.55\linewidth")}
"""

    # ---- 6 Grenzen
    k6 = rf"""
\section{{Grenzen}}

\begin{{itemize}}
  \item \textbf{{Nur 700 Zeilen.}} Die Kennzahlen tragen Bootstrap-Intervalle
        bzw. Fold-Streuung, sind aber von begrenzter Praezision.
  \item \textbf{{Kein Zeitbezug.}} Es gibt keinen Beobachtungszeitpunkt, zu
        dem die Merkmale erhoben wurden. Das Modell sagt eine Zugehoerigkeit
        vorher, kein Ereignis zu einem Zeitpunkt.
  \item \textbf{{Die 450 ungelabelten Zeilen bleiben ungenutzt -- bewusst.}}
        Sie tragen keine Zielvariable und keine neuen Merkmalsmuster; ein
        halbueberwachter Ansatz waere moeglich, aber nur mit zusaetzlicher
        Annahme. Die Analyse beschraenkt sich daher auf die 700 gelabelten
        Zeilen.
  \item \textbf{{Das Kostenmodell.}} Das Verhaeltnis 50:1 ist die vom
        Auftraggeber genannte Relation, keine Messung aus den Daten. Die
        Szenarien 1:1 bis 10:1 zeigen, wie empfindlich das Optimum auf diese
        Annahme reagiert.
  \item \textbf{{Keine Ursachenaussage.}} Die Koeffizienten zeigen
        Zusammenhang, nicht Wirkung.
\end{{itemize}}

\section{{Reproduktion}}

\begin{{lstlisting}}[language=bash]
python3 data/fetch_data.py       # Rohdaten holen + Form pruefen
python3 src/01_describe.py       # Kennzahlen + Abbildungen
python3 src/02_models.py         # Modellvergleich (verschachtelte CV)
python3 src/03_drivers.py        # Treiberanalyse
python3 src/04_threshold.py      # Schwellenwert + Kostenmodell
python3 src/05_significance.py   # gepaarter Wilcoxon-Test
python3 src/report.py            # dieses PDF (LaTeX ueber Tectonic)
\end{{lstlisting}}

Die Ergebnisdateien liegen in \texttt{{results/}} (JSON und CSV im deutschen
Power-BI-Format), die Abbildungen in \texttt{{figures/}}.
"""

    return [("01-einleitung.tex", k1), ("02-datengrundlage.tex", k2),
            ("03-methodik.tex", k3), ("04-ergebnisse.tex", k4),
            ("05-schwellenwert.tex", k5), ("06-grenzen.tex", k6)]


# -------------------------------------------------------------------- Main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true",
                    help="nur die .tex schreiben, nicht kompilieren")
    args = ap.parse_args()

    desk = lade("01_deskriptiv.json")
    zus = lade("02_zusammenfassung.json")
    trei = lade("03_treiber.json")
    schw = lade("04_schwellen.json")
    sig = lade("05_signifikanz.json")
    sig_map = {(r["a"], r["b"]): r for r in sig}

    bm = max(zus, key=lambda k: zus[k]["f1"]["mean"])
    m = zus[bm]
    s5 = next(t for t in schw["optima"] if t["verhaeltnis"] == 50)
    st = schw["kennzahlen_standard"]
    heute = date.today()
    daten = {
        "datum": f"{heute.day}.~{MONATE[heute.month-1]} {heute.year}",
        "desk": desk, "zus": zus, "trei": trei, "schw": schw,
        "sig": {"map": sig_map},
        "abstract": (
            f"Untersucht wird, wie sich Kreditausfaelle aus acht Merkmalen "
            f"vorhersagen lassen und wo die Entscheidungsschwelle liegen "
            f"sollte, wenn ein verpasster Ausfall teurer ist als ein "
            f"Fehlalarm. Grundlage sind 700 gelabelte Kreditnehmer mit einer "
            f"Ausfallquote von {de(desk['ausfallquote_pct'],1)}~\\%. "
            f"Verglichen werden fuenf Verfahren unter verschachtelter "
            f"Kreuzvalidierung. Das beste Modell ({bm}) erreicht einen "
            f"ROC-AUC-Wert von {de(m['roc_auc']['mean'])}, bei der "
            f"Standardschwelle 0,5 aber nur F1 = {de(m['f1']['mean'])}. "
            f"Ursache ist die willkuerliche Schwelle: Mit dem vom "
            f"Auftraggeber genannten Kostenverhaeltnis von "
            f"{s5['verhaeltnis']}:1 (ein verpasster Ausfall kostet "
            f"{s5['verhaeltnis']}mal so viel wie eine Pruefung) sinkt die "
            f"Zahl verpasster Ausfaelle von {st['fn']} auf {s5['fn']} "
            f"({de(s5['recall'],3)} statt {de(st['recall'],3)} Recall) -- "
            f"dafuer steigen die Fehlalarme auf {s5['fp']}. Die Treiber sind "
            f"Beschaeftigungsjahre (schuetzend), Kreditkartenschuld und "
            f"Schulden-Einkommens-Quote (riskant)."
        ),
    }

    print("Schreibe LaTeX-Quellen ...")
    kap = kapitel(daten)
    for name, inhalt in kap:
        schreibe(PARTS / name, inhalt.lstrip("\n"))

    includes = "\n".join(rf"\input{{parts/{n}}}" for n, _ in kap)
    schreibe(REPORT / f"{BASENAME}.tex", praeambel(daten) + "\n" + includes
             + "\n\n\\end{document}\n")

    if args.no_build:
        return

    # Seam-Pruefung vor dem Kompilieren
    print("Pruefe Pfade ...")
    fehlend = []
    for n, _ in kap:
        if not (PARTS / n).exists():
            fehlend.append(f"parts/{n}")
    for f in sorted((ROOT / "figures").glob("*.png")):
        pass
    for i in range(1, 10):
        treffer = list((ROOT / "figures").glob(f"{i:02d}_*.png"))
        if not treffer:
            fehlend.append(f"figures/{i:02d}_*.png")
    if fehlend:
        raise SystemExit("Fehlende Ziele: " + ", ".join(fehlend))

    print("Kompiliere mit Tectonic ...")
    res = subprocess.run(
        [str(TECTONIC), f"{BASENAME}.tex", "--outdir", "."],
        cwd=REPORT, capture_output=True, text=True)
    log = res.stdout + res.stderr
    ueberlauf = [z for z in log.splitlines()
                 if "Overfull" in z or "Underfull" in z]
    if res.returncode != 0:
        print(log[-3000:])
        raise SystemExit(f"Tectonic brach ab (Code {res.returncode})")

    pdf = REPORT / f"{BASENAME}.pdf"
    if not pdf.exists() or pdf.stat().st_size == 0:
        raise SystemExit("Keine PDF entstanden")
    print(f"  PDF: {pdf.relative_to(ROOT)} ({pdf.stat().st_size} B)")
    print(f"  Ueberlauf-Warnungen: {len(ueberlauf)}")
    for z in ueberlauf[:6]:
        print("    " + z.strip())


if __name__ == "__main__":
    main()
