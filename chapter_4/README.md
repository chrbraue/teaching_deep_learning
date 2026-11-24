# Schematische Lernfehler und Fehlerzerlegung

Beide Beispiele verwenden dieselbe blaue Kurve des Generalisierungsfehlers und
sind **keine empirischen Messergebnisse**.

Mit der bestehenden Umgebung aus der Repository-README:

```sh
python chapter_4/generalization_gap.py
python chapter_4/risk_decomposition.py
```

Benötigt NumPy, Matplotlib, pdflatex mit den bereits dokumentierten TeX-Paketen
und pdftoppm aus Poppler. Die Dateien entstehen unter `chapter_4/output/`:
PDF, PNG und JSON je Beispiel. Beim Import wird nichts geschrieben.
Die PNG-Vorschau stammt direkt aus der PDF. Die JSON-Dateien enthalten die
Kurvenwerte und Kapazitäten der Abstandspfeile zur Nachprüfung.

`risk_common.py` bündelt die kleinen gemeinsamen Funktionen und Druckeinstellungen.
Die bestehende Risikozerlegung bleibt numerisch unverändert. Für eine qualitative
Kapazität c von 0 bis 1 gilt:

- Bayes-Risiko: 0,12 (konstant).
- Approximationsfehler: 0,65 exp(−4c).
- Schätzfehler: 0,02 + 0,45c².
- Minimales wahres Risiko innerhalb der Hypothesenmenge: Bayes plus Approximation.
- Generalisierungsfehler: Bayes plus Approximation plus Schätzfehler.
- Zusätzlicher schematischer Trainingsfehler: 0,65 exp(−5c) + 0,02.

Die Generalisierungslücke ist Generalisierungsfehler minus Trainingsfehler.
Sie ist nicht der Schätzfehler, der den Abstand zwischen dem gelernten wahren
Risiko und dem minimalen wahren Risiko innerhalb der Hypothesenmenge bezeichnet.
Die Pfeile markieren diese Differenzen jeweils bei einer festen Kapazität.
Die optimale Kapazität ist das Minimum der gemeinsamen Kurve auf dem festen
Raster. Die Verläufe sind kein allgemeines Gesetz über reale Lernverfahren.

Beide PDFs verwenden dieselben Abmessungen, Achsengrenzen und Schriftgrößen.
Der Generalisierungsfehler ist durchgehend blau, Trainingsfehler orange gestrichelt,
Schätzfehler orange strichpunktiert. Das Bayes-Risiko und das minimale wahre Risiko
innerhalb der Hypothesenmenge sind durch neutrale Grautöne und andere Linienarten
unterschieden. Die Grafik erläutert die Risikozerlegung relativ zur
Hypothesenmenge, nicht die klassische Bias-Varianz-Identität.

Die Druckausgabe nutzt echten LaTeX-Satz mit Computer-Modern-Mathematik über
PGF/pdflatex. Feste Metadaten erlauben bytegleiche Wiederholungsexporte in der
geprüften Umgebung. Exportverzeichnis und Python-Caches sind aus Git ausgeschlossen.


## ℓ₁-Norm-Regularisierung und Soft-Thresholding

```sh
python chapter_4/l1_regularization.py
python chapter_4/l1_regularization.py --interactive
```

Der erste Aufruf erzeugt die feste Druckfassung `output/l1_regularization.pdf`
mit PNG-Vorschau und einer JSON-Datei der Druckparameter. Der zweite erzeugt
zunächst dieselbe feste Druckfassung und öffnet zusätzlich eine Ansicht mit
Schiebereglern. Slider-Änderungen überschreiben keine Druckdateien.

Links wird das regularisierte Gewicht als Funktion des ursprünglichen Gewichts
gezeigt, rechts als Funktion von gamma. Identität bzw. ursprüngliches Gewicht
stehen grau gestrichelt, die Lösung orange und die Dämpfungspfeile blau.
Die Originalvorlage hat nur symbolische Skalen. Die normierten Druckwerte sind
gamma = 1, H_ii = 1 und rechts w_hat_i = 1. Sie reproduzieren die dargestellten
Skalenverhältnisse und sind keine aus historischen Daten rekonstruierten Parameter.

Beide Ansichten berechnen dieselbe Soft-Thresholding-Funktion:
`sign(w_hat) * max(abs(w_hat) - gamma/H_ii, 0)`.
Sie löst das skalare Teilproblem der quadratischen Approximation mit positiver
diagonaler Hessematrix. Erforderlich ist H_ii > 0 und gamma >= 0.

Interaktiv ändern Sie gamma (0 bis 3), w_hat_i (-2 bis 2) und H_ii (0,25 bis 4).
Die Achsen passen sich an, damit Schwellen und Lösungen sichtbar bleiben.
Die rechten Schwellenwerte lauten für beliebiges Vorzeichen H_ii * abs(w_hat_i).
Der aktuelle Schwellenwert gamma/H_ii und die Lösung werden numerisch angezeigt.
Blaue Punkte markieren die aktuellen Werte. Reset stellt die Druckwerte wieder her.

Für die Regler wird eine grafische Sitzung mit einem Matplotlib-GUI-Backend
wie TkAgg oder QtAgg benötigt. Falls nötig unter Linux/macOS ausdrücklich mit
`MPLBACKEND=TkAgg python chapter_4/l1_regularization.py --interactive` starten.
Tk bzw. Qt muss im eigenen Python-System verfügbar sein. Die statische PDF
benötigt keinen Bildschirm und kein GUI-Backend. Es werden keine zusätzlichen
Frameworks oder Notebooks verwendet und keine Pakete automatisch installiert.

Geprüft sind die analytischen Fälle, Kurven und Pfeile sowie Slider- und Reset-
Callbacks unter Agg. Zusätzlich wurde am 24.11.2026 ein automatisierter Test mit
einem tatsächlich geöffneten TkAgg-Fenster im Benutzerkontext durchgeführt.
Slider und Reset wurden dabei über Canvas-Mausereignisse geprüft. Eine manuelle
Benutzerbedienung wurde nicht geprüft. Für interaktive Nutzung bleiben ein
geeignetes GUI-Backend und eine erreichbare grafische Sitzung erforderlich.
