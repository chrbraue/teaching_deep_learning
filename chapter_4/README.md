# Schematische Fehlerzerlegung

Dieses Beispiel illustriert die Zerlegung des Generalisierungsrisikos.
Die Kurven sind **keine empirischen Messergebnisse**.

Mit der bestehenden Umgebung aus der Repository-README:

```sh
python chapter_4/risk_decomposition.py
```

Benötigt NumPy, Matplotlib, pdflatex mit den bereits dokumentierten TeX-Paketen
und pdftoppm aus Poppler. Die Dateien entstehen unter `chapter_4/output/`:
`risk_decomposition.pdf`, `.png` und `.json`. Beim Import wird nichts erzeugt.

Für eine qualitative Kapazität c zwischen 0 und 1 werden einfache Funktionen
verwendet: Bayes-Risiko 0,12, Approximationsfehler 0,65 exp(−4c), Schätzfehler
0,02 + 0,45c². Das beste Risiko innerhalb der Hypothesenmenge ist Bayes plus
Approximation; das Gesamtrisiko enthält zusätzlich den Schätzfehler. Grün wird
stets aus diesen Komponenten berechnet. Die optimale Kapazität ist nur das
Minimum auf dem festen schematischen Raster, keine reale Modellwahl.

Der graue Verlauf ist damit das beste Klassenrisiko, nicht allein der
Approximationsfehler. Der Doppelpfeil bezeichnet den Abstand zur Bayes-Basis.
Ein steigender Schätzfehler ist keine allgemeine Monotoniebehauptung über
Lernverfahren. Die Grafik illustriert die Risikozerlegung relativ zur
Hypothesenmenge, nicht die klassische Bias-Varianz-Zerlegung.

Druckausgabe mit Computer-Modern-Mathematik über den PGF/pdflatex-Export. PDF-
Metadaten sind für wiederholbare Ausgaben festgelegt; die PNG-Vorschau stammt
direkt aus der PDF. Alle Kurven sind zusätzlich durch Linienarten unterschieden.
Es wird keine Simulation und kein Training ausgeführt.
