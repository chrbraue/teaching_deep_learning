# teaching deep learning

Verständliche, reproduzierbare Python-Beispiele zum Deep-Learning-Skript der
Leibniz FH Hannover. Die Dateinamen beschreiben den Inhalt und sind unabhängig
von der Nummerierung im Skript.

## Einrichtung

Getestet unter Linux mit Python 3.12.16, NumPy 2.2.6 und Matplotlib 3.11.2.
Im geklonten Repository:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows: `py -3.12 -m venv .venv`, dann `.venv\Scripts\activate`.
Andere Betriebssysteme und Python-Versionen sind bisher nicht geprüft.
Es werden keine Datensätze heruntergeladen und keine ML-Frameworks benötigt.

Für den Druckexport müssen zusätzlich **pdflatex** und **pdftoppm** (Poppler)
im PATH verfügbar sein. Die TeX-Installation benötigt die Pakete `amsmath`,
`bm`, `pgf`, `geometry`, `hyperref`, `underscore` und Computer-Modern-Schriften.
Matplotlib verwendet außerdem das vorhandene `scrextend` für die Schriftgrößen
(optional mit Backend-Fallback). Diese Komponenten waren in der geprüften
Umgebung bereits vorhanden; die Skripte installieren nichts automatisch.

## Beispiele ausführen

```sh
python chapter_1/logistic_regression.py
python chapter_1/logistic_regression_basis_functions.py
```

Die Skripte schreiben ausschließlich in `chapter_1/output/`:

- `logistic_regression.pdf` und `.png`: lineare logistische Regression auf
  linear trennbaren synthetischen Prüfungsdaten und nicht linear trennbaren
  zweidimensionalen Daten;
- `logistic_regression_basis_functions.pdf` und `.png`: ein auf quadratischen
  Merkmalen trainiertes Modell, dargestellt im Merkmals- und im Objektraum;
- Daten als CSV sowie Modellparameter, Trainingskonfiguration und Genauigkeiten
  als JSON mit entsprechenden semantischen Dateinamen.

Der Ausgabeordner ist aus Git ausgeschlossen. Zum Aufruf aus einem anderen
Arbeitsverzeichnis den vollständigen Skriptpfad verwenden. Ohne Bildschirm
unter Linux/macOS bei Bedarf `MPLBACKEND=Agg` vor den Aufruf setzen. Beim Import
werden weder Training noch Dateiausgaben ausgelöst.

## Daten und mathematische Bedeutung

`chapter_1/common.py` enthält die gemeinsame Datenerzeugung, das Training und
die Darstellung. Datenerzeugung wird nicht doppelt implementiert.

Die Trainingsmenge ist S = {(x^(i), y^(i))}. x bezeichnet ein Objekt, y das Label,
theta=(w,b) die Modellparameter. Die Wahrscheinlichkeit für Klasse 1 ist
sigma(wᵀx+b); die diskrete Hypothese h_theta liefert Klasse 1 genau dann, wenn
der Score mindestens null ist. Sterne kennzeichnen Klasse 1, Kreuze Klasse 0.
Die gemeinsame Farbskala reicht stets von 0 bis 1; die rote Linie bezeichnet
die Nulllinie des Scores und damit Wahrscheinlichkeit 0,5.

Das erste Beispiel links verwendet 23 fest vorgegebene synthetische Prüfungs-
objekte, keine personenbezogenen Daten. Rechts: 100 normalverteilte Objekte,
Seed 7, Standardabweichung 5. Klasse 0 gilt bei x1²+x2² <= 36, sonst Klasse 1.
Diese Daten sind eine reproduzierbare Neufassung der historischen Beispiele,
keine punktgenaue Rekonstruktion der früheren MATLAB-Stichprobe.

Das Beispiel zu Basisfunktionen verwendet **exakt dieselben Objekte und Labels**:
phi_1(x)=x1² und phi_2(x)=x2². Nur ein Modell wird auf diesen Merkmalen trainiert.
Links zeigt sigma(wᵀphi+b) einen linearen Rand im Merkmalsraum; rechts zeigt
sigma(wᵀphi(x)+b) den nichtlinearen Rand im ursprünglichen Objektraum. Beide
Darstellungen verwenden dieselben Parameter. Der gelernte Rand ist eine Ellipse
und muss nicht exakt mit dem Kreis übereinstimmen, der die Datenlabels definiert.

Trainiert wird die mittlere binäre Kreuzentropie ohne Regularisierung durch
volles Gradientenverfahren, Lernrate 0,3. Intern wird standardisiert; die
zurückgegebenen Parameter gelten in Original- bzw. phi-Koordinaten. Die linearen
Modelle erhalten jeweils 12.000 Schritte, das Modell auf quadratischen Merkmalen
100.000 Schritte. Bei trennbaren Daten besitzt unregularisierte logistische
Regression kein endliches Maximum-Likelihood-Optimum: Dargestellt wird ein
festgelegter endlicher Trainingsstand. Die Kreuzentropie minimiert nicht direkt
den Klassifikationsfehler; Trainingsgenauigkeit ist kein Generalisierungsnachweis.

Standardergebnisse: lineares Beispiel links 100 %, rechts 60 %; quadratische
Basisfunktionen 100 %. Historische Genauigkeiten werden nicht künstlich erreicht.

## Typografie und Druckfassung

Die Druck-PDF verwendet echtes LaTeX über Matplotlibs PGF-Backend mit pdflatex.
Wie im Skript wird keine alternative Mathematikschrift geladen: Computer Modern,
aufrechte Roman-Schrift für `\mathrm`, kursive Variablen und fette Vektoren.
Text und Überschriften verwenden Computer Modern Sans. Die PDF ist für eine
Einbindung mit 16 cm Breite vorgesehen; Plotflächen bleiben geometrisch unverzerrt.
Die PNG-Vorschau wird mit pdftoppm direkt aus der gesetzten PDF erzeugt.

Flächen werden rasterisiert; Linien und Punkte bleiben Vektorelemente. Symbole
und Ränder sind auch ohne Farbe unterscheidbar. Zeitabhängige PDF-Metadaten,
Trailer-IDs und zusätzliche temporäre pdfTeX-Dateiinformationen werden unterdrückt.
Bytegleiche Wiederholungsexporte sind in der geprüften Umgebung nachgewiesen,
nicht für beliebige Python-/TeX-Versionen oder Plattformen garantiert.

## Qualitätskontrolle

```sh
python chapter_1/validate.py
```

Die Prüfung belegt fehlende lineare Trennbarkeit im ursprünglichen Raum durch
ein konkretes Konvexkombinations-Zertifikat. Sie prüft Trennbarkeit nach
quadratischer Transformation, Modellgenauigkeiten, identische Daten und Scores
sowie analytische Punkte auf dem Entscheidungsrand. Die exportierten CSVs
`logistic_regression_right_data.csv` und
`logistic_regression_basis_functions_data.csv` müssen bytegleich sein.

## Optional: selbst mit Parametern experimentieren

```sh
python chapter_1/logistic_regression.py --interactive
python chapter_1/logistic_regression_basis_functions.py --interactive
```

Nach Erzeugung der unveränderlichen Standarddruckfassung öffnet sich eine
zusätzliche Ansicht mit Reglern für w1, w2 und b. Man verändert die Parameter
direkt, ohne Nachtraining. Im Basisfunktionen-Beispiel wirken die Parameter auf
die quadratischen Merkmale. Die Regler überschreiben keine Druckdateien.

Ein funktionierendes Matplotlib-GUI-Backend (z.B. Tk oder Qt) ist hierfür nötig,
aber keine Pflichtabhängigkeit der statischen Druckfassung. Die GUI verwendet
MathText mit Computer-Modern-Mathematikschrift. Regler-Callbacks wurden mit Agg
geprüft; manuelle Fensterbedienung ist in der bisherigen Prüfung noch offen.

## Schematische Fehlerzerlegung

Unter [chapter_4](chapter_4/README.md) liegt eine schematische Risikozerlegung. Aufruf:
`python chapter_4/risk_decomposition.py`. Die bestehenden Abhängigkeiten genügen.
