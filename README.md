# NLP-Projekt: Consumer Complaints

Beschreibung des Projekts

- Aufgabenstellung: unstrukturierte Beschwerdetexte durch eine NLP-Pipeline analysieren
- Ziel: einen Überblick über die dringlichsten Probleme zu erhalten
- Verwendete Datenquelle: shashwatwork/consume-complaints-dataset-fo-nlp
- Kaggle-Datensatz ist ein Beispieldatensatz, der unstrukturierte Consumer Complaints enthält
- Zur Analyse wurden keine echten kommunalen Bürgerbeschwerden verwendet. 

# Aufgabenerfüllung

| Anforderung | Umsetzung |
|---|---|
| Beispieldatenqulle | Kaggel-Datensatz shashwatwork/consume-complaints-dataset-fo-nlp |
| Laden der Datenquelle | Daten wurden nach Download im Projektordner/Repository /data abgelegt |
| GitHub-Repository | offenes GitHub-Repository (9ykhzydjr4-max/nlp_analysis.git) |
| Vorverarbeitung | Bereinigung der Texte: Duplikate entfernen, Platzhalter/Sonderzeichen/Zahlen entfernen, Lemmatisierung, Stoppwörter entfernen, Verneinungen erhalten |
| Vektorisierung | Anwendung zwei Methoden: Bag-of-Words (ContVectorizer) und TF-IDF (TfidfVecotizer) |
| Semantishche Analyse | Anwendung zwei Verfahren: LDA (Latent Dirichlet Allocation) und LSA (Latent Semantic Analysis) |
| Ergebnissdiskussion | Ausgabe in Verzeichnis /output |
| Abhängigkeiten | Dokumentation in requirenment.txt |

# Vorgehen

| Schritt | Skript | Inhalt | Bilbiothek |
|---|---|---|---|
| 1 | 01_vorverarbeitung.py | Leere Einträge und Duplikate entfernen, Platzhalter (xxxx), Zahlen und Sonderzeichen entfernen, Lemmatisierung, Stoppwörter entfernen (Verneinungen bleiben erhalten) | pandas, re, nltk, spaCy |
| 2 | 02_vektorisierung.py | Bag-of-Words und TF-IDF (Uni- und Bigramme, min_df=10, max_df=0.5, 5.000 Begriffe), Vergleich beider Verfahren | scikit-learn, matplotlib |
| 3 | 03_themenextration.py | Bestimmung der Themenanzahl k über die c_v-Kohärenz, Themenextraktion mit LSA (TF-IDF) und LDA (Bag-of-Words), Abgleich mit den Produktkategorien, interaktive Visualisierung | cikit-learn, gensim, pyLDAvis |
| - | main.py | Führt alle drei Skripte nacheinander aus und listet die erzeugten Dateien auf | - |

# Projektstruktur

```text
npl_analysis/
|-- data
|   |-- complaints_processes.csv        # Rohdaten (nicht in Repository)
|-- output
|   |-- complaints_cleaned.csv          # nicht in Repository
|   |-- complaints_vektorisierung.csv
|   |-- complaints_vektorisierung.png
|   |-- lda_grafik.html
|   |-- themen_kohaerenz.csv
|   |-- themen_kohaerenz.png
|   |-- themen_lda.csv
|   |-- themen_lsa.csv
|   |-- themen_produkt_abgleich.csv
|   |-- vektoren.joblib                 # nicht in Repository
|   |-- vektorisierung_top_begriffe.csv
|-- .gitignore
|-- 01_vorverarbeitung.py
|-- 02_vektorisierung.py
|-- 03_themenextraktion.py
|-- README.md
|-- main.py
|-- requirements.txt
```

Die Dateien "complaints_processes.csv, complaints_cleaned.csv, vektoren.joblib" wurden 
aufgrund der Dateigröße per .gitignore von der Versionskontrolle ausgeschlossen 
und nicht in das Repository geladen. 

Die Ergebnisse wurden mittels den Skipten aus den Rohdaten erzeugt.

# Technische Anwendung

Voraussetzung: Python 3.12
Entwicklungsumgebung: VS Code V1.143.0 unter Windows 11

Repository klonen:
        git clone https://github.com/9ykhzydjr4-max/nlp_analysis.git
        cd nlp_analysis

Installation:   
        pip install -r requirements.txt

Datensatz bereit stellen:
        Download: https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp
        Datei entpacken und im Verzeichnis data/ ablegen.


# Ausführung

Gesamte Analyse: 
        python main.py          # Durchführung aller Schritte, Dauer ca. 20min

Analyse in Schritten:
        python main.py --ab 2   # Vektorisierung, Vorverarbeitung ist schon erfolgt
        python main.py --ab 3   # Themenextraktion

Für schnelle Testläufe kann im Skript 01_vorverarbietung.py der Wert TESTING = True gesetzt werden.
Es wird eine Stichprobe von 1000 Datensätzen verarbeitet. 

# Ergebnisse

|---|---|
| Datei | Inhalt |
| complaints_cleaned.csv | Bereinigte und lemmatisierte Beschwerdetexte |
| complaints_vektorisierung.csv /.png | Häufigste Wörter in BoW und TF-IDF |
| lda_grafik.html | Grafische Darstellung von LDA |
| themen_kohaerenz.csv / .png | c_v Kohärenz von LSA und LDA für k = 4 bis k = 12 |
| themen_lsa.csv, themen_lda.csv | Themen mit ihren Top-Wörtern |
| themen_produkt_abgleich.csv | Abgleich der LDA-Themen mit den Produktkategorien |
| vektoren.joblib | BoW- und TF-IDF-Matrizen inklusive Vektorisierer |
| vektorisierung_top_begriffe.csv | Verlgeich der Top-Begriffe von BoW und TF-IDF |

# Erkentnissinterpretation

    * Datensätze waren schon vorverarbeitet (Kleinschreibung, keine Satzzeichen, Standard-Stoppwörter entfernt)
    * Bei der Vektorisierung hebt BoW die häufigkeit der Wörter hervor, TF-IDF die unterschiedliche Begriffe
    * Themenanzahl: Für das Modell wurde k = 8 durch die Tolerenz = 0.01 gewählt. 
    * Themen: 
        - LSA: bündelt die Wörter, die Themen können weniger unterschieden werden
        - LDA: interpretiert die Themen klarer



