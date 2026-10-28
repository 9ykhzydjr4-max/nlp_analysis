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

| Schritt  | Skript | Inhalt | Bilbiothek |
| 1 | 01_vorverarbeitung.py | Leere Einträge und Duplikate entfernen, Platzhalter (xxxx), Zahlen und Sonderzeichen entfernen, Lemmatisierung, Stoppwörter entfernen (Verneinungen bleiben erhalten) | pandas, re, nltk, spaCy |
| 2 | 02_vektorisierung.py | Bag-of-Words und TF-IDF (Uni- und Bigramme, min_df=10, max_df=0.5, 5.000 Begriffe), Vergleich beider Verfahren | scikit-learn, matplotlib |
| 3 | 03_themenextration.py | Bestimmung der Themenanzahl k über die c_v-Kohärenz, Themenextraktion mit LSA (TF-IDF) und LDA (Bag-of-Words), Abgleich mit den Produktkategorien, interaktive Visualisierung | cikit-learn, gensim, pyLDAvis |
| - | main.py | Führt alle drei Skripte nacheinander aus und listet die erzeugten Dateien auf | - |

# Projektstruktur

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

Die Dateien "complaints_processes.csv, complaints_cleaned.csv, vektoren.joblib" wurden aufgrund der Dateigröße per .gitignore von der Versionskontrolle ausgeschlossen. 
Die Ergebnisse wurden mittels den Skipten aus den Rohdaten erzeugt.