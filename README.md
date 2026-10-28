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
| Laden der Datenquelle | Daten wurden nach Download im Projektordner/Repository gespeichert |
| GitHub-Repository | offenes GitHub-Repository (9ykhzydjr4-max/nlp_analysis.git) |
| Vorverarbeitung | Bereinigung der Texte: Duplikate entfernen, Platzhalter/Sonderzeichen/Zahlen entfernen, Lemmatisierung, Stoppwörter entfernen, Verneinungen erhalten |
| Vektorisierung | Anwendung zwei Methoden: Bag-of-Words (ContVectorizer) und TF-IDF (TfidfVecotizer) |
| Semantishche Analyse | Anwendung zwei Verfahren: LDA (Latent Dirichlet Allocation) und LSA (Latent Semantic Analysis) |
| Ergebnissdiskussion | Ausgabe in Verzeichnis /output |
| Abhängigkeiten | Dokumentation in requirenment.txt |
|  |  |