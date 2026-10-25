# %% Bibliotheken importieren
import pandas as pd
import re
import pandas as pd
import nltk
import spacy
from os import path
from pathlib import Path
from collections import Counter

# %% Lokale Daten einlesen
PROJECT = Path(__file__).parent if "__file__" in globals() else Path.cwd()
pfad = PROJECT / "data" / "complaints_processed.csv"

dataframe = pd.read_csv(PROJECT / pfad, encoding="utf-8")
print("Geladene Datei:", pfad)

# %% Einblick in DataFrame
#print(dataframe.shape)
#print(dataframe.columns)
#print(dataframe.head())
# print(dataframe["narrative"].iloc[0]) # %% iloc[0] gibt die erste Zeile der Spalte "narrative" zurück

texte = dataframe["narrative"]

for wort in ["the" , "and", "not", "never", "xxxx"]:
    anzahl = texte.str.contains(rf"\b{wort}\b").sum()
    print(f"Das Wort '{wort}' kommt {anzahl} mal in der Spalte 'narrative' vor.")

# %% Indexspalte, leere Einträge und Duplikate entfernen
dataframe = dataframe.drop(columns=["Unnamed: 0"])

vorher = len(dataframe)
dataframe = dataframe.dropna(subset=["narrative"])          # entfernt die NaN-Zeilen
dataframe = dataframe.drop_duplicates(subset="narrative")
print(f"Datensätze Vorher: {vorher}, nachher: {len(dataframe)}")

# %% Sonderzeichen und Zahlen entfernen, überflüssige Leerzeichen entfernen
def clean_text(text):
    # Entferne Sonderzeichen und Zahlen
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # Entferne überflüssige Leerzeichen
    text = re.sub(r"\s+", " ", text).strip()
    return text

dataframe["cleaned_narrative"] = dataframe["narrative"].apply(clean_text)
print("Beispiel für bereinigte Sonderzeichen:", dataframe["cleaned_narrative"].str.contains(r"[^a-zA-Z\s]").sum())
print("Beispiel für bereinigte Leerzeichen:", dataframe["cleaned_narrative"].str.contains(r"\s+").sum())

# %% Stoppwörter entfernen
nltk.download("stopwords", quiet=True)
stoppwoerter = set(nltk.corpus.stopwords.words("english"))

# Verneinungen behalten (siehe Konzept Phase 1)
verneinungen = {"no", "not", "nor", "never", "without"}

# Eigene Stoppwörter
eigene = {"would", "could", "also", "get", "say", "tell", "ask", "go", "want",
          "know", "make", "take", "even", "still", "back", "day", "time",
          "year", "month", "week", "call", "please", "thank", "help",
          "since", "however", "one", "two"}

stoppwoerter = (stoppwoerter | eigene) - verneinungen

print("Anzahl der Stoppwörter:", len(stoppwoerter))

# %% Lemmatisierung
TESTING = True

if TESTING:
    dataframe = dataframe.sample(n=1000, random_state=42)  # Für Testzwecke eine Stichprobe von 1000 Dokumenten ziehen

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # Lade das englische Modell ohne Parser und NER

tokens_list = []
for document in nlp.pipe(dataframe["cleaned_narrative"], batch_size=1000):
    tokens = [token.lemma_ for token in document if token.text.lower() not in stoppwoerter]
    tokens_list.append(tokens)

dataframe["tokens"] = tokens_list
dataframe["cleaned_narrative"] =dataframe["tokens"].str.join(" ")

dataframe = dataframe[dataframe["tokens"].str.len() >= 5]  # Filtere Dokumente mit weniger als 5 Tokens heraus
print("Datensätze nach Lemmatisierung und Stoppwortentfernung:", len(dataframe))

# %% Vergleich
print("Rohtext Beispiel:", dataframe["narrative"].iloc[0][:300])
print("Bereinigter Text Beispiel:", dataframe["cleaned_narrative"].iloc[0][:300])

# %% häufig vorkommende Wörter
zaehler = Counter(wort for liste in dataframe["tokens"] for wort in liste)
print("Häufigste Wörter:", zaehler.most_common(25))

# %% Ausgabe speichern
output_path = PROJECT / "output" 
ziel = output_path / "complaints_cleaned.csv"
dataframe.to_csv(ziel, index=False, encoding="utf-8")
print(f"Datensätze gespeichert: {ziel} ({len(dataframe)} Zeilen)")
