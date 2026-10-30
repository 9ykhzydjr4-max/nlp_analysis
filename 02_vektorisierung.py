from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

PROJECT = Path(__file__).parent if "__file__" in globals() else Path.cwd()
AUSGABE = PROJECT / "output"

dataframe = pd.read_csv(AUSGABE / "complaints_cleaned.csv", encoding="utf-8")
texte = dataframe["cleaned_narrative"]
print("Anzahl der Dokumente für Vektorisierung:", len(texte))

# Parameter für Vektorisierung
parameter = dict(
    min_df=10,           # minimale Dokumentenhäufigkeit
    max_df=0.5,          # maximale Dokumentenhäufigkeit
    max_features=5000,   # maximale Anzahl an Wörter
    ngram_range=(1, 2),  # Einzelwörter/Wortpaare
)

# Bag-of-Words
bow_vectorizer = CountVectorizer(**parameter)
bow_X = bow_vectorizer.fit_transform(texte)
print("Bag-of-Words Matrix:", bow_X.shape)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(**parameter)
tfidf_X = tfidf_vectorizer.fit_transform(texte)
print("TF-IDF Matrix:", tfidf_X.shape)

def sparsamkeit_X(X):
    # "Anteil der Nullen in Matrix"
    n_zeros = (X != 0).sum()
    n_elements = X.shape[0] * X.shape[1]
    sparsity = 1 - n_zeros / n_elements
    return sparsity

print(f"Bag-of-Words Sparsamkeit: {sparsamkeit_X(bow_X):.2%}")
print(f"TF-IDF Sparsamkeit: {sparsamkeit_X(tfidf_X):.2%}")

# Häufigste Wörter von BoW und TF-IDF
def meist_woerter(vectorizer, X,  n=15):
    # Summiere die Spalten (Wörter) der Matrix
    summe = np.asarray(X.sum(axis=0)).ravel() 
    # Erstelle ein DataFrame mit Wörtern und deren Häufigkeit
    woerter = vectorizer.get_feature_names_out()

    # Sortiere nach Häufigkeit absteigend
    # words_df = words_df.sort_values(by="count", ascending=False).head(n)
    words_df = pd.Series(summe, index=woerter).nlargest(n)  
    return words_df

meist_bow = meist_woerter(bow_vectorizer, bow_X)
meist_tfidf = meist_woerter(tfidf_vectorizer, tfidf_X)

print(meist_bow.head())
print(meist_tfidf.head())

vergleich = pd.DataFrame({
    # "BoW Wörter": meist_bow.set_index("word")["count"],
    # "TF-IDF Wörter": meist_tfidf.set_index("word")["count"]

    "BoW_Wort": meist_bow.index, "BoW_Anzahl": meist_bow.values.astype(int),
    "TFIDF_Wort": meist_tfidf.index, "TFIDF_Gewicht": meist_tfidf.values.round(1),
# }).fillna(0)    
})

print("Häufigste Wörter in BoW und TF-IDF:")
print(vergleich)

# Ausgabe speichern
DATEI = AUSGABE / "complaints_vektorisierung.csv"
vergleich.to_csv(DATEI, index=False, encoding="utf-8")
print(f"Datensätze gespeichert: {DATEI } ({len(vergleich)} Zeilen)")

# Ausgabe als Grafik
fig, achsen = plt.subplots(1, 2, figsize=(13, 6))
meist_bow[::-1].plot.barh(ax=achsen[0], color="steelblue", title="Bag-of-Words: absolute Häufigkeit")
meist_tfidf[::-1].plot.barh(ax=achsen[1], color="darkorange", title="TF-IDF: summiertes Gewicht")
plt.tight_layout()
plt.savefig(AUSGABE / "complaints_vektorisierung.png", dpi=150)
plt.show()

joblib.dump({
    "bow_vectorizer": bow_vectorizer, "bow_X": bow_X,
    "tfidf_vectorizer": tfidf_vectorizer, "tfidf_X": tfidf_X,
}, AUSGABE / "vektoren.joblib")
vergleich.to_csv(AUSGABE / "vektorisierung_top_begriffe.csv", index=False)
print("Gespeichert")
