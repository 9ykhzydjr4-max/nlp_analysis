from pathlib import Path
from matplotlib import cm
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
import pyLDAvis.lda_model
import warnings   

PROJECT = Path(__file__).parent if "__file__" in globals() else Path.cwd()
AUSGABE = PROJECT / "output"

dataframe = pd.read_csv(AUSGABE / "complaints_cleaned.csv", encoding="utf-8")
dataframe = dataframe.dropna(subset=["cleaned_narrative"])

vektoren = joblib.load(AUSGABE / "vektoren.joblib")
bow_vektor = vektoren["bow_vectorizer"]
bow_X = vektoren["bow_X"]
tfidf_vektor = vektoren["tfidf_vectorizer"]
tfidf_X = vektoren["tfidf_X"]

woerter_bow = bow_vektor.get_feature_names_out()
woerter_tfidf = tfidf_vektor.get_feature_names_out()
print("Dokumente:", len(dataframe), "| BoW:", bow_X.shape, "| TF-IDF:", tfidf_X.shape)

def meist_woerter(modell, woerter, n=15):
    themen = []
    for gewichte in modell.components_:
        beste = gewichte.argsort()[::-1][:n]
        top_woerter = [woerter[i] for i in beste]
        themen.append(top_woerter)
    return themen

def themen_anzeigen(themen):
    for i, top_woerter  in enumerate(themen):
        print(f"Thema {i+1}: {', '.join(top_woerter)}")

def lsa_modell(k):
    lsa = TruncatedSVD(n_components=k, random_state=42)
    return lsa

def lda_modell(k):
    lda = LatentDirichletAllocation(n_components=k, learning_method="online",
                                    batch_size=2048, max_iter=10,
                                    random_state=42, n_jobs=-1)
    return lda

# Kohärenz aufbereiten
stichprobe = dataframe.sample(n=15000, random_state=42)
# stichprobe = dataframe.sample(n=10, random_state=42)

texte_token =[]
for text in stichprobe["cleaned_narrative"]:
    woerter = text.split()
    paare = [f"{a} {b}" for a, b in zip(woerter[:-1], woerter[1:])]
    texte_token.append(woerter + paare)

woerterbuch = Dictionary(texte_token)
bow_X_sp = bow_vektor.transform(stichprobe["cleaned_narrative"])
tfidf_X_sp = tfidf_vektor.transform(stichprobe["cleaned_narrative"])

def kohaerenz(themen, texte_token):
    modell = CoherenceModel(topics=themen, texts=texte_token, dictionary=woerterbuch, coherence="c_v", processes=1)
    # processes=1, da sonst Fehler auftreten können, wenn zu viele Prozesse gestartet werden
    return modell.get_coherence()

# Kohärenz für div. k berechnen
ergebnisse = []
for k in [4, 6, 8, 10, 12]:
    lsa_k = lsa_modell(k)
    lsa_k.fit(bow_X_sp)
    themen_lsa = meist_woerter(lsa_k, woerter_bow)
    koh_lsa = kohaerenz(themen_lsa, texte_token)

    lda_k = lda_modell(k)
    lda_k.fit(tfidf_X_sp)
    themen_lda = meist_woerter(lda_k, woerter_tfidf)
    koh_lda = kohaerenz(themen_lda, texte_token)

    ergebnisse.append({"k": k, "Kohärenz LSA": round(koh_lsa, 3), "Kohärenz LDA": round(koh_lda, 3)})
    print(f"k={k:2d} | Kohärenz LSA: {koh_lsa:.3f} | Kohärenz LDA: {koh_lda:.3f}")

koherenz_tabelle = pd.DataFrame(ergebnisse)
koherenz_tabelle.to_csv(AUSGABE / "themen_kohaerenz.csv", index=False)

koherenz_tabelle.plot(x="k", y=["Kohärenz LSA", "Kohärenz LDA"], marker="o", title="Kohärenz der Themenmodelle", xlabel="Anzahl der Themen (k)", ylabel="Kohärenz")
plt.ylabel("Kohärenz")
plt.grid(alpha=0.3)
plt.savefig(AUSGABE / "themen_kohaerenz.png", dpi=150)
# plt.show()
plt.close()

# LSA
# k = 8
TOLERANZ = 0.01

beste_kohaerenz = koherenz_tabelle["Kohärenz LDA"].max()
kandidaten = koherenz_tabelle[koherenz_tabelle["Kohärenz LDA"] >= beste_kohaerenz- TOLERANZ]
k = int(kandidaten["k"].max())

print(f"Beste Kohärenz LDA: {beste_kohaerenz:.3f}")
print(f"Gleichwertige k (Tolrenz {TOLERANZ}): {kandidaten['k'].tolist()}")
print(f"Gewählt: k = {k}")

lsa = lsa_modell(k)
lsa_dokumente = lsa.fit_transform(tfidf_X)
themen_lsa = meist_woerter(lsa, woerter_tfidf)
themen_anzeigen(themen_lsa)
print(f"Erklärte Varianz TF-IDF: {lsa.explained_variance_ratio_.sum():.2%}")

# LDA
lda = lda_modell(k)
lda_dokumente = lda.fit_transform(bow_X)
themen_lda = meist_woerter(lda, woerter_bow)
themen_anzeigen(themen_lda)
# print(f"Erklärte Varianz BoW: {lda.explained_variance_ratio_.sum():.2%}")
print(f"Perplexity LDA: {lda.perplexity(bow_X):.1f}")


# Aufteilung Thema zu Dokument
dataframe["LSA_Thema"] = lsa_dokumente.argmax(axis=1)
dataframe["LDA_Thema"] = lda_dokumente.argmax(axis=1)

print("Beispiel für Themenzuordnung:")
print(dataframe["LDA_Thema"].value_counts(normalize=True) * 100)
print("Beispiel für Themenzuordnung:")
print(dataframe["LSA_Thema"].value_counts(normalize=True) * 100)

# Plausibilitätsprüfung mit Produktkategorie
pd.set_option("display.width", 200)
abgleich = pd.crosstab(dataframe["LDA_Thema"], dataframe["product"], normalize="index").round(2)
print("Abgleich LDA-Thema mit Produktkategorie:")
print(abgleich)
abgleich.to_csv(AUSGABE / "themen_produkt_abgleich.csv", index=True, encoding="utf-8")

# Grafische Darstellung der Themenzuordnung
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold")
grafik = pyLDAvis.lda_model.prepare(lda, bow_X, bow_vektor, mds="mmds", sort_topics=False)
pyLDAvis.save_html(grafik, str(AUSGABE / "lda_grafik.html"))
print(f"LDA-Grafik gespeichert: {AUSGABE / 'lda_grafik.html'}")

# Themen speichern
pd.DataFrame({"Thema": range(k), "Top_Woerter": [", ".join(t) for t in themen_lda]}).to_csv(AUSGABE / "themen_lda.csv", index=False, encoding="utf-8")
pd.DataFrame({"Thema": range(k), "Top_Woerter": [", ".join(t) for t in themen_lsa]}).to_csv(AUSGABE / "themen_lsa.csv", index=False, encoding="utf-8")
print("Themen gespeichert")

