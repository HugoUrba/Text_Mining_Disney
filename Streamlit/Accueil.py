import os
import streamlit as st
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
from PIL import Image
from wordcloud import WordCloud
import numpy as np
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize 
from spacy.lang.fr.stop_words import STOP_WORDS
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer
import re
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from gensim.models import Word2Vec
import Fonctions as ft


#data = pd.read_csv("/Users/leo/Downloads/Textmining/disney.csv",encoding='latin-1',sep = ";")
df = pd.read_csv("/Users/leo/Documents/Streamlit/dataDisneyavis_final.csv",encoding='utf-8',sep = ",")

st.set_page_config(page_title="Image Example", page_icon=":camera:", layout="wide")
image_file = "/Users/leo/Documents/Streamlit/chateau.jpeg"
image = Image.open(image_file)
st.image(image, use_column_width=True)

st.title('Application Google Reviews - Text mining Disneyland') # Titre
st.markdown("Cette application crée avec streamlit permet de visualiser nos analyses textuelles obtenus par scrapping sur les avis Google. La navigation sur l'application est intéractive et guidée. La première page d'accueil constitue les éléments fondamentaux et globaux de nos données avec quelques analyses descriptives. Vous avez la possibilité de visualiser des analyses plus précise grâce aux onglets Hotels et Parcs. Vous y trouverez des graphiques intéractifs avec la possibilité de visualiser les résutlats selon différents critères : Le choix de la note et de la date du séjour. Nous avons également mis en place la possibilité d'importer des nouvelles données grâce au dernier onglet intitulé Import Data qui permet en un simple clic l'intégration de nouvelles données. Bonne visite.")
st.subheader('Google Reviews Dataset ')

# Données 
#df = pd.read_csv("/Users/leo/Documents/Streamlit/disney.csv",encoding='latin-1',sep = ";")
st.dataframe(df)

# Graphique histograme des notes moyennes 
st.title('Moyenne des notes attribuées par les clients')
st.bar_chart(df.note.value_counts())

# Graphique circulaire dynamique
st.title("Mesure de la satisfaction par structure")
df["commentaire"]= df["commentaire"].str.lower()
#nettoyage
AComment=[]
for comment in df["commentaire"].apply(str):
    Word_Tok = []
    for word in  re.sub("\W"," ",comment ).split():
        Word_Tok.append(word)
    AComment.append(Word_Tok)
df["Word_Tok"]= AComment
stop_words=set(STOP_WORDS)
deselect_stop_words = ['n\'', 'ne','pas','plus','personne','aucun','ni','aucune','rien']
for w in deselect_stop_words:
    if w in stop_words:
        stop_words.remove(w)
    else:
        continue 
AllfilteredComment=[]
for comment in df["Word_Tok"]:
    filteredComment = [w for w in comment if not ((w in stop_words) or (len(w) == 1))]
    AllfilteredComment.append(' '.join(filteredComment))
df["CommentAferPreproc"]=AllfilteredComment
tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
senti_list = []
for i in df["CommentAferPreproc"]:
    vs = tb(i).sentiment[0]
    if (vs > 0):
        senti_list.append('Positive')
    elif (vs < 0):
        senti_list.append('Negative')
    else:
        senti_list.append('Neutral')
df["sentiment"]=senti_list

fig = px.sunburst(df, path=['type de service', 'service', df['sentiment']])
st.plotly_chart(fig)

#Graphique nuage des mots 
st.title("Nuage de mots")
avis = ft.nettoyage_doc(df["commentaire"])
ft.WC(avis,mask= np.array(Image.open("/Users/leo/Documents/Streamlit/cloud.png")))
st.set_option('deprecation.showPyplotGlobalUse', False)
