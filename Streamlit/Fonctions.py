import streamlit as st
import string
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px

def nettoyage_doc(doc_param):
        #liste des ponctuations
    ponctuations = list(string.punctuation)
    #liste des chiffres
    chiffres = list("0123456789")
    #outil pour procéder à la lemmatisation - attention à charger le cas échéant
    #nltk.download('wordnet')
    lem = WordNetLemmatizer()
    #pour la tokénisation
    from nltk.tokenize import word_tokenize
    #liste des mots vides
    mots_vides = stopwords.words("french")
    tre = ["très"]
    mots_vides = mots_vides + tre

    #passage en minuscule
    doc = [msg.lower() for msg in doc_param]
    #retrait des ponctuations
    doc = "".join([w for w in list(doc) if not w in ponctuations])
    #retirer les chiffres
    doc = "".join([w for w in list(doc) if not w in chiffres])
    #transformer le document en liste de termes par tokénisation
    doc = word_tokenize(doc)
    #lematisation de chaque terme
    doc = [lem.lemmatize(terme) for terme in doc]
    #retirer les stopwords
    doc = [w for w in doc if not w in mots_vides]
    #retirer les termes de moins de 3 caractères
    doc = [w for w in doc if len(w)>=3]
    #fin
    return doc

def WC(com, mask):
    wordcloud = WordCloud(background_color="white", width=800, height=600, max_words=100, mask=mask).generate(str(com))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot()

def motfreq(df, nbmots):
        height = df["freq"].head(nbmots)
        bars = df["terme"].head(nbmots)

        fig = go.Figure(data=[go.Bar(x=bars, y=height)])
        fig.update_layout(xaxis_tickangle=-90)
        st.plotly_chart(fig)

def sentiPie(df):
        fig = px.pie(df, values=df.iloc[:,1], names=df.iloc[:,0], title="Analyse des sentiments")
        st.plotly_chart(fig)