import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy 
import nltk
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize 


st.title("Tableau de bord interactif")
st.markdown("The dashboard will help a researcher to get to know \
more about the given datasets and it's output")
st.sidebar.title("Selection du graphique : ")
st.sidebar.markdown("Selection graphiques/tracés :")

selected_status = st.sidebar.selectbox('Listing des hotels :',
                                       options = ['NewPort Bay Club', 
                                                  'Hotel Cheyenne', 'Hotel Santa Fe', 
                                                  'Hotel New-York - The Art of Marvel','Davy Crockett Ranch','Sequoia Lodge'])


if selected_status == 'NewPort Bay Club':
    df = pd.read_csv("/Users/leo/Downloads/Textmining/disney.csv",encoding='latin-1',sep = ";")

    choix_note = st.sidebar.multiselect('Choix de la note ', ['1/5', '2/5','3/5' ,'4/5', '5/5'])

    if len(choix_note) > 0:
        df = df[df['note'].isin(choix_note)]
        st.dataframe(df)
    else:
        st.dataframe(df)

    choix_date = st.sidebar.selectbox('Choix de la date : ', 
                                    ('Tout le fichier','Janvier', 'Février', 'Mars','Avril','Mai','Juin', 'Juillet', 'Août','Septembre','Octobre','Novembre','Décembre'))

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

    #********************************
    #fonction pour nettoyage document (chaîne de caractères)
    #le document revient sous la forme d'une liste de tokens
    #********************************
    def nettoyage_doc(doc_param):
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

    avis = nettoyage_doc(df["commentaire"])

    parseur = CountVectorizer(stop_words=mots_vides)
    X = parseur.fit_transform(avis)

    mdt = X.toarray()

    freq_mots = numpy.sum(mdt,axis=0)

    index = numpy.argsort(freq_mots)
    imp = {'terme':numpy.asarray(parseur.get_feature_names())[index],'freq':freq_mots[index]}
    
    imp1 = pd.DataFrame.from_dict(imp, orient= 'columns')
    imp2 = imp1.sort_values(by = 'freq', ascending = False)
    nbmots = numpy.sum(mdt,axis=0)

    import matplotlib.pyplot as plt
    import numpy as np

    def motfreq (df, nbmots):
        height = df["freq"].head(nbmots)
        bars = df["terme"].head(nbmots)
        y_pos = np.arange(len(bars))

        # Create bars
        plt.bar(y_pos, height)

        # Create names on the x-axis
        plt.xticks(y_pos, bars, rotation = 90)

        # Show graphic
        return st.pyplot()

    motfreq(imp2, 10)
    st.set_option('deprecation.showPyplotGlobalUse', False)


if selected_status == 'Hotel Cheyenne':
    df

if selected_status == 'Hotel Santa Fe':
    df

if selected_status == 'Hotel New-York - The Art of Marvel': 
    df

if selected_status == 'Davy Crockett Ranch':
    df

if selected_status == 'Sequoia Lodge':
    df
