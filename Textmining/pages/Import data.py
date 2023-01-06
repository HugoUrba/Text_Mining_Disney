import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import os
#import cx_Oracle
import matplotlib.pyplot as plt
import numpy as np
import oracledb


#cx_Oracle.init_oracle_client(lib_dir=r"C:/Users/ibtis/OneDrive/Bureau/me/BettyM2_SISE/textmining/instantclient-basic-windows.x64-21.8.0.0.0dbru/instantclient_21_8") #_8 pour mac 
#oracledb.init_oracle_client()
#print("yes")

#os.chdir('/Users/leo/Documents/Textmining')
os.chdir('C:/Users/ibtis/Downloads/Textmining-20230105T171826Z-001/Textmining')
st.title('Importation des données') # Titre



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


def motfreq (df, nbmots):
    height = df["freq"].head(nbmots)
    bars = df["terme"].head(nbmots)
    y_pos = np.arange(len(bars))

    # Create bars
    #plt.bar(y_pos, height)
    #st.pyplot.bar(y_pos, height)

    # Create names on the x-axis
    #st.pyplot.xticks(y_pos, bars, rotation = 90)
    fig, ax = plt.bar(y_pos, height), plt.xticks(y_pos, bars, rotation = 90)
    # Show graphic
    #return plt.show()
    st.pyplot()

if st.button('IMPORTATION'):
    

    dsnStr = oracledb.makedsn("db-etu.univ-lyon2.fr","1521","DBETU")
    #print(dsnStr)

    con = oracledb.connect(user="m132",password="m132",dsn=dsnStr)
    cursor = con.cursor()
    #print(con.version)

    #importer la table bay_club
    query_note = '''SELECT * from BAY_CLUB'''
    table_ = pd.read_sql(query_note, con=con)

    table_.to_csv('df_test_.csv', index=False)
    #print(table)
    st.dataframe(table_)
    st.subheader(type(table_))

    #cursor.close()


table_=pd.read_csv('df_test_.csv')

choix_note = st.selectbox('Choix de la note de l''avis : ', 
                                    ('Tout le fichier','1/5', '2/5', '3/5','4/5','5/5'))
choix_date = st.selectbox('Choix de la date : ', 
                                    ('Tout le fichier','Janvier', 'Février', 'Mars','Avril','Mai','Juin', 'Juillet', 'Août','Septembre','Octobre','Novembre','Décembre'))



if choix_note == 'Tout le fichier':
    st.dataframe(table_)
if choix_note == '1/5':
    df1 = st.dataframe(table_[table_.NOTE == '1/5'])
        #data = pd.read_excel("df_avis2.xlsx")
    #apres avoir filtre la note 1/5
    data = df1
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


    avis = nettoyage_doc(data["COMMENTAIRE"])

    parseur = CountVectorizer(stop_words=mots_vides)
    X = parseur.fit_transform(avis)

    mdt = X.toarray()

    freq_mots = numpy.sum(mdt,axis=0)

    index = numpy.argsort(freq_mots)
    imp = {'terme':numpy.asarray(parseur.get_feature_names())[index],'freq':freq_mots[index]}

    imp1 = pd.DataFrame.from_dict(imp, orient= 'columns')
    imp2 = imp1.sort_values(by = 'freq', ascending = False)
    nbmots = numpy.sum(mdt,axis=0)





    motfreq(imp2, 10)
    #st.dataframe(df1)
if choix_note == '2/5':
    st.dataframe(table_[table_.NOTE == '2/5'])
if choix_note == '3/5':
    st.dataframe(table_[table_.NOTE == '3/5'])
if choix_note == '4/5':
    st.dataframe(table_[table_.NOTE == '4/5'])
if choix_note == '5/5':
    st.dataframe(table_[table_.NOTE == '5/5'])


import numpy
import nltk
#modification du dossier de travail
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

#os.chdir("C:/Users/hugou/Downloads/")

import pandas as pd

