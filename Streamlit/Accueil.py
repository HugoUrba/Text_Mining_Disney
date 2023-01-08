import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import Fonctions as ft

st.title('Projet Text Mining') # Titre
st.subheader('Application réalisée par M2 SISE') # Sous Titre
st.markdown("Cette application crée avec streamlit permet de visualiser les analyses et les données brutes obtenus par scrapping sur les avis Google. Explication rapide des onglets, des aperçus, des foncitonnalités de l'appli etc...")
st.subheader('Visualisation des données : ')

# Données 
df = pd.read_csv("/Users/leo/Documents/Streamlit/disney.csv",encoding='latin-1',sep = ";")
st.dataframe(df)

# Graphique histograme des notes moyennes 
st.subheader('Moyenne des notes du séjour attribuées par les clients')
st.bar_chart(df.note.value_counts())

#Graphique nuage des mots 
avis = ft.nettoyage_doc(df["commentaire"])
ft.WC(avis,mask= np.array(Image.open("/Users/leo/Documents/Streamlit/cloud.png")))
st.set_option('deprecation.showPyplotGlobalUse', False)

# AUTRES GRAPHIQUES
#st.subheader('Date des visites des clients')
#st.dataframe(df['date de visite'].value_counts())
#st.line_chart(df['date de visite'].value_counts())