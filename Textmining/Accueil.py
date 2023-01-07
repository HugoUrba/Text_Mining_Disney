import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

os.chdir('/Users/leo/Documents/Textmining')

st.title('Projet Text Mining') # Titre
st.subheader('Application réalisée par M2 SISE') # Sous Titre
st.markdown("Cette application crée avec streamlit permet de visualiser les analyses et les données brutes obtenus par scrapping sur les avis Google. Explication rapide des onglets, des aperçus, des foncitonnalités de l'appli etc...")
st.subheader('Visualisation des données : ')

# Données 
df = pd.read_csv("/Users/leo/Downloads/Textmining/disney.csv",encoding='latin-1',sep = ";")
st.dataframe(df)

st.subheader('Moyenne des notes du séjour attribuées par les clients')
st.bar_chart(df.note.value_counts())

st.subheader('Date des visites des clients')
st.dataframe(df['date de visite'].value_counts())
st.line_chart(df['date de visite'].value_counts())