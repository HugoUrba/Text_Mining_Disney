import streamlit as st
import pandas as pd

df = pd.read_csv("/Users/leo/Downloads/Textmining/disney.csv",encoding='latin-1',sep = ";")


choix_note = st.multiselect('Choix de la note ', ['1/5', '2/5','3/5' ,'4/5', '5/5'],['1/5', '2/5','3/5' ,'4/5', '5/5'])

df = df[df['note'].isin(choix_note)]

# Afficher le DataFrame filtré
st.dataframe(df)