import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.title("Tableau de bord interactif")
st.markdown("The dashboard will help a researcher to get to know \
more about the given datasets and it's output")
st.sidebar.title("Selection du graphique : ")
st.sidebar.markdown("Selection graphiques/tracés :")

data = pd.read_csv("demo_data_set.csv",encoding='latin-1',sep = ",")

chart_visual = st.sidebar.selectbox('Type du graphique : ', 
                                    ('Line Chart', 'Bar Chart', 'Bubble Chart'))

selected_status = st.sidebar.selectbox('Listing des Parcs :',
                                       options = ['Disneyland Paris','Parc Walt Disney Studios'])

fig = go.Figure()

if chart_visual == 'Line Chart':
    if selected_status == 'Disneyland Paris':
        fig.add_trace(go.Scatter(x = data.Country, y = data.formerly_smoked,
                                 mode = 'lines',
                                 name = 'Formerly_Smoked'))
    if selected_status == 'Parc Walt Disney Studios':
        fig.add_trace(go.Scatter(x = data.Country, y = data.Smokes,
                                 mode = 'lines', name = 'Smoked'))
  
elif chart_visual == 'Bar Chart':
    if selected_status == 'Disneyland Paris':
        fig.add_trace(go.Bar(x=data.Country, y=data.formerly_smoked,
                             name='Formerly_Smoked'))
    if selected_status == 'Parc Walt Disney Studios':
        fig.add_trace(go.Bar(x=data.Country, y=data.Smokes,
                             name='Smoked'))
  
elif chart_visual == 'Bubble Chart':
    if selected_status == 'Disneyland Paris':
        fig.add_trace(go.Scatter(x=data.Country, 
                                 y=data.formerly_smoked,
                                 mode='markers',
                                 marker_size=[40, 60, 80, 60, 40, 50],
                                 name='Formerly_Smoked'))
          
    if selected_status == 'Parc Walt Disney Studios':
        fig.add_trace(go.Scatter(x=data.Country, y=data.Smokes,
                                 mode='markers', 
                                 marker_size=[40, 60, 80, 60, 40, 50],
                                 name='Smoked'))
    
st.plotly_chart(fig, use_container_width=True)

