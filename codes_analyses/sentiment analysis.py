# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 17:22:32 2023

@author: hugou
"""

import os
import pandas as pd 
import re
from spacy.lang.fr.stop_words import STOP_WORDS
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer
import plotly.express as px

os.chdir("C:/Users/hugou/Downloads/")

data = pd.read_csv("df_aviscsv.csv", encoding='latin=1', sep = ";")

data["commentaire"]= data["commentaire"].str.lower()

#nettoyage
AComment=[]
for comment in data["commentaire"].apply(str):
    Word_Tok = []
    for word in  re.sub("\W"," ",comment ).split():
        Word_Tok.append(word)
    AComment.append(Word_Tok)

data["Word_Tok"]= AComment

stop_words=set(STOP_WORDS)

deselect_stop_words = ['n\'', 'ne','pas','plus','personne','aucun','ni','aucune','rien']
for w in deselect_stop_words:
    if w in stop_words:
        stop_words.remove(w)
    else:
        continue
    
AllfilteredComment=[]
for comment in data["Word_Tok"]:
    filteredComment = [w for w in comment if not ((w in stop_words) or (len(w) == 1))]
    AllfilteredComment.append(' '.join(filteredComment))
    
data["CommentAferPreproc"]=AllfilteredComment

#avoir les sentiments des commentaires
tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())

senti_list = []
for i in data["CommentAferPreproc"]:
    vs = tb(i).sentiment[0]
    if (vs > 0):
        senti_list.append('Positive')
    elif (vs < 0):
        senti_list.append('Negative')
    else:
        senti_list.append('Neutral')
        
data["sentiment"]=senti_list

senti_base_df = data.drop(["CommentAferPreproc", "Word_Tok"], axis = 1)

Number_sentiment = senti_base_df.groupby(["sentiment"]).count().reset_index().reset_index(drop=True)
Number_sentiment = Number_sentiment.iloc[:,:2]
Number_sentiment.rename(columns={'note':'nombre de commentaire'}, inplace=True)

def sentiPie (df):
    fig = px.pie(df, values=df.iloc[:,1], names=df.iloc[:,0], title = "Analyse des sentiments")
    return fig.show()

print(sentiPie(Number_sentiment))
import iplot
iplot(sentiPie(Number_sentiment))
