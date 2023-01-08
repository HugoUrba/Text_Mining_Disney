import streamlit as st
import pandas as pd
import numpy 
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from spacy.lang.fr.stop_words import STOP_WORDS
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer
import re
import Fonctions as ft

st.markdown("The dashboard will help a researcher to get to know \
more about the given datasets and it's output")

st.sidebar.title("Selection du graphique : ")
st.sidebar.markdown("Selection graphiques/tracés :")
selected_status = st.sidebar.selectbox('Listing des hotels :',
                                       options = ['NewPort Bay Club', 
                                                  'Hotel Cheyenne', 'Hotel Santa Fe', 
                                                  'Hotel New-York - The Art of Marvel','Davy Crockett Ranch','Sequoia Lodge'])
if selected_status == 'NewPort Bay Club':
    df = pd.read_csv("/Users/leo/Documents/Streamlit/disney.csv",encoding='latin-1',sep = ";")
    st.title("Analyse Text mining Hôtels")
    
    choix_note = st.sidebar.multiselect('Choix de la note ', ['1/5', '2/5','3/5' ,'4/5', '5/5'])

    if len(choix_note) > 0:
        df = df[df['note'].isin(choix_note)]
        st.dataframe(df)
    else:
        st.dataframe(df)

    choix_date = st.sidebar.selectbox('Choix de la date : ', 
                                    ('Tout le fichier','Janvier', 'Février', 'Mars','Avril','Mai','Juin', 'Juillet', 'Août','Septembre','Octobre','Novembre','Décembre'))

    mots_vides = stopwords.words("french")
    tre = ["très"]
    mots_vides = mots_vides + tre

    avis = ft.nettoyage_doc(df["commentaire"])

    parseur = CountVectorizer(stop_words=mots_vides)
    X = parseur.fit_transform(avis)
    mdt = X.toarray()
    freq_mots = numpy.sum(mdt,axis=0)

    index = numpy.argsort(freq_mots)
    imp = {'terme':numpy.asarray(parseur.get_feature_names())[index],'freq':freq_mots[index]}
    imp1 = pd.DataFrame.from_dict(imp, orient= 'columns')
    imp2 = imp1.sort_values(by = 'freq', ascending = False)
    nbmots = numpy.sum(mdt,axis=0)

    ft.motfreq(imp2, 10)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    #Graphique circulaire sur la satisfaction client
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

    #Obtention les sentiments des commentaires
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

    senti_base_df = df.drop(["CommentAferPreproc", "Word_Tok"], axis = 1)

    Number_sentiment = senti_base_df.groupby(["sentiment"]).count().reset_index().reset_index(drop=True)
    Number_sentiment = Number_sentiment.iloc[:,:2]
    Number_sentiment.rename(columns={'note':'nombre de commentaire'}, inplace=True)

    ft.sentiPie(Number_sentiment)

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
