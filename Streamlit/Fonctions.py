import streamlit as st
import string
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from nltk.corpus import stopwords
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

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
    tre = ["très","tout","plus","c'est","aussi","avon","donc","n'est","chambres"]
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
        fig = px.pie(df, values=df.iloc[:,1], names=df.iloc[:,0])
        st.plotly_chart(fig)

def nettoyage_corpus(corpus,vire_vide=True):
    output = [nettoyage_doc(doc) for doc in corpus if ((len(doc) > 0) or (vire_vide == False))]
    return output

def my_doc_2_vec(doc,trained):
    p = trained.vectors.shape[1]
    vec = np.zeros(p)
    nb = 0
    for tk in doc:
        if ((tk in trained.key_to_index.keys()) == True):
            values = trained[tk]
            vec = vec + values
            nb = nb + 1.0
    if (nb > 0.0):
        vec = vec/nb
    return vec

def my_corpora_2_vec(corpora,trained):
    docsVec = list()
    #pour chaque document du corpus nettoyé
    for doc in corpora:
        #calcul de son vecteur
        vec = my_doc_2_vec(doc,trained)
        #ajouter dans la liste
        docsVec.append(vec)
    #transformer en matrice numpy
    matVec = np.array(docsVec)
    return matVec

def my_cah_from_doc2vec(corpus,trained,seuil=1.0,nbTermes=7):
    mat = my_corpora_2_vec(corpus,trained)
    Z = linkage(mat,method='ward',metric='euclidean')
    #plt.title("CAH")
    dendrogram(Z,orientation='left',color_threshold=0)
    #st.pyplot()
    #plt.title("CAH")
    dendrogram(Z,orientation='left',color_threshold=seuil)
    st.pyplot()
    grCAH = fcluster(Z,t=seuil,criterion='distance')
    #st.write(np.unique(grCAH,return_counts=True))
    parseur = CountVectorizer(binary=True)
    corpus_string = [" ".join(doc) for doc in corpus]
    mdt = parseur.fit_transform(corpus_string).toarray()
    #st.write("Dim. matrice documents-termes = {}".format(mdt.shape))
    for num_cluster in range(np.max(grCAH)):
        st.write("")
        st.write("Numero du cluster = {}".format(num_cluster+1))
        groupe = np.where(grCAH==num_cluster+1,1,0)
        effectifs = np.unique(groupe,return_counts=True)
        st.write("Effectifs = {}".format(effectifs[1][1]))
        cooc = np.apply_along_axis(func1d=lambda x: np.sum(x*groupe),axis=0,arr=mdt)
        tmpDF = pd.DataFrame(data=cooc,columns=['freq'],index=parseur.get_feature_names_out())  
        st.dataframe(tmpDF.sort_values(by='freq',ascending=False).iloc[:nbTermes,:])
    return grCAH, mat
