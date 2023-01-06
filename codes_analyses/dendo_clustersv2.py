# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:46:18 2023

@author: hugou
"""

#numpy
import numpy
import nltk
#modification du dossier de travail
import os
os.chdir("C:/Users/hugou/Downloads/")

import pandas as pd

data = pd.read_excel("df_avis2.xlsx")

avis = data.commentaire.to_list()

avis1 = [msg.lower() for msg in avis]

#retirer la ponctuation
import string
ponctuations = list(string.punctuation)
avis2 = ["".join([w for w in list(msg) if not w in ponctuations]) for msg in avis1]

#retirer les chiffres 
chiffres = list("0123456789")
avis3 = ["".join([w for w in list(msg) if not w in chiffres]) for msg in avis2]

from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()

#pour la tokénisation
nltk.download('punkt')
from nltk.tokenize import word_tokenize
avis4 = [word_tokenize(msg) for msg in avis3]


#liste des mots vides
nltk.download('stopwords')
from nltk.corpus import stopwords
mots_vides = stopwords.words("french")
tre = ["très"]
mots_vides = mots_vides + tre
avis5 = [[w for w in msg if not w in mots_vides] for msg in avis4]

avis6 = [[w for w in msg if len(w)>=3] for msg in avis5]


from gensim.models import Word2Vec
modele = Word2Vec(avis6,vector_size=100,window=3,min_count=1, epochs = 100)
words = modele.wv

words.similarity("hôtel","cher")

df = pd.DataFrame(words.vectors,index=words.key_to_index.keys())


#graphique dans le plan
import matplotlib.pyplot as plt
#plt.scatter(dfListe.V1,dfListe.V2,s=0.5)
#for i in range(dfListe.shape[0]):
    #plt.annotate(dfListe.index[i],(dfListe.V1[i],dfListe.V2[i]))
#plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
#pour transformation en MDT
from sklearn.feature_extraction.text import CountVectorizer

def my_doc_2_vec(doc,trained):
    #dimension de représentation
    p = trained.vectors.shape[1]
    #initialiser le vecteur
    vec = numpy.zeros(p)
    #nombre de tokens trouvés
    nb = 0
    #traitement de chaque token du document
    for tk in doc:
        #ne traiter que les tokens reconnus
        if ((tk in trained.key_to_index.keys()) == True):
            values = trained[tk]
            vec = vec + values
            nb = nb + 1.0
    #faire la moyenne des valeurs
    #uniquement si on a trouvé des tokens reconnus bien sûr
    if (nb > 0.0):
        vec = vec/nb
    #renvoyer le vecteur
    #si aucun token trouvé, on a un vecteur de valeurs nulles
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
    matVec = numpy.array(docsVec)
    return matVec

def my_cah_from_doc2vec(corpus,trained,seuil=1.0,nbTermes=7):

    #matrice doc2vec pour la représentation à 100 dim.
    #entraînée via word2vec sur les documents du corpus
    mat = my_corpora_2_vec(corpus,trained)

    #dimension
    #mat.shape

    #générer la matrice des liens
    Z = linkage(mat,method='ward',metric='euclidean')

    #affichage du dendrogramme
    plt.title("CAH")
    dendrogram(Z,orientation='left',color_threshold=0)
    plt.show()

    #affichage du dendrogramme avec le seuil
    plt.title("CAH")
    dendrogram(Z,orientation='left',color_threshold=seuil)
    plt.show()

    #découpage en 4 classes
    grCAH = fcluster(Z,t=seuil,criterion='distance')
    #print(grCAH)

    #comptage
    print(numpy.unique(grCAH,return_counts=True))

    #***************************
    #interprétation des clusters
    #***************************
    
    #parseur
    parseur = CountVectorizer(binary=True)
    
    #former corpus sous forme de liste de chaîne
    corpus_string = [" ".join(doc) for doc in corpus]
    
    #matrice MDT
    mdt = parseur.fit_transform(corpus_string).toarray()
    print("Dim. matrice documents-termes = {}".format(mdt.shape))
    
    #passer en revue les groupes
    for num_cluster in range(numpy.max(grCAH)):
        print("")
        #numéro du cluster à traiter
        print("Numero du cluster = {}".format(num_cluster+1))
        groupe = numpy.where(grCAH==num_cluster+1,1,0)
        effectifs = numpy.unique(groupe,return_counts=True)
        print("Effectifs = {}".format(effectifs[1][1]))
        #calcul de co-occurence
        cooc = numpy.apply_along_axis(func1d=lambda x: numpy.sum(x*groupe),axis=0,arr=mdt)
        #print(cooc)
        #création d'un data frame intermédiaire
        tmpDF = pd.DataFrame(data=cooc,columns=['freq'],index=parseur.get_feature_names_out())    
        #affichage des "nbTermes" termes les plus fréquents
        print(tmpDF.sort_values(by='freq',ascending=False).iloc[:nbTermes,:])
        
    #renvoyer l'indicateur d'appartenance aux groupes
    return grCAH, mat

g1,mat1 = my_cah_from_doc2vec(avis6,words,seuil=10)
