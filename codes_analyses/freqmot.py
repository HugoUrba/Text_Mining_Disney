# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 19:36:39 2023

@author: hugou
"""

import numpy
import nltk
#modification du dossier de travail
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize

os.chdir("C:/Users/hugou/Downloads/")

import pandas as pd

data = pd.read_csv("df_aviscsv.csv", encoding='latin=1', sep = ";")


#liste des ponctuations
ponctuations = list(string.punctuation)

#liste des chiffres
chiffres = list("0123456789")

#outil pour procéder à la lemmatisation - attention à charger le cas échéant
#nltk.download('wordnet')
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
lem = FrenchLefffLemmatizer()

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
    doc = word_tokenize(doc, language="french")
    #lematisation de chaque terme
    doc = [lem.lemmatize(terme) for terme in doc]
    #retirer les stopwords
    doc = [w for w in doc if not w in mots_vides]
    #retirer les termes de moins de 3 caractères
    doc = [w for w in doc if len(w)>=3]
    #fin
    return doc

avis = nettoyage_doc(data["commentaire"])

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
    return plt.show()

motfreq(imp2, 10)
