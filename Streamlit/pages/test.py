import os
import streamlit as st
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
from PIL import Image
from wordcloud import WordCloud
import numpy as np
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize 
from spacy.lang.fr.stop_words import STOP_WORDS
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer
import re
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from gensim.models import Word2Vec
import Fonctions as ft


data = pd.read_csv("/Users/leo/Downloads/Textmining/disney.csv",encoding='latin-1',sep = ";")

