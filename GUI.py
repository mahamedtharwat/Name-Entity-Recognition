import streamlit as st
from joblib import dump, load
from sklearn.neural_network import MLPClassifier
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle

import spacy 
model_loaded = load("mlp_Classifier.joblib")
st.sidebar.header("Name Entity Recognition Task: ")

text = st.sidebar.text_area("Enter Your Text:")

# def preprocess_text(text):
#     sentences = tokenize.sent_tokenize(text)
#     words = [tokenize.word_tokenize(sent) for sent in sentences]
#     word_tags = [nltk.pos_tag(sent) for sent in words]
    
#     word=[]
#     pos=[]
#     for idx in word_tags:
#         for idj in idx:
#             word.append(idj[0])

#     for idx in word_tags:
#         for idj in idx:
#             pos.append(idj[1])

#     frame={'Word':word,
#         'Pos_Tag':pos}
#     frame = pd.DataFrame(frame)
#     X = frame[["Word", "Pos_Tag"]]
    
#     with open('word_vectorizer.pkl', 'rb') as f:
#         word_vectorizer = pickle.load(f)

#     with open('pos_vectorizer.pkl', 'rb') as f:
#         pos_vectorizer = pickle.load(f)
        
#     X_words = word_vectorizer.transform(X["Word"])

#     X_pos = pos_vectorizer.transform(X["Pos_Tag"])
#     X_combined = pd.concat(
#         [pd.DataFrame(X_words.toarray()), pd.DataFrame(X_pos.toarray())], axis=1
#     )
    
#     return [ X_combined , frame.Word] 
nlp = spacy.load("en_core_web_md")
if text:
#    Text_Features , word = preprocess_text(text)
#    word = list(word)
#    text_pred = model_loaded.predict(Text_Features)
      NER = nlp(text)

   
      st.header("The Result:")
      for ent in NER.ents:
            st.write(f"{ent.text} : {ent.label_}")
#    for idx in range(len(text_pred)):
#       if text_pred[idx] == "O":
#             continue
#             # print (f"{word[idx]}: O") 
#       elif text_pred[idx] == "PERSON":
#             st.write(f"{word[idx]}: PERSON")
#       elif text_pred[idx] == "DATE":
#             st.write(f"{word[idx]}: DATE")  
#       elif text_pred[idx] == "P-NUMBER":
#             st.write(f"{word[idx]}: P-NUMBER")  
#       elif text_pred[idx] == "CARDINAL":
#             st.write(f"{word[idx]}: CARDINAL")
#       elif text_pred[idx] == "COMPANY":
#             st.write(f"{word[idx]}: COMPANY")
#       elif text_pred[idx] == "ORG":
#             st.write(f"{word[idx]}: ORG")
#       elif text_pred[idx] == "COLOR":
#             st.write(f"{word[idx]}: COLOR")
#       elif text_pred[idx] == "Org":
#             st.write(f"{word[idx]}: Org")
#       elif text_pred[idx] == "CITY":
#             st.write(f"{word[idx]}: CITY")
#       elif text_pred[idx] == "COUNTRY":
#             st.write(f"{word[idx]}: COUNTRY")
      
   


