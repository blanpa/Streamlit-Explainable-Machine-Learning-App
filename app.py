import logging, os

#Streamlit
import streamlit as st
from streamlit import caching
import streamlit.components.v1 as components
import SessionState
import codecs

#Standart
import pandas as pd
import numpy as np
import os 
import time
import json
import base64

# sql
import sqlalchemy as sqla

# ml
from sklearn.metrics import classification_report, confusion_matrix
import imblearn

import pycaret.classification as pcc
import pycaret.regression as pcr
from pycaret.datasets import get_data

# plotting
import matplotlib.pyplot as plt
# from dtreeviz.trees import *

# Pandas Profiling 
from pandas_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report

# Interpretation
import dalex as dx

# #Redis
# import redis
# from redis_namespace import StrictRedis

def upload_data(file):
    DF = pd.read_csv(file, encoding='utf-8')
  
  
################################################
# Init
################################################
st.set_page_config(
    page_title="Explainable-ml-app",
    #page_icon = "",
    layout="wide",
    initial_sidebar_state="expanded")

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)


st.sidebar.header("Explainable-ml-app")
BREITE = st.sidebar.slider(label ="Display-size", min_value = 300, max_value=3000, value = 1400, step= 100)

st.markdown(
    f"""
    <style>
    .reportview-container .main .block-container{{
        max-width: {BREITE}px;
    }}
    </style>
    """,unsafe_allow_html=True)





def main():
    TYPE = st.sidebar.selectbox(label = "Type", options = [ "", "Classification", "Regression"])
    st.header(TYPE)

    if TYPE == "":
        st.stop()

    DATA = st.sidebar.selectbox(label = "Data", options = ["Dummydata", "CSV-File"])

    if DATA == "Dummydata":
        INDEX = get_data("index")
        AUSWAHL = st.sidebar.selectbox(label = "Dataset", options = [""] + INDEX.Dataset.tolist())

        if AUSWAHL == "":
            st.dataframe(INDEX, height=3000)
            st.stop()

        else:
            DATENSATZ = get_data(AUSWAHL)
            

    elif DATA == "CSV-File":
        FILE = st.sidebar.file_uploader("Choose a file")
        try:
            DATENSATZ = pd.read_csv(FILE)
        except:
            st.stop()

    st.write("Dataset")
    st.write(DATENSATZ.head(100).head(100).style.highlight_null(null_color='red'))

    PANDASPR = st.checkbox(label ="show data report", value = False)

    if PANDASPR == True:
        with st.beta_expander("Data Report"):
            import streamlit.components.v1 as components
            components.html(html=ProfileReport(df = DATENSATZ, minimal = False).to_html(), scrolling = True, height = 1000)


    st.write("_______")
    st.header("Target")

    TARGET = st.selectbox(
        label = "choose target", 
        options = [" "] + DATENSATZ.columns.tolist(), 
        index = (0))

    col1, _ = st.beta_columns(2)
    try:
        with col1:
            st.bar_chart(DATENSATZ[TARGET].value_counts())
        import numpy as np
        neg, pos = np.bincount(DATENSATZ[TARGET])
        total = neg + pos
        st.text('Instanzen: \n Gesamt: {}\n Positiv: {} ({:.2f}% von allen)\n'.format(total, pos, 100 * pos / total))
    except:
        pass


    if TYPE == "Classification":
        if st.button("Train Model"):
            SETUPCLASSIFICATION = pcc.setup(data = DATENSATZ, target = TARGET, silent = True, html = False)
            BEST = pcc.compare_models()
        
        try:
            st.write(pcc.get_config("display_container")[1])
            st.write(BEST)
        except:
            pass

    elif TYPE == "Regression":
        if st.button("Train Model"):
            SETUPREGRESSION = pcr.setup(data = DATENSATZ, target = TARGET, silent = True, html = False)
            BEST = pcr.compare_models()

        try:
            st.write(pcr.get_config("display_container")[1])
            st.write(BEST)
        except:
            pass
        

if __name__ == "__main__":
    main()
