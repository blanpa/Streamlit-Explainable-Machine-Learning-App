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
    return DF


st.sidebar.header("Explainable-ml-app")


def main():
    TYPE = st.sidebar.selectbox(label = "Type", options = ["Classification", "Regression"])
    st.header(TYPE)

    DATA = st.sidebar.selectbox(label = "Data", options = ["Dummydata", "CSV-File"])

    if DATA == "Dummydata":
        INDEX = get_data("index")
        AUSWAHL = st.sidebar.selectbox(label = "Dataset", options = [""] + INDEX.Dataset.tolist())

        if AUSWAHL == "":
            st.dataframe(INDEX, height=3000)

        else:
            DATENSATZ = get_data(AUSWAHL)
            st.write(DATENSATZ.head(100))
            

    elif DATA == "CSV-File":
        FILE = st.sidebar.file_uploader("Choose a file")
        try:
            CSV_FILE = pd.read_csv(FILE)
            st.write(CSV_FILE.head())
        except:
            st.stop()


if __name__ == "__main__":
    main()
