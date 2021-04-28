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

import pycaret.classification as pyc
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


st.header("Explainable-ml-app")

INDEX = get_data("index")

st.write("Datasets")
st.dataframe(INDEX, height=3000)

AUSWAHL = st.selectbox(label = "Dataset", options = [""] + INDEX.Dataset.tolist())

if AUSWAHL == "":
    st.stop()

else:
    DATENSATZ = get_data(AUSWAHL)

    st.write(DATENSATZ)
    st.balloons()

