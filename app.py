import logging, os
from pycaret.internal.preprocess import DFS_Classic

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
    page_icon = "media/Icon.ico",
    layout="wide",
    initial_sidebar_state="expanded")

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)


def sidebar_render_svg(svg):
    """Renders the given svg string."""
    f = open(svg,"r")
    lines = f.readlines()
    line_string=''.join(lines)
    b64 = base64.b64encode(line_string.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.sidebar.write(html, unsafe_allow_html=True)

sidebar_render_svg("media/logo.svg")

with st.sidebar.beta_expander("Settings"):
    BREITE = st.slider(label ="Display-size", min_value = 300, max_value=3000, value = 1400, step= 100)

st.markdown(
    f"""
    <style>
    .reportview-container .main .block-container{{
        max-width: {BREITE}px;
    }}
    </style>
    """,unsafe_allow_html=True)

@st.cache
def trained_classification_model():
    BEST = pcc.compare_models()
    return BEST

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
    st.header("Train ML Model")

    TARGET = st.selectbox(
        label = "choose target", 
        options = [" "] + DATENSATZ.columns.tolist(), 
        index = (0))
    
    if TARGET == " ":
        st.stop()

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

    with st.beta_expander("Settings"):
        pass

    if TYPE == "Classification":
        MODELS = [" ", "lr", "knn", "nb", "dt", "svm", "rbfsvm", "gpc", "mlp", "ridge", "rf", "qda", "ada", "gbc", "lda", "et", "xgboost", "lightgbm", "catboost"]
        MODEL_LIST = st.multiselect(
            label = "models", 
            options = MODELS, 
            default = ["lr", "knn", "nb", "dt", "svm", "rbfsvm", "gpc", "mlp", "ridge", "rf", "qda", "ada", "gbc", "lda", "et", "xgboost", "lightgbm", "catboost"] )

        if st.button("Train Model"):
            SETUPCLASSIFICATION = pcc.setup(data = DATENSATZ, target = TARGET, silent = True, html = False)
            trained_classification_model()

        with st.beta_expander("Scores"):
            try:
                st.write(pcc.get_config("display_container")[1])
                st.write(trained_classification_model())
            except:
                pass

            PLOTS = st.multiselect(
                    label = "AUSWAHL_PLOTS", 
                    options = ["auc", "threshold", "pr", "confusion_matrix", "error", "class_report", "boundary", "rfe", "learning", "manifold", "calibration", "vc", "dimension", "feature", "lift", "gain", "tree"], 
                    default = ["confusion_matrix", "class_report", "auc"]
                    )

            # Plots 
            col1, col2 = st.beta_columns(2)
            with col1:
                st.markdown("### Ergebnisse TRAININGS-Datensatz")
                for i in PLOTS:
                    try:
                        st.markdown(f"#### {i}")
                        pcc.plot_model(trained_classification_model(), i ,use_train_data = True, display_format="streamlit")
                        #st.image(pyc.plot_model(LOAD_REDIS("pycaretmodel"), i , save= True, use_train_data = True), use_column_width=True)
                    except:
                        st.write(f"Plot {i} konnte nicht erstellt werden!")
                        
            with col2:
                st.markdown("### Ergebnisse TEST-Datensatz") 
                for i in PLOTS:
                    try:
                        st.markdown(f"#### {i}")
                        pcc.plot_model(trained_classification_model(), i ,use_train_data = False, display_format="streamlit")
                        #st.image(pyc.plot_model(LOAD_REDIS("pycaretmodel"), i , save= True, use_train_data = False), use_column_width=True)
                    except:
                        st.write(f"Plot {i} konnte nicht erstellt werden!")


    elif TYPE == "Regression":
        if st.button("Train Model"):
            SETUPREGRESSION = pcr.setup(data = DATENSATZ, target = TARGET, silent = True, html = False)
            BEST = pcr.compare_models()

        try:
            st.write(pcr.get_config("display_container")[1])
            st.write(BEST)
        except:
            pass

        PLOTS = st.multiselect(
            label = "AUSWAHL_PLOTS", 
            options = ["residuals_interactive", "residuals", "error", "cooks", "rfe", "learning", "boundary", "rfe", "vc", "manifold", "feature", "feature_all", "parameter", "feature", "tree"], 
            default = ["residuals_interactive", "residuals", "error"]
            )

        col1, col2 = st.beta_columns(2)

        with col1:
            st.markdown("### Ergebnisse TRAININGS-Datensatz")
            for i in PLOTS:
                try:
                    st.markdown(f"#### {i}")
                    pcc.plot_model(BEST, i ,use_train_data = True, display_format="streamlit")
                    #st.image(pyc.plot_model(LOAD_REDIS("pycaretmodel"), i , save= True, use_train_data = True), use_column_width=True)
                except:
                    st.write(f"Plot {i} konnte nicht erstellt werden!")
                    
        with col2:
            st.markdown("### Ergebnisse TEST-Datensatz") 
            for i in PLOTS:
                try:
                    st.markdown(f"#### {i}")
                    pcc.plot_model(BEST, i ,use_train_data = False, display_format="streamlit")
                    #st.image(pyc.plot_model(LOAD_REDIS("pycaretmodel"), i , save= True, use_train_data = False), use_column_width=True)
                except:
                    st.write(f"Plot {i} konnte nicht erstellt werden!")
        

    st.write("_______")
    st.header("Explain trained Model")

    with st.beta_expander("Explainer Settings"):
        pass

    EXPLAINER = dx.Explainer(
        model = trained_classification_model(),
        data = DATENSATZ,
        y = DATENSATZ[TARGET],
        model_type= TYPE)

    

if __name__ == "__main__":
    main()
