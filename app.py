import logging, os
from pycaret.internal.preprocess import DFS_Classic

#Streamlit
import streamlit as st
from streamlit import caching
import streamlit.components.v1 as components
from streamlit.elements.image import _BytesIO_to_bytes
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
import pycaret as pc
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

  
################################################
# Site Startup
################################################
st.set_page_config(
    page_title="Explainable-ml-app",
    page_icon = "media/Icon.ico",
    layout="wide",
    initial_sidebar_state="expanded")


# with st.sidebar.expander("Settings"):
#     BREITE = st.slider(label ="Display-size", min_value = 300, max_value=3000, value = 1400, step= 100)

# st.markdown(
#     f"""
#     <style>
#     .reportview-container .main .block-container{{
#         max-width: {BREITE}px;
#     }}
#     </style>
#     """,unsafe_allow_html=True)

################################################
# Functions
################################################

def sidebar_render_svg(svg):
    """Renders the given svg string."""
    f = open(svg,"r")
    lines = f.readlines()
    line_string=''.join(lines)
    b64 = base64.b64encode(line_string.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.sidebar.write(html, unsafe_allow_html=True)

sidebar_render_svg("media/logo.svg")
st.sidebar.write("Build with Streamlit, PyCaret & Dalex")
st.sidebar.info("Big Datasets cause slow behavior of the application")
st.sidebar.write("Streamlit: ", st.__version__)
st.sidebar.write("PyCaret: ", pc.__version__)
st.sidebar.write("Dalex: ", dx.__version__)

def upload_data(file):
    DF = pd.read_csv(file, encoding='utf-8')

def plot_model_classification(MODEL, PLOTS, use_train_data ):
    for i in PLOTS:
        try:
            st.markdown(f"#### {i}")
            pcc.plot_model(MODEL, i ,use_train_data = use_train_data, display_format="streamlit")
        except:
            st.write(f"Plot {i} konnte nicht erstellt werden!")

def plot_model_regression(MODEL, PLOTS, use_train_data ):
    for i in PLOTS:
        try:
            st.markdown(f"#### {i}")
            pcr.plot_model(MODEL, i ,use_train_data = use_train_data, display_format="streamlit")
        except:
            st.write(f"Plot {i} konnte nicht erstellt werden!")
################################################
# Main
################################################
def main():
    TYPE = st.sidebar.selectbox(label = "Type", options = ["Classification", "Regression"])
    st.write(f"# {TYPE}")

    DATA = st.sidebar.selectbox(label = "Data", options = ["Dummydata", "CSV-File"])

    if DATA == "Dummydata":
        INDEX = get_data("index")
        AUSWAHL = st.sidebar.selectbox(label = "Dataset", options = INDEX.Dataset.tolist(), index = 21)

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
        with st.expander("Data Report"):
            import streamlit.components.v1 as components
            components.html(html=ProfileReport(df = DATENSATZ, minimal = False).to_html(), scrolling = True, height = 1000)

    st.header("Target")

    TARGET = st.selectbox(
        label = "choose target", 
        options = [" "] + DATENSATZ.columns.tolist(), 
        index = (0))
    
    if TARGET == " ":
        st.stop()

    col1, _ = st.columns(2)
    try:
        with col1:
            st.bar_chart(DATENSATZ[TARGET].value_counts())
        import numpy as np
        neg, pos = np.bincount(DATENSATZ[TARGET])
        total = neg + pos
        st.text('Instance: \n Total: {}\n Positiv: {} ({:.2f}% of all)\n'.format(total, pos, 100 * pos / total))
    except:
        pass

    st.header("Data Preprocessing")
    if TYPE == "Classification":
        #Setup
        with st.form("Setup"):
            train_size = st.slider(label= "train_size", min_value = 0.01, max_value=1.00, value= 0.70)
            normalize = st.selectbox(label = "normalize", options = [False, True])
            submitted = st.form_submit_button("Run Setup")

        if submitted:
            with st.spinner("Setup"):
                SETUPCLASSIFICATION = pcc.setup(
                    data = DATENSATZ, 
                    target = TARGET, 
                    silent = True,
                    html = False,

                    train_size = train_size,
                    normalize = normalize
                    )
                pcc.save_config(file_name = 'config/classification_config.pkl')
        
        #SETUPCLASSIFICATION = pcc.load_config(file_name = 'config/classification_config.pkl')
        #st.write(SETUPCLASSIFICATION)

        try:
            with st.expander(label = "Setup Result"):
                st.write("Setup")
                import streamlit.components.v1 as components
                components.html(pcc.get_config("display_container")[0].to_html(), scrolling = True)
                st.write("Traindata")
                st.write(pd.concat([pcc.get_config("X_train"),pcc.get_config("y_train")], axis=1, join='inner'))
                st.write("Testdata")
                st.write(pd.concat([pcc.get_config("X_test"),pcc.get_config("y_test")], axis=1, join='inner'))
        except:
            st.stop()

        # Model
        st.header("Train Model")
        
        MODELS = ["lr", "knn", "nb", "dt", "svm", "rbfsvm", "gpc", "mlp", "ridge", "rf", "qda", "ada", "gbc", "lda", "et", "xgboost", "lightgbm", "catboost"]
        with st.form(key='Train_classification_Model'):
            MODELS_WAHL = st.multiselect(
                label = "models", 
                options = MODELS,
                default = MODELS)
            SORT = st.selectbox(label = "sort", options = ["Accuracy", "AUC", "Recall", "Prec.", "F1"])

            classification_submit_button = st.form_submit_button(label='Train Model(s)')

        if classification_submit_button:
            with st.spinner("Train Model(s)"):
                if len(MODELS_WAHL) > 1:
                    BEST = pcc.compare_models(
                        include = MODELS_WAHL,
                        sort = SORT,
                        )
                else:
                    BEST = pcc.create_model(
                        estimator = MODELS_WAHL[0]
                        )

                pcc.save_model(model = BEST, model_name="Model/model", model_only =True)
                pcc.save_model(model = BEST, model_name="Model/modelpipeline", model_only =False)

        try:
            display_container1 = pcc.get_config("display_container")
            MODEL = pcc.load_model(model_name="Model/model")
            MODELPIPELINE = pcc.load_model(model_name="Model/modelpipeline")
        except:
            st.stop()

        with st.expander("Training Result"):
            st.dataframe(display_container1[1])

            PLOTS = st.multiselect(
                    label = "AUSWAHL_PLOTS", 
                    options = ["auc", "threshold", "pr", "confusion_matrix", "error", "class_report", "boundary", "rfe", "learning", "manifold", "calibration", "vc", "dimension", "feature", "lift", "gain", "tree"], 
                    default = ["confusion_matrix", "pr"]
                    )

            # Plots 
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Ergebnisse TRAININGS-Datensatz")
                plot_model_classification(MODEL, PLOTS, True)
                        
            with col2:
                st.markdown("### Ergebnisse TEST-Datensatz") 
                plot_model_classification(MODEL, PLOTS, False)



    elif TYPE == "Regression":
        with st.form("Setup Regression"):
            train_size = st.slider(label= "train_size", min_value = 0.01, max_value=1.00, value= 0.70)
            normalize = st.selectbox(label = "normalize", options = [False, True])
            submitted = st.form_submit_button("Run Setup")

        if submitted:
            with st.spinner("Setup"):
                SETUPREGRESSION = pcr.setup(
                    data = DATENSATZ, 
                    target = TARGET, 
                    silent = True, 
                    html = False)
                
                pcr.save_config(file_name = 'config/classification_config.pkl')

        try:
            with st.expander(label = "Setup Result Regression"):
                st.write("Setup")
                import streamlit.components.v1 as components
                components.html(pcr.get_config("display_container")[0].to_html(), scrolling = True)
                st.write("Traindata")
                st.write(pd.concat([pcr.get_config("X_train"),pcr.get_config("y_train")], axis=1, join='inner'))
                st.write("Testdata")
                st.write(pd.concat([pcr.get_config("X_test"),pcr.get_config("y_test")], axis=1, join='inner'))
        except:
            st.stop()

        st.header("Train Model Regression")
            
        with st.form(key = "Train_classification_Model"):
            MODELS_regression = ["lr", "lasso", "ridge", "en", "lar", "llar", "omp", "br", "ard", "par", "ransac", "tr", "huber", "kr", "svm", "knn", "dt", "rf", "et", "ada", "gbr", "mlp", "xgboost", "lightgbm", "catboost"]
            MODELS_WAHL = st.multiselect(
                label = "models", 
                options = MODELS_regression,
                default = MODELS_regression)
            SORT = st.selectbox(label = "sort", options = ["Accuracy", "AUC", "Recall", "Prec.", "F1"])
            regression_submit_button = st.form_submit_button(label='Train Model(s)')

        if regression_submit_button:
            with st.spinner("Train Model(s)"):
                if len(MODELS_WAHL) > 1:
                    BEST = pcr.compare_models(
                        include = MODELS_WAHL,
                        )
                else:
                    BEST = pcr.create_model(
                        estimator = MODELS_WAHL[0]
                        )

                pcr.save_model(model = BEST, model_name="Model/model", model_only =True)
                pcr.save_model(model = BEST, model_name="Model/modelpipeline", model_only =False)

        try:
            display_container1 = pcr.get_config("display_container")
            MODEL = pcr.load_model(model_name="Model/model")
            MODELPIPELINE = pcr.load_model(model_name="Model/modelpipeline")
        except:
            st.stop()

        with st.expander("Training Result"):
            st.dataframe(display_container1[1])

            PLOTS = st.multiselect(
                label = "AUSWAHL_PLOTS", 
                options = ["residuals_interactive", "residuals", "error", "cooks", "rfe", "learning", "boundary", "rfe", "vc", "manifold", "feature", "feature_all", "parameter", "feature", "tree"], 
                default = ["residuals_interactive", "residuals", "error"]
                )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Ergebnisse TRAININGS-Datensatz")
                plot_model_regression(MODEL, PLOTS, True)
                        
            with col2:
                st.markdown("### Ergebnisse TEST-Datensatz") 
                plot_model_regression(MODEL, PLOTS, False)
        

    st.header("Explain trained Model")
    with st.form(key='Explain trained Model'):
        ERKLÄRUNGEN = st.multiselect(
            label="Auswahl",
            options = ["predict_parts", "predict_profile", "predict_surrogate", "model_parts", "model_profile", "model_surrogate"],
            default = ["predict_parts","predict_profile", "predict_surrogate", "model_parts", "model_profile", "model_surrogate"]
            )

        if "predict_parts" in ERKLÄRUNGEN:
            predict_parts_type = st.selectbox(label = "type_predict_parts", options = ["break_down", "break_down_interactions", "shap"]) #"break_down_interactions" "shap_wrapper"

        if "predict_profile" in ERKLÄRUNGEN:
            COLUMNS = pcc.get_config("X_test").columns
            variable_type_model_profile_cat = st.multiselect(label = "variable_type_model_profile_cat", options = COLUMNS , default = COLUMNS[0])


        if "predict_surrogate" in ERKLÄRUNGEN:
            pass
        
        if "model_parts" in ERKLÄRUNGEN:
            pass

        if "model_profile" in ERKLÄRUNGEN:
            model_profile_var = st.multiselect(label = "model_profile_var", options = COLUMNS, default=COLUMNS[0])
            variable_type_model_profile = st.selectbox(label = "variable_type_model_profile", options = ['numerical', 'categorical'])
            #variable_type_model_profile_cat = st.multiselect(label = "variable_type_model_profile_cat", options =COLUMNS)

        if "model_surrogate" in ERKLÄRUNGEN:
            MAX_VARS = st.number_input(label= "max_vars", value = 5)
            MAX_DEPTH = st.number_input(label="max_depth", value = 3)
        
        if any(item in ["predict_parts", "predict_profile", "predict_surrogate"] for item in ERKLÄRUNGEN) is True:
            slider_idx = st.multiselect(label ="Instanz", options = pcc.get_config("X_test").index.values.tolist() , default = pcc.get_config("X_test").index.values[0])
        
        if any(item in ["model_parts", "model_profile", "model_surrogate"] for item in ERKLÄRUNGEN)is True:
            pass

        ex_plain_submit_button = st.form_submit_button(label='Calculate Explainations')

    if ex_plain_submit_button:
        if TYPE == "Classification":
            TYPE_DALEX = "classification"
        else:
            TYPE_DALEX = "regression"

        EXPLAINER = dx.Explainer(
            model = MODEL,
            data = pcc.get_config("X_test"),
            y = pcc.get_config("y_test"),
            model_type= TYPE_DALEX)

        with open('explainer/explainer.pkl', 'wb') as fd:
            EXPLAINER.dump(fd)

    with open('explainer/explainer.pkl', 'rb') as fd:
        EXPLAINER = dx.Explainer.load(fd)
    
    OPS = pcc.get_config("X_test")
    LABELS = pcc.get_config("y_test")

    #Erkärungen neu berechnen
    my_bar_training_head=st.empty()
    my_bar_training_head.write("Fortschritt Plots")
    my_bar_training = st.empty()
    my_bar_training.progress(0)

    my_bar_training.progress(10)

    # Instanzbasiert
    if any(item in ["predict_parts", "predict_profile", "predict_surrogate"] for item in ERKLÄRUNGEN) is True:
    
        st.subheader("Local Explainations")

        for i in slider_idx:
            st.text(f"Instanz: {i} | Wahrscheinlichkeit des Modells: {EXPLAINER.predict(OPS.loc[[i]])[0]} | Vorhersage: {round(EXPLAINER.predict(OPS.loc[[i]])[0])} | Ist-Wert: {LABELS.loc[i]}")
            pass

        if "predict_parts" in ERKLÄRUNGEN:

            with st.expander(label="predict_parts - Break Down, Shap", expanded=True):

                pp_list = []
                for i in slider_idx:
                    pp = EXPLAINER.predict_parts( new_observation = OPS.loc[i], type= predict_parts_type , label = str(i))
                    pp_list += [pp]
                
                st.plotly_chart(pp_list[0].plot(pp_list[1::], show=False), use_column_width=True)
                    

        my_bar_training.progress(20)

        if "predict_profile" in ERKLÄRUNGEN:
            with st.expander(label="predict_profile - Ceteris Paribus", expanded=True):
               
                ppr_list = []
                for i in slider_idx:
                    player = OPS.loc[i]
                    ppr = EXPLAINER.predict_profile(new_observation = player, variables=variable_type_model_profile_cat ,  type="ceteris_paribus",  label=i)
                    ppr_list += [ppr]

                try:
                    st.plotly_chart(ppr_list[0].plot(ppr_list[1::], show=False), use_column_width=True)
                except:
                    st.warning("Button - Erklärungen berechnen lassen -  drücken")
                    st.stop()


        my_bar_training.progress(30)

        if "predict_surrogate" in ERKLÄRUNGEN:
            with st.expander(label="predict_surrogate - Lime", expanded=True):
                    
                lime_explanation_list = []
                lime_explanation_dataframe = list()
                for i in slider_idx:
                    # st.write("Zeile: ", str(i))
                    lime = EXPLAINER.predict_surrogate(OPS.loc[i],mode='classification')
                    lime_explanation_list.append(lime.as_html(show_all=False))
                    lime_explanation_dataframe.append(lime.result)

                import streamlit.components.v1 as components
                for i in lime_explanation_list:
                    components.html(i, height=1000)


    if any(item in ["model_parts", "model_profile", "model_surrogate"] for item in ERKLÄRUNGEN)is True:
        st.write("________")
        st.subheader("Global Explainations")

        my_bar_training.progress(40)

        if "model_parts" in ERKLÄRUNGEN:
            with st.expander(label="model_parts - Permutationsbasierte Merkmalswichtigkeit, Shap Summary", expanded=True):
                
                #model_parts_type = st.selectbox(label = "Type", options = ["variable_importance"]) # "ratio", "difference"

                variable_importance = EXPLAINER.model_parts(loss_function = "1-auc", type = "variable_importance")
                
                try:
                    exp = EXPLAINER.model_parts(type='shap_wrapper', shap_explainer_type = "TreeExplainer" ,processes=6, random_state = 42)
                except:
                    exp = EXPLAINER.model_parts(type='shap_wrapper', processes=6, random_state = 42)

                shap_wrapper_result = exp.result
                try:
                    shap_wrapper = exp
                except:
                    shap_wrapper = exp.plot(show = False)

                try:
                    st.write("Permutationsbasierte Merkmalswichtigkeit")
                    st.plotly_chart(variable_importance.plot(show = False))

                    st.write("Shap Summary")
                    try:
                        st.pyplot(shap_wrapper.plot(show = False))
                    except:
                        st.pyplot(shap_wrapper)

                except:
                    st.warning("Button - Erklärungen berechnen lassen -  drücken")
                    st.stop()

        my_bar_training.progress(50)

        if "model_profile" in ERKLÄRUNGEN:
            with st.expander(label="model_profile - Partiellen Abhängigkeitskurven", expanded=True):
                
                # pdp plots
                partial = EXPLAINER.model_profile(type='partial', label='partial',variable_type =variable_type_model_profile, variables =model_profile_var)
                partial = partial.plot(geom = "profiles", size = 1,  show=False)
                # groups = variable_type_model_profile_cat
                try:
                    st.plotly_chart(partial, use_column_width=True) #[accumulated,conditional]
                except:
                    st.warning("Button - Erklärungen berechnen lassen -  drücken")
                    st.stop()

        my_bar_training.progress(60)

        if "model_surrogate" in ERKLÄRUNGEN:
            with st.expander(label="model_surrogate - Lime Entscheidungsbaum", expanded=True):

                model_surrogate_tree = EXPLAINER.model_surrogate(type='tree', max_vars= MAX_VARS, max_depth = MAX_DEPTH)
                
                from sklearn import tree

                model_surrogate_tree_performane = model_surrogate_tree.performance
                model_surrogate_tree_feature_names = model_surrogate_tree.feature_names
                
                model_surrogate_tree_plot = tree.export_graphviz(
                    decision_tree = model_surrogate_tree,
                    feature_names = model_surrogate_tree_feature_names ,
                    class_names = model_surrogate_tree.class_names , 
                    filled =True,
                    rounded = True)

                from dtreeviz.trees import dtreeviz

                st.graphviz_chart(model_surrogate_tree_plot)

                try:
                    # Vergleich mit verwendeten Modell
                    st.write("Eingereichtes Modell:")
                    # Performance des model_surrogates
                    st.write("Surrogate Modell:")
                    st.write(model_surrogate_tree_performane)
                    # Verwendete Klassen
                    st.write("Verwendete Attribute für den Surrogate:")
                    st.write(model_surrogate_tree_feature_names)
                    # Baum Plotten

                except:
                    st.stop()

    my_bar_training.progress(70)

    my_bar_training.progress(100)
    time.sleep(2)
    my_bar_training.empty()
    my_bar_training_head.empty()
    st.stop()


if __name__ == "__main__":
    main()
