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

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)

# with st.sidebar.beta_expander("Settings"):
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

st.sidebar.write("# Explainable-Machine-Learning-App")
st.sidebar.write("Build with PyCaret & Dalex")
st.sidebar.info("Big Datasets cause slow behavior of the application")

def upload_data(file):
    DF = pd.read_csv(file, encoding='utf-8')

def plot_model(MODEL, PLOTS, use_train_data ):
    for i in PLOTS:
        try:
            st.markdown(f"#### {i}")
            pcc.plot_model(MODEL, i ,use_train_data = use_train_data, display_format="streamlit")
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
        with st.beta_expander("Data Report"):
            import streamlit.components.v1 as components
            components.html(html=ProfileReport(df = DATENSATZ, minimal = False).to_html(), scrolling = True, height = 1000)

    st.header("Target")

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
            with st.beta_expander(label = "Setup Result"):
                st.write("Setup")
                st.write(pcc.get_config("display_container")[0])
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
            if len(MODELS_WAHL) > 1:
                NUMBER = 1
            else:
                NUMBER = 2
            display_container1 = pcc.get_config("display_container")[1]
            MODEL = pcc.load_model(model_name="Model/model")
            MODELPIPELINE = pcc.load_model(model_name="Model/modelpipeline")
        except:
            st.stop()

        with st.beta_expander("Training Result"):
            st.write(display_container1)

            PLOTS = st.multiselect(
                    label = "AUSWAHL_PLOTS", 
                    options = ["auc", "threshold", "pr", "confusion_matrix", "error", "class_report", "boundary", "rfe", "learning", "manifold", "calibration", "vc", "dimension", "feature", "lift", "gain", "tree"], 
                    default = ["confusion_matrix", "pr"]
                    )

            # Plots 
            col1, col2 = st.beta_columns(2)
            with col1:
                st.markdown("### Ergebnisse TRAININGS-Datensatz")
                plot_model(MODEL, PLOTS, True)
                        
            with col2:
                st.markdown("### Ergebnisse TEST-Datensatz") 
                plot_model(MODEL, PLOTS, False)



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
            plot_model(MODEL, PLOTS, True)
                    
        with col2:
            st.markdown("### Ergebnisse TEST-Datensatz") 
            plot_model(MODEL, PLOTS, False)
        

    st.header("Explain trained Model")
    with st.form(key='Explain trained Model'):
        ERKLÄRUNGEN = st.multiselect(
            label="Auswahl",
            options = ["predict_parts", "predict_profile", "predict_surrogate", "model_parts", "model_profile", "model_surrogate"],
            )
        GRAFIK_WAHL = st.selectbox(label = "Darstellung der Erklärungen (Wenn möglich! Standartmäßig Grafiken)", options = ["Grafiken", "Tabellen"])
        
        ex_plain_submit_button = st.form_submit_button(label='Calculate Explainations')

    if ex_plain_submit_button:
        EXPLAINER = dx.Explainer(
            model = MODEL,
            data = DATENSATZ,
            y = DATENSATZ[TARGET],
            model_type= TYPE)

        with open('explainer/explainer.pkl', 'wb') as fd:
            EXPLAINER.dump(fd)


    with open('explainer/explainer.pkl', 'rb') as fd:
        EXPLAINER = dx.Explainer.load(fd)
    

    #Erkärungen neu berechnen
    my_bar_training_head=st.empty()
    my_bar_training_head.write("Fortschritt Plots")
    my_bar_training = st.empty()
    my_bar_training.progress(0)

    my_bar_training.progress(10)

    # Instanzbasiert
    if "predict_parts" or "predict_profile" or "predict_surrogate"  in ERKLÄRUNGEN:
    
        st.subheader("Lokale Erklärungen")

        st.write("Rohdaten")
        try:
            st.write(LOAD_REDIS("DF_VAL_SAUBER").loc[LOAD_REDIS("X_test").index].style.highlight_max(subset=LOAD_REDIS("target_col"), color='lightgreen').highlight_min(subset=LOAD_REDIS("target_col"),color='#cd4f39'))
        except:
            st.write(LOAD_REDIS("DF_SAUBER").loc[LOAD_REDIS("X_test").index].style.highlight_max(subset=LOAD_REDIS("target_col"), color='lightgreen').highlight_min(subset=LOAD_REDIS("target_col"),color='#cd4f39'))
        
        st.write("Verarbeitete Daten")
        st.write(pd.concat([LOAD_REDIS("y_test"),LOAD_REDIS("X_test")], axis=1, join='inner').style.highlight_max(subset=LOAD_REDIS("target_col"), color='lightgreen').highlight_min(subset=LOAD_REDIS("target_col"),color='#cd4f39'))
        

        SAVE_REDIS(WERT = "slider_idx", INHALT = st.multiselect(label ="Instanz", options = LOAD_REDIS("X_test").index.values.tolist() , default = LOAD_REDIS("X_test").index.values[0]))
        #LOAD_REDIS("slider_idx")

        for i in LOAD_REDIS("slider_idx"):
            st.text(f"Instanz: {i} | Wahrscheinlichkeit des Modells: {EXPLAINER.predict(OPS.loc[[i]])[0]} | Vorhersage: {round(EXPLAINER.predict(OPS.loc[[i]])[0])} | Ist-Wert: {LABELS.loc[i]}")
            pass

    if "predict_parts" in LOAD_REDIS("Auswahl_Erklärung"):

        with st.beta_expander(label="predict_parts - Break Down, Shap", expanded=True):
            
            predict_parts_type = st.selectbox(label = "type", options = ["break_down", "break_down_interactions", "shap", "shap_wrapper"]) #"break_down_interactions" "shap_wrapper"

            if predict_parts_type == "shap_wrapper":
                shap_html_list = []
                if LOAD_REDIS("BUTTON_ERKLÄR"):
                    import streamlit.components.v1 as components
                    for i in LOAD_REDIS("slider_idx"):
                            #st.write("Zeile: ", str(i))
                            shap_html = EXPLAINER.predict_parts(new_observation = OPS.loc[i], type= predict_parts_type, keep_distributions = False, interaction_preference = 0)

                try:
                    for i in shap_html_list:
                        st.pyplot(i.plot())
                except:
                    st.warning("Button - Erklärungen berechnen lassen -  drücken")
                    st.stop()
            
            else:
                if LOAD_REDIS("BUTTON_ERKLÄR"):
                    predict_parts_plot = {'ibd':[]}
                    predict_parts_dataframe = list()
                    for i in LOAD_REDIS("slider_idx"):
                        idb = EXPLAINER.predict_parts( new_observation = OPS.loc[i], type= predict_parts_type, keep_distributions = False , label = str(i), interaction_preference = 0 )
                        predict_parts_plot['ibd'].append(idb)
                        predict_parts_dataframe.append(idb.result)
                    
                    SAVE_REDIS(WERT = "predict_parts", INHALT = predict_parts_plot)
                    SAVE_REDIS(WERT = "predict_parts_dataframe", INHALT = predict_parts_dataframe)
                
                try:
                    if LOAD_REDIS("GRAFIK_WAHL") == "Tabellen":
                        for i in LOAD_REDIS("predict_parts_dataframe"):
                            st.write(i.iloc[:,2:-3])

                    if LOAD_REDIS("GRAFIK_WAHL") == "Grafiken":
                        st.plotly_chart(LOAD_REDIS("predict_parts")['ibd'][0].plot(LOAD_REDIS("predict_parts")['ibd'][1:], show=False, max_vars=100), use_column_width=True)

                except:
                    st.warning("Button - Erklärungen berechnen lassen -  drücken")
                    st.stop()

    my_bar_training.progress(20)

    if "predict_profile" in LOAD_REDIS("Auswahl_Erklärung"):
        with st.beta_expander(label="predict_profile - Ceteris Paribus", expanded=True):

            display_info("Interpretation","Break_down_Attributions") # Information Einblenden
            
            variable_type_model_profile_cat = st.multiselect(label = "variable_type_model_profile_cat", options = COLUMNS , default = COLUMNS[0])

            if LOAD_REDIS("BUTTON_ERKLÄR"):

                va = {'ibd':[]}
                predict_profile_dataframe = list()
                for i in LOAD_REDIS("slider_idx"):
                    player = OPS.loc[i]
                    idb = EXPLAINER.predict_profile(new_observation = player, variables=variable_type_model_profile_cat ,  type="ceteris_paribus",  label=i)
                    va['ibd'].append(idb)
                    predict_profile_dataframe.append(idb.result)

                
                SAVE_REDIS(WERT ="predict_profile", INHALT = va)
                SAVE_REDIS(WERT ="predict_profile_dataframe", INHALT = predict_profile_dataframe)

            try:
                if LOAD_REDIS("GRAFIK_WAHL") == "Tabellen":
                    for i in LOAD_REDIS("predict_profile_dataframe"):
                        st.write(i)

                if LOAD_REDIS("GRAFIK_WAHL") == "Grafiken":
                    st.plotly_chart(LOAD_REDIS("predict_profile")['ibd'][0].plot(LOAD_REDIS("predict_profile")['ibd'][1:], show=False), use_column_width=True)
            except:
                st.warning("Button - Erklärungen berechnen lassen -  drücken")
                st.stop()

        print(NAMESPACE, "predict_profile fertig")

    my_bar_training.progress(30)

    if "predict_surrogate" in LOAD_REDIS("Auswahl_Erklärung"):
        with st.beta_expander(label="predict_surrogate - Lime", expanded=True):

            #Lime plotten
            if LOAD_REDIS("BUTTON_ERKLÄR"):
                
                lime_explanation_list = []
                lime_explanation_dataframe = list()
                for i in LOAD_REDIS("slider_idx"):
                    # st.write("Zeile: ", str(i))
                    lime = EXPLAINER.predict_surrogate(OPS.loc[i],mode='classification')
                    lime_explanation_list.append(lime.as_html(show_all=False))
                    lime_explanation_dataframe.append(lime.result)
                
                SAVE_REDIS(WERT= "lime_explanation_list", INHALT = lime_explanation_list)
                SAVE_REDIS(WERT= "lime_explanation_dataframe", INHALT = lime_explanation_dataframe)

            try:
                if LOAD_REDIS("GRAFIK_WAHL") == "Tabellen":
                    for i in LOAD_REDIS("lime_explanation_dataframe"):
                        st.write(i)
                
                if LOAD_REDIS("GRAFIK_WAHL") == "Grafiken":
                    import streamlit.components.v1 as components
                    for i in LOAD_REDIS("lime_explanation_list"):
                        components.html(i, height=1000)
            except:
                st.warning("Button - Erklärungen berechnen lassen -  drücken")
                st.stop()

        print(NAMESPACE, "predict_surrogate fertig")

    if "model_parts" in LOAD_REDIS("Auswahl_Erklärung") or "model_profile" in LOAD_REDIS("Auswahl_Erklärung") or "model_surrogate" in LOAD_REDIS("Auswahl_Erklärung"):
        st.write("________")
        st.subheader("Globale Erklärungen")

    my_bar_training.progress(40)

    if "model_parts" in LOAD_REDIS("Auswahl_Erklärung"):
        with st.beta_expander(label="model_parts - Permutationsbasierte Merkmalswichtigkeit, Shap Summary", expanded=True):
            
            #model_parts_type = st.selectbox(label = "Type", options = ["variable_importance"]) # "ratio", "difference"

            if LOAD_REDIS("BUTTON_ERKLÄR"):
                SAVE_REDIS(WERT = "variable_importance", INHALT = EXPLAINER.model_parts(loss_function = "1-auc", type = "variable_importance"))
                
                try:
                    exp = EXPLAINER.model_parts(type='shap_wrapper', shap_explainer_type = "TreeExplainer" ,processes=6, random_state = 42)
                except:
                    exp = EXPLAINER.model_parts(type='shap_wrapper', processes=6, random_state = 42)

                SAVE_REDIS(WERT = "shap_wrapper_result", INHALT = exp.result)
                try:
                    SAVE_REDIS(WERT = "shap_wrapper", INHALT = exp)
                except:
                    SAVE_REDIS(WERT = "shap_wrapper", INHALT = exp.plot(show = False))
            try:
                if LOAD_REDIS("GRAFIK_WAHL") == "Tabellen":
                    st.write("Permutationsbasierte Merkmalswichtigkeit")
                    st.dataframe(LOAD_REDIS("variable_importance").result[:-1].style.bar(subset=["dropout_loss"], color='#d65f5f'), height= 10000, width = 10000)
                    st.write("Shap Summary")
                    st.write(LOAD_REDIS("shap_wrapper_result"))

                if LOAD_REDIS("GRAFIK_WAHL") == "Grafiken":
                    st.write("Permutationsbasierte Merkmalswichtigkeit")
                    st.plotly_chart(LOAD_REDIS("variable_importance").plot(show = False))

                    st.write("Shap Summary")
                    try:
                        st.pyplot(LOAD_REDIS("shap_wrapper").plot(show = False))
                    except:
                        st.pyplot(LOAD_REDIS("shap_wrapper"))

            except:
                st.warning("Button - Erklärungen berechnen lassen -  drücken")
                st.stop()

        print(NAMESPACE, "model_parts fertig")

    my_bar_training.progress(50)

    if "model_profile" in LOAD_REDIS("Auswahl_Erklärung"):
        with st.beta_expander(label="model_profile - Partiellen Abhängigkeitskurven", expanded=True):
            
            # pdp plots
            model_profile_var = st.multiselect(label = "model_profile_var", options = COLUMNS, default=COLUMNS[0])
            variable_type_model_profile = st.selectbox(label = "variable_type_model_profile", options = ['numerical', 'categorical'])
            #variable_type_model_profile_cat = st.multiselect(label = "variable_type_model_profile_cat", options =COLUMNS)

            st.write("numeric_features: ")
            st.write(LOAD_REDIS(WERT = "numeric_features"))
            st.write("categorical_features: ")
            st.write(LOAD_REDIS(WERT = "categorical_features"))

            if LOAD_REDIS("BUTTON_ERKLÄR"):
                partial = EXPLAINER.model_profile(type='partial', label='partial',variable_type =variable_type_model_profile, variables =model_profile_var)
                SAVE_REDIS(WERT = "partial", INHALT = partial.plot(geom = "profiles", size = 1,  show=False))
                SAVE_REDIS(WERT = "partial_dataframe", INHALT = partial.result)
            # groups = variable_type_model_profile_cat
            try:
                if LOAD_REDIS("GRAFIK_WAHL") == "Tabellen":
                    st.write(LOAD_REDIS("partial_dataframe"))
                if LOAD_REDIS("GRAFIK_WAHL") == "Grafiken":
                    st.plotly_chart(LOAD_REDIS("partial"), use_column_width=True) #[accumulated,conditional]
            except:
                st.warning("Button - Erklärungen berechnen lassen -  drücken")
                st.stop()

        print(NAMESPACE, "model_profile fertig")

    my_bar_training.progress(60)

    if "model_surrogate" in LOAD_REDIS("Auswahl_Erklärung"):
        with st.beta_expander(label="model_surrogate - Lime Entscheidungsbaum", expanded=True):

            MAX_VARS = st.number_input(label= "max_vars", value = 5)
            MAX_DEPTH = st.number_input(label="max_depth", value = 3)

            if LOAD_REDIS("BUTTON_ERKLÄR"):
            
                model_surrogate_tree = EXPLAINER.model_surrogate(type='tree', max_vars= MAX_VARS, max_depth = MAX_DEPTH)
                from sklearn import tree

                SAVE_REDIS(WERT ="model_surrogate_tree_performane", INHALT = model_surrogate_tree.performance)
                SAVE_REDIS(WERT ="model_surrogate_tree_feature_names", INHALT = model_surrogate_tree.feature_names)
                SAVE_REDIS( WERT = "model_surrogate_tree",  INHALT = tree.export_graphviz(
                    decision_tree = model_surrogate_tree,
                    feature_names = model_surrogate_tree.feature_names ,
                    class_names = model_surrogate_tree.class_names , 
                    filled =True,
                    rounded = True,
                    out_file=None))

            try:
                # Vergleich mit verwendeten Modell
                st.write("Eingereichtes Modell:")
                st.write(EXPLAINER.model_performance('classification').result)
                # Performance des model_surrogates
                st.write("Surrogate Modell:")
                st.write(LOAD_REDIS("model_surrogate_tree_performane"))
                # Verwendete Klassen
                st.write("Verwendete Attribute für den Surrogate:")
                st.write(LOAD_REDIS("model_surrogate_tree_feature_names"))
                # Baum Plotten
                st.graphviz_chart(LOAD_REDIS("model_surrogate_tree"))

            except:
                st.warning("Button - Erklärungen berechnen lassen -  drücken")
                st.stop()

        print(NAMESPACE, "model_surrogate fertig")

    my_bar_training.progress(70)

    my_bar_training.progress(100)
    time.sleep(2)
    my_bar_training.empty()
    my_bar_training_head.empty()
    st.stop()





    

if __name__ == "__main__":
    main()
