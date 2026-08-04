# Step 1: Load Important modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
from sklearn.datasets import load_iris
import pickle

import os

st.write("Current Directory:", os.getcwd())
st.write("Files in Directory:", os.listdir())

# LOAD DATASET
data = load_iris()
df = pd.DataFrame(data['data'], 
                  columns = data['feature_names'])
df['target'] = data['target']
classes = data['target_names']

X = df.iloc[:,:-1]

# MODEL_LIST
all_model_name = ['Logistic Regression',
                 'Naive Bayes',"Decision Tree",
                 "SVM",
                 "KNN"]



# MODEL_LIST
all_model_name = [
    'Logistic Regression',
    'Naive Bayes',
    'Decision Tree',
    'SVM',
    'KNN'
]

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
st.write("Current Directory:", os.getcwd())
st.write("Files Found:", os.listdir(BASE_DIR))

all_models = []

for model_name in all_model_name:
    file_path = os.path.join(BASE_DIR, f"{model_name}.pkl")

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            model = pickle.load(f)
            all_models.append(model)
    else:
        st.error(f"Model file not found: {file_path}")

# USER INPUT AND PAGE TITLE
st.title("ML Flower Classification Project")
# Image url
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQF2roQNP1rPFtklA8xgZt76jyhj6x2BUjVe6gxwxJ53pI0_TYfQLRZh8oZ&s=10"
st.image(url)

# Show Dataframe sample
st.dataframe(df.sample(5))

# LEFT SIDE BAR for USER VALUE INPUT
st.sidebar.title("Select Iris Features")
st.sidebar.image(url)

user_input = []
for i in X:
    min_i = X[i].min()
    max_i = X[i].max()
    ans = float(st.sidebar.slider(f"Select value of {i}:", min_value = min_i, max_value = max_i))

    user_input.append(ans)

# USER INPUT SHOW
st.markdown("""
<h2> User Input Value</h2>
""",unsafe_allow_html=True)
st.write(user_input)

# MODEL PREDICTION
if st.button("Click here to Predict"):
    with st.spinner("Predicting..."):
        import time
        time.sleep(2)
        counter = 0
        model_ans = []
        model_prob = []
        for model in all_models:
            ans = model.predict([user_input])[0]
            try:
                prob = model.predict_proba([user_input]).max()
            except:
                prob = 1
            model_prob.append(prob)
            class_ans = classes[ans]
            model_ans.append(class_ans)
           
            counter += 1

        st.markdown("""
        <h2> Model Comparison </h2>
        """,unsafe_allow_html=True)

        comp_df = pd.DataFrame({"x":all_model_name, 
                                "y":model_prob,
                               'Model-Prediction':class_ans})
        
        import altair as alt
        chart = (alt.Chart(comp_df).mark_bar().encode(
            x = 'x',
            y = 'y',
            tooltip = ['x','y','Model-Prediction']
        ))

        st.altair_chart(chart, use_container_width = True)
        
        st.markdown("""
        <h2> Final Prediction </h2>
        """,unsafe_allow_html=True)
        
        data = pd.Series(model_ans)
        final_ans = data.mode().values[0]
        st.success(final_ans)

footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: #888888;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}
</style>
<div class="footer">
    <p>Made with ❤️ using Streamlit • © 2026</p>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)

