# app.py (FINAL)
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn import datasets
import matplotlib.pyplot as plt

# Load resources
model = joblib.load('iris_model.pkl')
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
species_names = iris.target_names

# Sidebar mode selector
mode = st.sidebar.radio("🧭 Navigation", ["Prediction", "Data Exploration"])

if mode == "Prediction":
    st.title("🌸 Iris Flower Species Predictor")
    st.markdown("""
    Enter sepal and petal measurements to predict the Iris species.
    Model: **Logistic Regression** trained on the Iris dataset.
    """)

    # Input sliders with help tooltips
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, help="Typical range: 4.3–7.9 cm")
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, help="Typical range: 2.0–4.4 cm")
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, help="Typical range: 1.0–6.9 cm")
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2, help="Typical range: 0.1–2.5 cm")

    # Predict
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    predicted_species = species_names[prediction]

    # Color-coded output
    if predicted_species == "setosa":
        st.success(f"**Predicted Species**: 🌼 *Iris setosa*")
    elif predicted_species == "versicolor":
        st.warning(f"**Predicted Species**: 🌺 *Iris versicolor*")
    else:
        st.error(f"**Predicted Species**: 🌻 *Iris virginica*")

    # Probabilities
    st.write("**Prediction Confidence:**")
    prob_df = pd.DataFrame({
        "Species": species_names,
        "Probability": probabilities
    })
    st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))

else:  # Data Exploration
    st.title("📊 Iris Dataset Exploration")
    st.markdown("Explore the distribution and relationships in the Iris dataset.")

    # Histogram
    st.subheader("📈 Feature Distribution (Histogram)")
    feature = st.selectbox("Select a feature to visualize", iris.feature_names)
    fig, ax = plt.subplots()
    ax.hist(iris_df[feature], bins=15, color='skyblue', edgecolor='black')
    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Scatter plot
    st.subheader("🔍 Feature Pair Scatter Plot")
    col1, col2 = st.columns(2)
    with col1:
        x_feature = st.selectbox("X-axis", iris.feature_names, index=0)
    with col2:
        y_feature = st.selectbox("Y-axis", iris.feature_names, index=2)

    fig2, ax2 = plt.subplots()
    colors = ['red', 'green', 'blue']
    for i, species in enumerate(species_names):
        subset = iris_df[iris_df['species'] == i]
        ax2.scatter(subset[x_feature], subset[y_feature], label=species, color=colors[i])
    ax2.set_xlabel(x_feature)
    ax2.set_ylabel(y_feature)
    ax2.legend()
    st.pyplot(fig2)