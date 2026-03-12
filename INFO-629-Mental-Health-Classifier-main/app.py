import streamlit as st
from backend.backend import *

st.title("Mental Health Classifier")

st.write("Student Project for INFO-629 class")

options = [classifier.value for classifier in MentalHealthClassifiers]
classifier_option = st.selectbox(
    "Select a classifier:",
    options)

text_to_classify = st.text_area("Enter text to classify:")

if st.button("Classify"):
    if text_to_classify:
        result = classify_mental_health(MentalHealthClassifiers(classifier_option), text_to_classify)
        st.write(f"Classification Result: {result.value}")
    else:
        st.write("Please enter some text to classify.")