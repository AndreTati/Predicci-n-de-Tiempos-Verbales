import torch
import streamlit as st
from model import VerbClassifier
from utils import load_model, analizar_oracion

modelTime, modelMood, modelPerson, modelNumber,id2tense, id2mood, id2person, id2number,tokenizer, bert_model, nlp = load_model()

st.title("Predicción de Tiempos Verbales en Español")
oracion = st.text_input("Ingrese una oración en español:")

if st.button("Analizar"):
    if oracion.strip() == "":
        st.warning("Por favor, ingrese una oración.")
    else:
        resultados = analizar_oracion(oracion, nlp, modelTime, modelMood, modelPerson, modelNumber, id2tense, id2mood, id2person, id2number, tokenizer, bert_model)
        if not resultados:
            st.info("No se detectaron verbos o no se pudieron procesar.")
        else:
            for r in resultados:
                st.markdown(f"### 🔹 Verbo: {r['verbo']}")
                st.write(f"• Tiempo: {r['tiempo']}")
                st.write(f"• Modo: {r['modo']}")
                st.write(f"• Persona: {r['persona']}")
                st.write(f"• Número: {r['numero']}")