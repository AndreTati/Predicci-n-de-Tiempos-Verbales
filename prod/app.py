import streamlit as st
import spacy
import torch
import plotly.express as px
from model import VerbClassifier
from utils import load_model, get_bert_embeddings, get_verb_embedding

# Cargar modelos y diccionarios
modelTime, modelPerson, modelNumber, id2tense, id2person, id2number, tokenizer, bert_model, nlp = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Título
st.title("Predicción de Tiempos Verbales")

# Entrada
oracion = st.text_input("Ingrese una oración en español:")

# Limpiar sesión si cambia la oración
if "prev_oracion" not in st.session_state:
    st.session_state["prev_oracion"] = ""

if oracion != st.session_state["prev_oracion"]:
    st.session_state["selected_verbo"] = None
    st.session_state["selected_index"] = None
    st.session_state["prev_oracion"] = oracion

if st.button("Analizar") and oracion.strip() != "":
    doc = nlp(oracion)
    verbos = []

    for i, token in enumerate(doc):
        if token.pos_ == "VERB":
            if i > 0 and doc[i - 1].pos_ == "AUX":
                verbos.append((doc[i - 1].text, doc[i - 1].i))  # Usar AUX
            else:
                verbos.append((token.text, token.i))

    # Mostrar la oración como botones para los verbos
    st.markdown("### Hacé clic sobre un verbo para analizarlo:")
    cols = st.columns(len(doc))
    for i, token in enumerate(doc):
        token_text = token.text_with_ws
        if any(i == v_i for (_, v_i) in verbos):
            with cols[i]:
                if st.button(token.text, key=f"verbo_{i}"):
                    st.session_state["selected_verbo"] = token.text
                    st.session_state["selected_index"] = i
        else:
            cols[i].markdown(token_text)

# Mostrar análisis si hay verbo seleccionado
if "selected_verbo" in st.session_state and st.session_state["selected_verbo"]:
    verbo = st.session_state["selected_verbo"]
    st.markdown(f"### Resultados para **{verbo}**")

    inputs, hidden_states = get_bert_embeddings(oracion, tokenizer, bert_model)
    embTenseMood = get_verb_embedding(inputs, hidden_states, verbo, strategy="sum_all", tokenizer=tokenizer)
    embPerson = get_verb_embedding(inputs, hidden_states, verbo, strategy="second_last", tokenizer=tokenizer)
    embNumber = get_verb_embedding(inputs, hidden_states, verbo, strategy="sum_all", tokenizer=tokenizer)

    if embTenseMood is not None and embPerson is not None and embNumber is not None:
        embTenseMood = embTenseMood.unsqueeze(0).to(device)
        embPerson = embPerson.unsqueeze(0).to(device)
        embNumber = embNumber.unsqueeze(0).to(device)

        # Predicciones
        logits_tm = modelTime(embTenseMood).detach().cpu()
        probs_tm = torch.softmax(logits_tm, dim=1).numpy()[0]
        labels_tm = [id2tense[i] for i in range(len(probs_tm))]

        logits_p = modelPerson(embPerson).detach().cpu()
        probs_p = torch.softmax(logits_p, dim=1).numpy()[0]
        labels_p = [id2person[i] for i in range(len(probs_p))]

        logits_n = modelNumber(embNumber).detach().cpu()
        probs_n = torch.softmax(logits_n, dim=1).numpy()[0]
        labels_n = [id2number[i] for i in range(len(probs_n))]

        # Gráficos
        st.plotly_chart(px.bar(x=labels_tm, y=probs_tm, title="Tiempo - Modo", labels={"x": "Etiqueta", "y": "Probabilidad"}))
        st.plotly_chart(px.bar(x=labels_p, y=probs_p, title="Persona", labels={"x": "Etiqueta", "y": "Probabilidad"}))
        st.plotly_chart(px.bar(x=labels_n, y=probs_n, title="Número", labels={"x": "Etiqueta", "y": "Probabilidad"}))
    else:
        st.warning(f"No se pudo obtener el embedding de '{verbo}'")
