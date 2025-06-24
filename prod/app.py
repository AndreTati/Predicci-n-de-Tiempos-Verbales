import streamlit as st 
import spacy
import torch
import plotly.express as px
from utils import load_model, get_bert_embeddings, get_verb_embedding, descripcion_tiempos, descripcion_modos, descripcion_personas, descripcion_numeros

# Cambiar ancho de página
description = "Predicción de Tiempos Verbales"
st.set_page_config(page_title=description, layout="wide")

# Cargar modelos
modelTime, modelPerson, modelNumber, id2tense, id2person, id2number, tokenizer, bert_model, nlp = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title(description)

# Mantener estado
if "oracion" not in st.session_state:
    st.session_state.oracion = ""
if "verbos" not in st.session_state:
    st.session_state.verbos = []
if "seleccionado" not in st.session_state:
    st.session_state.seleccionado = None

# Entrada
oracion = st.text_input("Ingrese una oración en español:", value=st.session_state.oracion)

if st.button("Analizar"):
    st.session_state.oracion = oracion
    st.session_state.seleccionado = None
    doc = nlp(oracion)
    verbos = []
    for i, token in enumerate(doc):
        if token.pos_ == "VERB":
            if i > 0 and doc[i - 1].pos_ == "AUX":
                verbos.append((doc[i - 1].text, doc[i - 1].i))
            else:
                verbos.append((token.text, token.i))
    st.session_state.verbos = verbos

# Mostrar oración con verbos clickeables
if st.session_state.oracion and st.session_state.verbos:
    st.markdown("#### Seleccione el verbo que desea analizar:")
    doc = nlp(st.session_state.oracion)
    cols = st.columns([
        1.5 if any(idx == token.i for _, idx in st.session_state.verbos) else 1
        for token in doc
    ])

    for i, token in enumerate(doc):
        if any(idx == token.i for _, idx in st.session_state.verbos):
            if cols[i].button(f"🔍 {token.text}", key=f"btn_{i}"):
                st.session_state.seleccionado = token.text
        else:
            cols[i].markdown(f"<span style='font-size: 16px'>{token.text}</span>", unsafe_allow_html=True)

# Mostrar resultado del verbo seleccionado
if st.session_state.seleccionado:
    verbo = st.session_state.seleccionado
    st.markdown(f"#### 🔹 Verbo: {verbo}")
    inputs, hidden_states = get_bert_embeddings(st.session_state.oracion, tokenizer, bert_model)

    embTenseMood = get_verb_embedding(inputs, hidden_states, verbo, strategy="sum_all", tokenizer=tokenizer)
    embPerson = get_verb_embedding(inputs, hidden_states, verbo, strategy="second_last", tokenizer=tokenizer)
    embNumber = get_verb_embedding(inputs, hidden_states, verbo, strategy="sum_all", tokenizer=tokenizer)

    if embTenseMood is None or embPerson is None or embNumber is None:
        st.warning(f"No se pudo obtener el embedding de '{verbo}'")
    else:
        embTenseMood = embTenseMood.unsqueeze(0).to(device)
        embPerson = embPerson.unsqueeze(0).to(device)
        embNumber = embNumber.unsqueeze(0).to(device)

        logits_tm = modelTime(embTenseMood).detach().cpu()
        probs_tm = torch.softmax(logits_tm, dim=1).numpy()[0]
        labels_tm = [id2tense[i] for i in range(len(probs_tm))]
        tiempo, modo= labels_tm[probs_tm.argmax()].split("_")

        logits_p = modelPerson(embPerson).detach().cpu()
        probs_p = torch.softmax(logits_p, dim=1).numpy()[0]
        labels_p = [id2person[i] for i in range(len(probs_p))]

        logits_n = modelNumber(embNumber).detach().cpu()
        probs_n = torch.softmax(logits_n, dim=1).numpy()[0]
        labels_n = [id2number[i] for i in range(len(probs_n))]

        # Predicciones principales
        pred_tiempo = tiempo
        pred_modo = modo
        pred_persona = labels_p[probs_p.argmax()]
        pred_numero = labels_n[probs_n.argmax()]

        # Mostrar etiquetas humanas
        st.write(f"• Tiempo: {descripcion_tiempos.get(pred_tiempo, pred_tiempo)}")
        st.write(f"• Modo: {descripcion_modos.get(pred_modo, pred_modo)}")
        st.write(f"• Persona: {descripcion_personas.get(int(pred_persona), pred_persona)}")
        st.write(f"• Número: {descripcion_numeros.get(pred_numero, pred_numero)}")

        #st.plotly_chart(px.bar(x=labels_tm, y=probs_tm, text_auto=True, title="Tiempo - Modo", labels={"x": "Etiqueta", "y": "Probabilidad"}), width=300, use_container_width=False)
        #st.plotly_chart(px.bar(x=labels_p, y=probs_p, text_auto=True, title="Persona", labels={"x": "Etiqueta", "y": "Probabilidad"}), width=300, use_container_width=False)
        #st.plotly_chart(px.bar(x=labels_n, y=probs_n, text_auto=True, title="Número", labels={"x": "Etiqueta", "y": "Probabilidad"}), width=300, use_container_width=False)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(
                px.bar(x=labels_tm, y=probs_tm, text_auto=True, title="Tiempo - Modo", color_discrete_sequence=["#eb984e"], labels={"x": "Etiqueta", "y": "Probabilidad"}),
                use_container_width=True
            )
        with col2:
            st.plotly_chart(
                px.bar(x=labels_p, y=probs_p, text_auto=True, title="Persona", color_discrete_sequence=["#73c6b6"], labels={"x": "Etiqueta", "y": "Probabilidad"}),
                use_container_width=True
            )
        with col3:
            st.plotly_chart(
                px.bar(x=labels_n, y=probs_n, text_auto=True, title="Número", color_discrete_sequence=["#7fb3d5"], labels={"x": "Etiqueta", "y": "Probabilidad"}),
                use_container_width=True
            )
