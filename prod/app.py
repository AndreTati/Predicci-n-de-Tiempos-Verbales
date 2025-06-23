import streamlit as st
import spacy
import torch
import plotly.express as px
from utils import load_model, get_bert_embeddings, get_verb_embedding

# Cargar modelos y diccionarios
modelTime, modelPerson, modelNumber, id2tense, id2person, id2number, tokenizer, bert_model, nlp = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicializar estados
if "oracion" not in st.session_state:
    st.session_state.oracion = ""
if "verbos" not in st.session_state:
    st.session_state.verbos = []
if "verbo_seleccionado" not in st.session_state:
    st.session_state.verbo_seleccionado = None

st.title("Predicción de Tiempos Verbales")

# Entrada de texto
oracion = st.text_input("Ingrese una oración en español:", value=st.session_state.oracion)

# Al cambiar la oración, reiniciar estados
if oracion != st.session_state.oracion:
    st.session_state.oracion = oracion
    st.session_state.verbo_seleccionado = None
    st.session_state.verbos = []

# Analizar al hacer clic
if st.button("Analizar"):
    doc = nlp(oracion)
    verbos = []
    for i, token in enumerate(doc):
        if token.pos_ == "VERB":
            if i > 0 and doc[i - 1].pos_ == "AUX":
                verbos.append((doc[i - 1].text, doc[i - 1].i))
            else:
                verbos.append((token.text, token.i))
    st.session_state.verbos = verbos

# Mostrar oración con botones de verbos
if st.session_state.verbos:
    st.markdown("#### Oración:")
    tokens = list(nlp(oracion))
    formatted = []
    for token in tokens:
        if (token.text, token.i) in st.session_state.verbos:
            if st.button(f"🔍 {token.text}", key=f"{token.text}_{token.i}"):
                st.session_state.verbo_seleccionado = (token.text, token.i)
            formatted.append(f"<b style='color:blue'>{token.text_with_ws}</b>")
        else:
            formatted.append(token.text_with_ws)
    st.markdown("".join(formatted), unsafe_allow_html=True)

# Mostrar resultados si se seleccionó un verbo
if st.session_state.verbo_seleccionado:
    verbo, _ = st.session_state.verbo_seleccionado
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

        logits_p = modelPerson(embPerson).detach().cpu()
        probs_p = torch.softmax(logits_p, dim=1).numpy()[0]
        labels_p = [id2person[i] for i in range(len(probs_p))]

        logits_n = modelNumber(embNumber).detach().cpu()
        probs_n = torch.softmax(logits_n, dim=1).numpy()[0]
        labels_n = [id2number[i] for i in range(len(probs_n))]

        st.markdown(f"#### Resultados para **{verbo}**")
        st.plotly_chart(px.bar(x=labels_tm, y=probs_tm, title="Tiempo - Modo", labels={"x": "Etiqueta", "y": "Probabilidad"}))
        st.plotly_chart(px.bar(x=labels_p, y=probs_p, title="Persona", labels={"x": "Etiqueta", "y": "Probabilidad"}))
        st.plotly_chart(px.bar(x=labels_n, y=probs_n, title="Número", labels={"x": "Etiqueta", "y": "Probabilidad"}))
