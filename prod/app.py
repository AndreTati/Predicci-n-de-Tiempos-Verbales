import streamlit as st
import spacy
import torch
import plotly.express as px
from utils import load_model, get_bert_embeddings, get_verb_embedding

# Cargar modelos
modelTime, modelPerson, modelNumber, id2tense, id2person, id2number, tokenizer, bert_model, nlp = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("Predicción de Tiempos Verbales")

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
    st.markdown("### Oración con verbos clickeables:")
    doc = nlp(st.session_state.oracion)
    cols = st.columns(len(doc))

    for i, token in enumerate(doc):
        matched = [(v, idx) for v, idx in st.session_state.verbos if idx == i]
        if matched:
            v_text, _ = matched[0]
            if cols[i].button(v_text, key=f"verbo_{i}"):
                st.session_state.seleccionado = v_text
        else:
            cols[i].markdown(f"<span style='color:white'>{token.text}</span>", unsafe_allow_html=True)

# Mostrar resultado del verbo seleccionado
if st.session_state.seleccionado:
    verbo = st.session_state.seleccionado
    st.markdown(f"### Resultados para **{verbo}**")
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
        probs_p = torch.softmax(logits_p, dim=1).nump
