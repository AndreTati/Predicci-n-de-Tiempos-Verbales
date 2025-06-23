import streamlit as st
import spacy
import torch
import plotly.express as px
from model import VerbClassifier
from utils import load_model, analizar_oracion, get_bert_embeddings, get_verb_embedding

modelTime, modelPerson, modelNumber,id2tense, id2person, id2number,tokenizer, bert_model, nlp = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nlp = spacy.load("es_core_news_md")

st.title("Predicción de Tiempos Verbales")
oracion = st.text_input("Ingrese una oración en español:")

if st.button("Analizar"):
    if oracion.strip() == "":
        st.warning("Por favor, ingrese una oración.")
    else:
        # Detectar verbos con spaCy
        doc = nlp(oracion)
        verbos = []

        for i, token in enumerate(doc):
            if token.pos_ == "VERB":
                if i > 0 and doc[i - 1].pos_ == "AUX":
                    aux_token = doc[i - 1]
                    verbos.append((aux_token.text, aux_token.i))
                else:
                    verbos.append((token.text, token.i))

        # Mostrar oración con verbos en azul
        tokens = [tok.text_with_ws for tok in doc]
        colored = "".join([
            f"<span style='color:blue;font-weight:bold'>{tok}</span>" if tok.strip() in verbos else tok
            for tok in tokens
        ])
        st.markdown(colored, unsafe_allow_html=True)

        # Botón por verbo
        for verbo in verbos:
            if st.button(f"🔍 Analizar '{verbo}'"):
                inputs, hidden_states = get_bert_embeddings(oracion, tokenizer, bert_model)
                embTenseMood = get_verb_embedding(inputs, hidden_states, verbo, strategy="sum_all", tokenizer=tokenizer)
                embPerson = get_verb_embedding(inputs, hidden_states, verbo, strategy="second_last", tokenizer=tokenizer)
                embNumber = get_verb_embedding(inputs, hidden_states, verbo, strategy="sum_all", tokenizer=tokenizer)
                if emb is None:
                    st.warning(f"No se pudo obtener el embedding de '{verbo}'")
                    continue

                embTenseMood = embTenseMood.unsqueeze(0).to(device)
                embPerson = embPerson.unsqueeze(0).to(device)
                embNumber = embNumber.unsqueeze(0).to(device)

                # Predicción
                logits_tm = modelTime(embTenseMood).detach().cpu()
                probs_tm = torch.softmax(logits_tm, dim=1).numpy()[0]
                labels_tm = [id2tense[i] for i in range(len(probs_tm))]

                logits_p = modelPerson(embPerson).detach().cpu()
                probs_p = torch.softmax(logits_p, dim=1).numpy()[0]
                labels_p = [id2person[i] for i in range(len(probs_p))]

                logits_n = modelNumber(embNumber).detach().cpu()
                probs_n = torch.softmax(logits_n, dim=1).numpy()[0]
                labels_n = [id2number[i] for i in range(len(probs_n))]

                # Mostrar resultados
                st.markdown(f"### Resultados para **{verbo}**")

                st.plotly_chart(px.bar(x=labels_tm, y=probs_tm, title="Tiempo - Modo", labels={"x": "Etiqueta", "y": "Probabilidad"}))
                st.plotly_chart(px.bar(x=labels_p, y=probs_p, title="Persona", labels={"x": "Etiqueta", "y": "Probabilidad"}))
                st.plotly_chart(px.bar(x=labels_n, y=probs_n, title="Número", labels={"x": "Etiqueta", "y": "Probabilidad"}))
                
        """resultados = analizar_oracion(oracion, nlp, modelTime, modelPerson, modelNumber, id2tense, id2person, id2number, tokenizer, bert_model)
        if not resultados:
            st.info("No se detectaron verbos o no se pudieron procesar.")
        else:
            for r in resultados:
                st.markdown(f"### 🔹 Verbo: {r['verbo']}")
                st.write(f"• Tiempo: {r['tiempo']}")
                st.write(f"• Modo: {r['modo']}")
                st.write(f"• Persona: {r['persona']}")
                st.write(f"• Número: {r['numero']}")"""



