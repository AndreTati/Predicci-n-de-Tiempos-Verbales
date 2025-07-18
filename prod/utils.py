import torch
from transformers import BertTokenizer, BertModel
import spacy
import pickle
import os
from model import VerbClassifier

def load_model():
    # Cargar diccionarios de etiquetas
    with open("prod/label_dicts.pkl", "rb") as f:
        label_dicts = pickle.load(f)

    id2tense = label_dicts["tense"]
    #id2mood = label_dicts["mood"]
    id2person = label_dicts["person"]
    id2number = label_dicts["number"]

    # Crear modelos
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelTime = VerbClassifier(768, len(id2tense))
    #modelMood = VerbClassifier(768, len(id2mood))
    modelPerson = VerbClassifier(768, len(id2person))
    modelNumber = VerbClassifier(768, len(id2number))

    # Cargar pesos
    modelTime.load_state_dict(torch.load("prod/modelTime.pth", map_location=device))
    #modelMood.load_state_dict(torch.load("prod/modelMood.pth", map_location=device))
    modelPerson.load_state_dict(torch.load("prod/modelPerson.pth", map_location=device))
    modelNumber.load_state_dict(torch.load("prod/modelNumber.pth", map_location=device))

    # Modo evaluación
    modelTime.eval()
    #modelMood.eval()
    modelPerson.eval()
    modelNumber.eval()


    # Tokenizador y modelo BERT
    #tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
    #bert_model = BertModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased", output_hidden_states=True, trust_remote_code=True)
    repo_id = "tatiduran/bertmodel"
    subfolder = "bert_model"  # si corresponde

    tokenizer = BertTokenizer.from_pretrained(repo_id, subfolder=subfolder)
    bert_model = BertModel.from_pretrained(repo_id, output_hidden_states=True, subfolder=subfolder)
    bert_model.eval()

    # Cargar spaCy

    nlp = spacy.load("es_core_news_md")
    return (modelTime, modelPerson, modelNumber,
            id2tense, id2person, id2number,
            tokenizer, bert_model, nlp)

def get_bert_embeddings(sentence, tokenizer, bert_model):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return inputs, outputs.hidden_states

def get_verb_embedding(inputs, hidden_states, verb, strategy, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    verb_subtokens = tokenizer.tokenize(verb)
    for i in range(len(tokens) - len(verb_subtokens) + 1):
        if tokens[i:i + len(verb_subtokens)] == verb_subtokens:
            indices = list(range(i, i + len(verb_subtokens)))
            break
    else:
        return None
    if strategy == "second_last":
        return torch.stack([hidden_states[-2][0][i] for i in indices], dim=0).mean(dim=0)
    elif strategy == "sum_last4":
        return sum(hidden_states[-i][0][indices].mean(dim=0) for i in range(1, 5))
    elif strategy == "concat_last4":
        return torch.cat([hidden_states[-i][0][indices].mean(dim=0) for i in range(1, 5)], dim=-1)
    elif strategy == "sum_all":
        return sum(hidden_states[i][0][indices].mean(dim=0) for i in range(1, 13))
    else:
        return None

# Diccionarios descriptivos
descripcion_tiempos = {
    "Pres": "Presente",
    "Past": "Pasado",
    "Imp": "Imperfecto",
    "Fut": "Futuro"
}

descripcion_modos = {
    "Ind": "Indicativo",
    "Sub": "Subjuntivo"
}

descripcion_personas = {
    1: "Primera",
    2: "Segunda",
    3: "Tercera"
}

descripcion_numeros = {
    "Sing": "Singular",
    "Plur": "Plural"
}

def detectar_verbos_spacy(nlp, oracion):
    doc = nlp(oracion)
    verbos = []
    for i, token in enumerate(doc):
        if token.pos_ == "VERB":
            if i > 0 and doc[i - 1].pos_ == "AUX":
                verbos.append((doc[i - 1].text, doc[i - 1].i))
            else:
                verbos.append((token.text, token.i))
    return verbos

def analizar_verbo(oracion, tokenizer, bert_model, modelTime, modelPerson, modelNumber,
                  id2tense, id2person, id2number, verbo, device):
    inputs, hidden_states = get_bert_embeddings(oracion, tokenizer, bert_model)

    embTenseMood = get_verb_embedding(inputs, hidden_states, verbo, strategy="sum_all", tokenizer=tokenizer)
    embPerson = get_verb_embedding(inputs, hidden_states, verbo, strategy="second_last", tokenizer=tokenizer)
    embNumber = get_verb_embedding(inputs, hidden_states, verbo, strategy="sum_all", tokenizer=tokenizer)

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

    return pred_tiempo, pred_modo, pred_persona, pred_numero, probs_tm, labels_tm, probs_p, labels_p, probs_n, labels_n

def analizar_oracion(oracion, nlp, modelTime, modelPerson, modelNumber, id2tense, id2person, id2number, tokenizer, bert_model):

    doc = nlp(oracion)
    verbos = [(token.text, token.i) for token in doc if token.pos_ == "VERB"]
    resultados = []

    inputs, hidden_states = get_bert_embeddings(oracion, tokenizer, bert_model)
    for verbo, _ in verbos:
        emb_time = get_verb_embedding(inputs, hidden_states, verbo, "sum_all", tokenizer)
        #emb_mood = get_verb_embedding(inputs, hidden_states, verbo, "sum_last4", tokenizer)
        emb_person = get_verb_embedding(inputs, hidden_states, verbo, "second_last", tokenizer)
        emb_number = get_verb_embedding(inputs, hidden_states, verbo, "sum_all", tokenizer)

        if None in [emb_time, emb_person, emb_number]:
            continue

        tiempo = id2tense[modelTime(emb_time.unsqueeze(0)).argmax(dim=1).item()]
        #modo = id2mood[modelMood(emb_mood.unsqueeze(0)).argmax(dim=1).item()]
        persona = id2person[modelPerson(emb_person.unsqueeze(0)).argmax(dim=1).item()]
        numero = id2number[modelNumber(emb_number.unsqueeze(0)).argmax(dim=1).item()]
        tiempo, modo = tiempo.split("_")

        resultados.append({
            "verbo": verbo,
            "tiempo": tiempo,
            "modo": modo,
            "persona": persona,
            "numero": numero
        })

    return resultados