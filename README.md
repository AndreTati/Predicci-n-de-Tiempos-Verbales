# Predicción de Tiempos Verbales
TRABAJO PRÁCTICO INTEGRADOR DE LA CÁTEDRA REDES NEURONALES PROFUNDAS DE INGENIERÍA EN SISTEMAS DE INFORMACIÓN 

UTN - FRM

Red Neuronal con Aplicación Web

## Descripción del problema
El reconocimiento y análisis de los tiempos verbales en una oración es una tarea compleja que requiere conocimientos gramaticales específicos. Muchas personas, tanto estudiantes como hablantes no nativos del idioma español, presentan dificultades para identificar con precisión el tiempo de los verbos que aparecen en distintos contextos. Esta dificultad puede derivar en errores de concordancia, problemas en la escritura formal y obstáculos en el aprendizaje del idioma.

El proyecto tiene el objetivo de ayudar a dichas personas a identificar el tiempo verbal, mediante el desarrollo de una red neuronal que realice procesamiento de lenguaje natural, como análisis sintáctico y clasificación de texto.
## Dataset seleccionado
El dataset seleccionado para realizar la aplicación será uno basado en datos en español del corpus AnCora. El dataset contiene 17662 oraciones, 547558 tokens y 560137 palabras sintácticas entre otros detalles que se describen en el siguiente link:
https://universaldependencies.org/treebanks/es_ancora/index.html

Para descargarlo se utilizará el siguiente repositorio de GitHub:
https://github.com/UniversalDependencies/UD_Spanish-AnCora
## Aplicación propuesta
Se propone el desarrollo de una aplicación web utilizando Streamlit que permita al usuario interactuar con la red neuronal desarrollada.

El usuario podrá ingresar una oración y el modelo ya entrenado deberá identificar el/los verbo/s dentro de la oración e indicar tiempo, modo, persona y número. Para esto se utilizará BERT que es un modelo de lenguaje basado en la arquitectura de transformers, diseñado para procesar texto de manera bidireccional, es decir, teniendo en cuenta tanto el contexto anterior como el posterior de cada palabra en una oración. Esta capacidad lo convierte en una herramienta especialmente poderosa para tareas de procesamiento del lenguaje natural (PLN) como el análisis sintáctico, la clasificación de texto, el reconocimiento de entidades, la traducción y la predicción de etiquetas gramaticales.

Dicho modelo será implementado en PyTorch, documentando el proceso de selección del modelo, entrenamiento y testeo.

La aplicación será desplegada en Streamlit Cloud pudiendo ser accedida por cualquier persona que lo requiera.
