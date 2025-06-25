import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv

# Configuración inicial
st.set_page_config(page_title="Chatbot Minería", layout="centered")
st.title("🤖 Chatbot de Minería")
st.write("Haz una pregunta sobre los temas de la asignatura de minería:")

# Cargar variables de entorno
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Función cacheada para cargar documentos
@st.cache_resource
def cargar_documentos():
    return SimpleDirectoryReader("docs_mineria").load_data()

# Función cacheada para construir el índice
@st.cache_resource
def construir_indice():
    documentos = cargar_documentos()
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-3.5-turbo", api_key=api_key)  # puedes usar "gpt-4o" si lo tienes habilitado
    )
    return VectorStoreIndex.from_documents(documentos, service_context=service_context)

# Crear índice una sola vez
indice = construir_indice()
query_engine = indice.as_query_engine()

# Entrada del usuario
pregunta = st.text_input("Escribe tu pregunta")

if pregunta:
    with st.spinner("Pensando..."):
        respuesta = query_engine.query(pregunta)
        st.success("Respuesta:")
        st.write(str(respuesta))
else:
    st.info("Por favor, ingresa una pregunta.")