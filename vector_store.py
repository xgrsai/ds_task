import chromadb
import streamlit as st


@st.cache_resource
def get_chroma_collection():
    """створює і кешує підключення до локальної Chroma-векторної бази і повертає з неї потрібну колекцію"""
    client = chromadb.PersistentClient(path="chroma_langchain_db")
    return client.get_collection(name="multimodal_collection")