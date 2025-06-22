import streamlit as st
from PIL import Image
import os

from utils import *
from query import *
from vector_store import *

# --- Ініціалізація Chroma ---
collection = get_chroma_collection()

# --- Заголовок ---
st.title("🔎 Multimodal RAG Search")

# --- Панель для вводу ---
col1, col2 = st.columns(2)
with col1:
    query_text = st.text_input("Введіть текстовий запит:")
with col2:
    uploaded_img = st.file_uploader("Або завантажте зображення", type=["png", "jpg", "jpeg"])

# --- Отримання запиту ---
if uploaded_img:
    img = img_np(uploaded_img)
    st.image(img, caption="Завантажене зображення", use_column_width=True)

    retriever = ImageInputRetriever(image=img)
    response = retriever.query(collection)

    st.subheader("🗣️ Відповідь LLM на основі зображення:")
    st.write(response)

elif query_text:
    retriever = TextInputRetriever(text=query_text)
    

    response = retriever.query(collection)\

    st.subheader("🗣️ Відповідь LLM на основі текстового запиту:")
    st.write(response)

else:
    st.info("Завантажте зображення або введіть текстовий запит для пошуку.")