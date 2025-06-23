import streamlit as st

from utils import *
from query import *
from vector_store import *

# --- Ініціалізація Chroma ---
collection = get_chroma_collection()

# --- Ініціалізація стану чату ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processing" not in st.session_state:
    st.session_state.processing = False

# --- Заголовок ---
st.title("Multimodal RAG (The Batch)")

# --- Відображення історії чату ---
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["type"] == "text":
                st.write(message["content"])
            elif message["type"] == "image":
                st.image(message["content"], caption="Завантажене зображення")
            elif message["type"] == "response":
                # Розділяємо відповідь на текст та зображення
                if isinstance(message["content"], dict):
                    if "text" in message["content"] and message["content"]["text"]:
                        st.write("**Текстова відповідь:**")
                        st.write(message["content"]["text"])
                    
                    if "image_path" in message["content"] and message["content"]["image_path"]:
                        st.write("**Пов'язане зображення:**")
                        try:
                            st.image(message["content"]["image_path"])
                        except:
                            st.write(f"Шлях до зображення: {message['content']['image_path']}")
                else:
                    st.write(message["content"])

# --- Обробка надісланого повідомлення ---
def process_message():
    st.session_state.processing = True
    
    try:
        # Обробка зображення
        if uploaded_file:
            img = img_np(uploaded_file)
            
            # Додаємо зображення до чату
            st.session_state.messages.append({
                "role": "user",
                "type": "image",
                "content": img
            })
            
            # Обробляємо запит
            retriever = ImageInputRetriever(image=img)
            response = retriever.query(collection)
            
            # Додаємо відповідь
            st.session_state.messages.append({
                "role": "assistant",
                "type": "response",
                "content": response
            })
        
        # Обробка текстового запиту
        elif user_input:
            # Додаємо текстове повідомлення до чату
            st.session_state.messages.append({
                "role": "user",
                "type": "text",
                "content": user_input
            })
            
            # Обробляємо запит
            retriever = TextInputRetriever(text=user_input)
            response = retriever.query(collection)
            
            # Додаємо відповідь
            st.session_state.messages.append({
                "role": "assistant",
                "type": "response",
                "content": response
            })
        
        # Очищаємо поля вводу
        st.session_state.user_input = ""
        
    except Exception as e:
        st.error(f"Помилка при обробці запиту: {str(e)}")
    
    finally:
        st.session_state.processing = False

# --- Область вводу повідомлень ---
with st.container():
    st.divider()
    
    # Створюємо колонки для вводу
    input_col1, input_col2, send_col = st.columns([3, 2, 1])

    with input_col1:
        user_input = st.text_input(
            "Введіть ваш запит:", 
            key="user_input",
            placeholder="Напишіть ваше питання...",
            disabled=st.session_state.processing
        )

    with input_col2:
        uploaded_file = st.file_uploader(
            "Або завантажте зображення:",
            type=["png", "jpg", "jpeg"],
            key="file_uploader",
            disabled=st.session_state.processing
        )

    with send_col:
        send_button = st.button(
            "📤 Надіслати", 
            disabled=st.session_state.processing or (not user_input and not uploaded_file),
            use_container_width=True
        )

    # Кнопка очищення чату
    if st.button("🗑️ Очистити чат"):
        st.session_state.messages = []
        st.rerun()

# Обробка натискання кнопки
if send_button:
    process_message()
    st.rerun()