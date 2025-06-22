### швидка перевірка
import pytest
import numpy as np
from PIL import Image
from query import TextInputRetriever, ImageInputRetriever
import chromadb

@pytest.fixture(scope="module")
def collection():
    client = chromadb.PersistentClient(path="chroma_langchain_db")
    return client.get_collection(name="multimodal_collection")

@pytest.fixture(scope="module")
def sample_image():
    img = Image.open("example.jpg").convert("RGB")
    return np.array(img)

def test_text_input_retriever(collection):
    query_text = "Impact of AI on data centers"
    retriever = TextInputRetriever(text=query_text)
    result = retriever.query(collection)

    assert isinstance(result, dict)
    assert "image_path" in result
    assert "text" in result
    # Можна додати ще перевірку, що хоч щось не None
    assert result["text"] is None or isinstance(result["text"], str)

def test_image_input_retriever(collection, sample_image):
    retriever = ImageInputRetriever(image=sample_image)
    result = retriever.query(collection)

    assert isinstance(result, dict)
    assert "image_path" in result
    assert "text" in result
    assert result["image_path"] is None or isinstance(result["image_path"], str)
