"""запити"""
from abc import ABC, abstractmethod
import numpy as np

from fetcher import *
from llm import *

class Retriever(ABC):
    @abstractmethod
    def query(self, collection):
        """Повертає відповідь на запит"""
        pass

    @abstractmethod
    def _fetch_result(self, results, collection):
        """повернення фільтрованого результату"""
        pass

    def _llm_summarize(self, txt_img_dict):
        """повернення тексту від LLM"""
        llm = LLMResponder()
        return llm.respond(txt_img_dict)

class ImageInputRetriever(Retriever):
    """клас для отрмання відповіді на запит-зображення"""
    def __init__(self, image: np.ndarray):
        self.image = image

    def query(self, collection):
        """запит для зображень"""
        results = collection.query(
        query_images=[self.image]
        )
        return self._fetch_result(results, collection)
    
    def _fetch_result(self, results, collection):
        fetсh = TextContextFetcher(collection)
        dict_fetch = fetсh.build_context(results)
        return self._llm_summarize(dict_fetch)
    
class TextInputRetriever(Retriever):
    """клас для отрмання відповіді на запит-текст"""
    def __init__(self, text: str):
        self.text = text

    def query(self, collection):
        """запит для тексту"""
        results = collection.query(
        query_texts=[self.text]
        )
        return self._fetch_result(results, collection)
    
    def _fetch_result(self, results, collection):
        fetсh = ImageContextFetcher(collection)
        dict_fetch = fetсh.build_context(results)

        return self._llm_summarize(dict_fetch)

