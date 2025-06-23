"""функціонал для об'єднання з retrieve"""
from abc import ABC, abstractmethod
from typing import Dict
from utils import file_by_prefix

class AbstractContextFetcher(ABC):
    def __init__(self, collection):
        self.collection = collection

    @abstractmethod
    def build_context(self, query_result: Dict) -> Dict:
        """Повертає пов'язаний контекст"""
        pass

class ImageContextFetcher(AbstractContextFetcher):
    """пов’язані тексти на основі зображення"""
    def build_context(self, query_result: Dict) -> Dict:
        first_metadata = query_result['metadatas'][0][0]
        results_imgs = self.collection.get(
            where={
                "$and": [
                    {"source": first_metadata.get('source')},
                    {"type": "image"}
                    ]
            }
        )

        filepath = file_by_prefix(results_imgs['ids'][0]) # на перспективу з можливістю більшої кількості зображень
        text = query_result['documents'][0][0] if query_result['documents'][0] else None
   
        return {"image_path": filepath, "text": text}

class TextContextFetcher(AbstractContextFetcher):
    """пов’язанні(((в однині))) зображення(((в однині))) на основі текстового запиту"""
    def build_context(self, query_result: Dict) -> Dict:
        docs = query_result.get('documents', [[]])[0]  # перший список документів
        doc = next((doc for doc in docs if doc is not None), None)
        
        img_id = query_result['ids'][0][0]
        filepath = file_by_prefix(img_id)

        return {"image_path": filepath, "text": doc}
