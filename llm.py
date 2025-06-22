# інтеграція retrieve з llm
from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API")

class LLMResponder:
    def __init__(self):
        self.client = genai.Client(api_key=api_key)
        # self.model = self.client.models.get("gemini-2.5-flash")

    def respond(self, text_img_dict):
        filepath = text_img_dict['image_path']
        text = text_img_dict['text']

        client = genai.Client(api_key=api_key)

        with open(filepath, 'rb') as f:
            image_bytes = f.read()

        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=[
                types.Part(text=(
                    "You are given a piece of text and an image. "
                    "Based on both, provide a clear, structured, and factual response\n\n"
                    "Context:\n"
                    f"{text}\n\n"
                    "Use only the information available in the context and image. If you cannot answer based on that, say so honestly."
                    )),
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type='image/jpeg',
                    ),
            ]
        )
        text_img_dict['text'] = response.text 
        return text_img_dict