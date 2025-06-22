import os
import numpy as np
from PIL import Image

def img_np(uploaded_img):
    """перетворення зображення в масив numpy"""
    image = Image.open(uploaded_img).convert("RGB")
    
    return np.array(image)

def file_by_prefix(prefix, dir='downloaded_images'):
    """файл за ім'ям, тобто без ext"""
    for filename in os.listdir(dir):
        if filename.startswith(prefix):
            return os.path.join(dir, filename)
    return None