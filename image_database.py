import numpy as np
import wikipedia
from deepface import DeepFace
import requests
from PIL import Image
from io import BytesIO


def get_image_from_wiki(pageid=18404696, img_path=None):
    images = wikipedia.page(pageid=pageid).images
    url = images[0]
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    img = Image.open(BytesIO(response.content))
    if img_path is not None:
        img.save(img_path)
    
    return img



def get_embedding(img_path):
    embedding = DeepFace.represent(img_path)
    return embedding[0]['embedding']


if __name__ == "__main__":
    embedding = DeepFace.represent("t1.jpg")
    print(f"Debug : embedding : {embedding[0]}")
    get_embedding() 
