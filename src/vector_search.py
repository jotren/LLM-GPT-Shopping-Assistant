import numpy as np
import hnswlib
import clip
import torch
import os
import psutil
from IPython.display import HTML, display
from PIL import Image

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load HNSWlib index and image ids
hnsw_index_path = '../data/processed/hnsw_index.bin'
image_ids_path = '../data/processed/image_ids.npy'
image_folder_path = '../data/raw/images'

# Load the index
hnsw_index = hnswlib.Index(space='l2', dim=512)
hnsw_index.load_index(hnsw_index_path)
image_ids = np.load(image_ids_path)

def generate_text_embedding(text, model, device):
    text_input = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_input).cpu().numpy().flatten()
    # Normalize the embedding
    text_embedding = text_embedding / np.linalg.norm(text_embedding)
    return text_embedding

def search_similar_images(index, query_embedding, image_ids, top_k=5):
    labels, distances = index.knn_query(query_embedding, k=top_k)
    similar_image_ids = [image_ids[i] for i in labels[0]]
    return similar_image_ids

def query_images(query_text, top_k=5):
    """
    Query the database with an unstructured text query and return a dictionary with query text as the key
    and a list of image paths as the value.
    
    Args:
    query_text (str): Unstructured text query describing the clothing items.
    top_k (int): Number of top results to return.
    
    Returns:
    dict: A dictionary with query text as the key and a list of image paths as the value.
    """
    query_embedding = generate_text_embedding(query_text, model, device)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    similar_image_ids = search_similar_images(hnsw_index, query_embedding, image_ids, top_k)
    image_paths = [os.path.join(image_folder_path, f"{image_id}.jpg") for image_id in similar_image_ids]
    return {query_text: image_paths}



def display_images(image_paths_dict, max_images=5):
    """
    Display images with item names at the top in a row format.
    
    Args:
    image_paths_dict (dict): Dictionary with item names as keys and lists of image paths as values.
    max_images (int): Maximum number of images to display per item.
    """
    html_content = '<table>'
    
    for item_name, image_paths in image_paths_dict.items():
        html_content += '<tr>'
        # Add the item name
        html_content += f'<td><strong>{item_name}</strong></td>'
        
        # Add the images
        html_content += '<td>'
        for image_path in image_paths[:max_images]:
            if os.path.exists(image_path):
                html_content += f'<img src="{image_path}" style="max-width: 150px; margin: 5px;" />'
            else:
                html_content += f'<p>Image {image_path} not found</p>'
        html_content += '</td>'
        
        html_content += '</tr>'
    
    html_content += '</table>'
    display(HTML(html_content))

# Monitor system resources
def monitor_resources():
    process = psutil.Process(os.getpid())
    print(f"Memory Usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")
    print(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")
