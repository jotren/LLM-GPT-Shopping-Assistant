import os
import clip
import torch
from PIL import Image as PILImage
import pandas as pd
import numpy as np
import faiss
from sqlalchemy.orm import sessionmaker
from models import Image as ImageModel, Attribute, ItemAttribute, init_db

# Initialize the database and session
engine = init_db()
Session = sessionmaker(bind=engine)
session = Session()

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Path to the folder containing images and the metadata CSV file
images_folder = os.path.join("../", "data", "raw", "images")
metadata_file = os.path.join("../", "data", "raw", "styles.csv")

# Load metadata
metadata = pd.read_csv(metadata_file)

# Check for missing columns
required_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColor', 'season', 'use']
missing_columns = [col for col in required_columns if col not in metadata.columns]
if missing_columns:
    print(f"Missing columns in metadata: {missing_columns}")
    exit(1)

# Function to generate embedding for an image
def generate_embedding(image_path):
    image = PILImage.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
    return embedding.cpu().numpy()

# Function to get or create attribute id
def get_or_create_attribute(session, name, value):
    instance = session.query(Attribute).filter_by(name=name, value=value).first()
    if not instance:
        instance = Attribute(name=name, value=value)
        session.add(instance)
        session.commit()
    return instance.id

# Define the dimension of the embeddings
d = 512  # Embedding dimension for CLIP ViT-B/32

# Collect embeddings and metadata
embeddings = []
faiss_index = 0
for idx, row in metadata.iterrows():
    image_path = os.path.join(images_folder, row['image'])
    image_path = os.path.normpath(image_path)  # Normalize the path to ensure consistency
    if os.path.exists(image_path):
        embedding = generate_embedding(image_path)
        embeddings.append(embedding)
        
        # Insert image metadata into the database
        image_instance = ImageModel(
            image_path=image_path,
            base_color=row.get('baseColor'),
            season=row.get('season'),
            article_type=row.get('articleType'),
            faiss_index=faiss_index
        )
        session.add(image_instance)
        session.commit()
        
        faiss_index += 1
        
        # Insert attribute data and mapping
        for attribute in required_columns:
            value = row.get(attribute)
            attribute_id = get_or_create_attribute(session, attribute, value)
            item_attribute = ItemAttribute(image_id=image_instance.id, attribute_id=attribute_id)
            session.add(item_attribute)
        session.commit()

# Convert embeddings to numpy array
embeddings = np.array(embeddings)

# Create FAISS index
nlist = 100  # Number of clusters
quantizer = faiss.IndexFlatL2(d)  # Base quantizer
index = faiss.IndexIVFPQ(quantizer, d, nlist, 8, 8)  # IVF index with PQ

# Train the index with the embeddings
index.train(embeddings)  # This step learns the structure of the vector space

# Add embeddings to the index
index.add(embeddings)

# Save the FAISS index
faiss.write_index(index, '../data/faiss_large.index')

# Save the mapping of image ids to a file
image_ids = [image.id for image in session.query(Image).all()]
np.save('../data/image_ids.npy', image_ids)

# Close the session
session.close()
