from sqlalchemy.orm import sessionmaker
from models import Image, init_db
from query_metadata import query_metadata
from vector_search import generate_text_embedding, search_similar_images

# Initialize the database and session
engine = init_db()
Session = sessionmaker(bind=engine)
session = Session()

def combined_query(filters, query_text, top_k=5):
    # Query metadata
    metadata_results = query_metadata(filters)
    metadata_image_ids = [image.faiss_index for image in metadata_results]

    # Perform vector search
    query_embedding = generate_text_embedding(query_text)
    similar_image_ids = search_similar_images(query_embedding, top_k=top_k)

    # Combine results
    combined_results = [image_id for image_id in similar_image_ids if image_id in metadata_image_ids]
    return combined_results

# Example usage
filters = {'season': 'summer', 'baseColor': 'green'}
query_text = "long dress"
results = combined_query(filters, query_text)
print(results)

# Close the session
session.close()
