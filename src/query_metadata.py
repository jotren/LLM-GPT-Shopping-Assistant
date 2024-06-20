from sqlalchemy.orm import sessionmaker
from models import Attribute, ItemAttribute, Image, init_db

# Initialize the database and session
engine = init_db()
Session = sessionmaker(bind=engine)
session = Session()

def query_metadata(filters):
    query = session.query(Image)
    for attr_name, attr_value in filters.items():
        query = query.join(ItemAttribute).join(Attribute).filter(Attribute.name == attr_name, Attribute.value == attr_value)
    results = query.all()
    return results

# Example usage
filters = {'season': 'summer', 'baseColor': 'green'}
metadata_results = query_metadata(filters)

# Close the session
session.close()
