import os
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship, sessionmaker, declarative_base

Base = declarative_base()

class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True)
    image_path = Column(String, nullable=False)
    base_color = Column(String, nullable=False)
    season = Column(String, nullable=False)
    article_type = Column(String, nullable=False)
    faiss_index = Column(Integer, nullable=True)  # Column to store FAISS index position

    # Index for frequently queried columns
    __table_args__ = (
        Index('ix_image_faiss_index', 'faiss_index'),
    )

class Attribute(Base):
    __tablename__ = 'attributes'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    value = Column(String, nullable=False)
    __table_args__ = (UniqueConstraint('name', 'value', name='_name_value_uc'),)

class ItemAttribute(Base):
    __tablename__ = 'item_attributes'
    image_id = Column(Integer, ForeignKey('images.id'), primary_key=True)
    attribute_id = Column(Integer, ForeignKey('attributes.id'), primary_key=True)
    image = relationship("Image", back_populates="attributes")
    attribute = relationship("Attribute", back_populates="items")

Image.attributes = relationship("ItemAttribute", back_populates="image", cascade="all, delete-orphan")
Attribute.items = relationship("ItemAttribute", back_populates="attribute", cascade="all, delete-orphan")

def init_db():
    db_dir = os.path.abspath('../data')
    db_path = os.path.join(db_dir, 'fashion_items.db')
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)
    return engine

engine = init_db()
Session = sessionmaker(bind=engine)
session = Session()
