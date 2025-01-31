import streamlit as st
import os
from pymilvus import connections, Collection, utility
import fitz
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import tempfile
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)

# Milvus connection details
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "zanobab123"

# Model IDs
DENSE_MODEL_ID = 'intfloat/multilingual-e5-large-instruct'
SPARSE_MODEL_ID = 'naver/splade-cocondenser-ensembledistil'

def create_collection(collection_name: str) -> None:
    """Create a new Milvus collection with the specified schema."""
    from pymilvus import CollectionSchema, FieldSchema, DataType
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="embedding_raw", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=65535)
    ]
    
    schema = CollectionSchema(
        fields=fields,
        description="Hybrid collection schema",
        enable_dynamic_field=True
    )
    
    Collection(name=collection_name, schema=schema, shards_num=2)
    st.success(f"Created collection: {collection_name}")

def build_indexes(collection_name: str):
    """Build both dense and sparse indexes for the collection."""
    collection = Collection(name=collection_name)
    
    # Dense vector index
    collection.create_index(
        field_name="dense_vector",
        index_params={
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
    )
    
    # Sparse vector index
    collection.create_index(
        field_name="sparse_vector",
        index_params={
            "metric_type": "IP",
            "index_type": "SPARSE_INVERTED_INDEX",
            "params": {"drop_ratio_build": 0.2}
        }
    )
    
    st.success("Built indexes successfully")

@st.cache_resource
def load_models():
    """Load and cache the embedding models."""
    dense_model = SentenceTransformer(DENSE_MODEL_ID)
    sparse_tokenizer = AutoTokenizer.from_pretrained(SPARSE_MODEL_ID)
    sparse_model = AutoModelForMaskedLM.from_pretrained(SPARSE_MODEL_ID)
    return dense_model, sparse_tokenizer, sparse_model

def process_pdf(file, document_id: str, dense_model, sparse_tokenizer, sparse_model):
    """Process a PDF file and generate embeddings."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Extract text from PDF
        doc = fitz.open(tmp_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        
        # Split text into chunks
        chunk_size = 600
        overlap = 100
        split_docs = []
        
        for i in range(0, len(full_text), chunk_size - overlap):
            chunk = full_text[i:i + chunk_size]
            if chunk:
                split_docs.append(chunk.strip())
        
        # Generate embeddings
        dense_embeddings = dense_model.encode(split_docs)
        
        sparse_embeddings = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sparse_model.to(device)
        
        for doc in split_docs:
            inputs = sparse_tokenizer(doc, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = sparse_model(**inputs).logits
            relu_log = torch.log(1 + torch.relu(logits))
            sparse_vec = relu_log.max(dim=1).values.squeeze().cpu()
            sparse_dict = {int(k): float(v) for k, v in enumerate(sparse_vec.tolist()) if v > 0}
            sparse_embeddings.append(sparse_dict)
        
        # Insert data into Milvus
        collection = Collection(name=COLLECTION_NAME)
        data = [
            dense_embeddings.tolist(),
            sparse_embeddings,
            split_docs,
            [document_id] * len(split_docs),
            [file.name] * len(split_docs)
        ]
        
        collection.insert(data)
        st.success(f"Successfully processed and inserted {file.name}")
        
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)

def main():
    st.title("Document Upload System")
    st.write("Upload PDF documents to be processed and indexed for searching.")
    
    # Connect to Milvus
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        st.success("Connected to Milvus successfully")
    except Exception as e:
        st.error(f"Failed to connect to Milvus: {str(e)}")
        return
    
    # Check if collection exists
    collection_exists = utility.has_collection(COLLECTION_NAME)
    
    if not collection_exists:
        st.warning(f"Collection {COLLECTION_NAME} does not exist.")
        if st.button("Create Collection"):
            create_collection(COLLECTION_NAME)
            build_indexes(COLLECTION_NAME)
    else:
        st.info(f"Using existing collection: {COLLECTION_NAME}")
    
    # Load models
    with st.spinner("Loading models..."):
        dense_model, sparse_tokenizer, sparse_model = load_models()
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, file in enumerate(uploaded_files):
            status_text.text(f"Processing {file.name}...")
            document_id = os.path.splitext(file.name)[0]
            
            try:
                process_pdf(
                    file,
                    document_id,
                    dense_model,
                    sparse_tokenizer,
                    sparse_model
                )
                progress = (idx + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                continue
        
        status_text.text("All files processed!")
        progress_bar.progress(1.0)
        
        # Show collection statistics
        collection = Collection(name=COLLECTION_NAME)
        st.write(f"Total entities in collection: {collection.num_entities}")

if __name__ == "__main__":
    main()